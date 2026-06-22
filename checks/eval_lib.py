"""Shared infrastructure for host-side pass checks.

A check reads only the evidence files captured by the container during a run
(``facts.json``, ``run.log``, ``gpu_usage.csv``) — it never compiles or executes
anything. Each problem subclasses :class:`ProblemCheck` and implements
:meth:`ProblemCheck.stages`, returning an ordered list of stage methods. The base
:meth:`ProblemCheck.run` executes them with fatal short-circuit and aggregates a
:class:`Verdict`.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class StageResult:
    """Outcome of one check stage.

    ``value`` carries the extracted measurement (e.g. mse=0.0 or the gpu stats
    dict) so the verdict is self-documenting. ``fatal`` marks a stage whose
    failure makes downstream stages meaningless (e.g. build → run). ``skipped``
    is set by the runner when an earlier fatal stage already failed.
    """

    name: str
    passed: bool
    value: Any = None
    detail: str = ""
    fatal: bool = False
    skipped: bool = False


@dataclass
class Verdict:
    passed: bool
    stages: list[StageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "stages": [asdict(s) for s in self.stages],
        }


class ProblemCheck:
    """Base class for per-problem checks.

    Subclasses set threshold constants and implement :meth:`stages`. The helper
    accessors below hide all raw parsing so problem authors work with values,
    not text.
    """

    def __init__(self, eval_dir: str | Path):
        self.eval_dir = Path(eval_dir)
        self._facts: Optional[dict] = None

    # -- evidence accessors --------------------------------------------------

    def facts(self) -> dict:
        """Load and cache ``facts.json`` (exit codes, timings, cmds)."""
        if self._facts is None:
            # utf-8-sig tolerates a stray BOM; bash writes none, but be defensive.
            with open(self.eval_dir / "facts.json", "r", encoding="utf-8-sig") as fh:
                self._facts = json.load(fh)
        return self._facts

    def log_value(self, pattern: str, file: str = "run.log") -> Optional[float]:
        """Return the first capture group of ``pattern`` in ``file`` as a float.

        Replaces the old awk/gsub extraction: ``"0"`` parses to ``0.0`` (not an
        empty string), and a missing match / unparseable value returns ``None``.
        """
        path = self.eval_dir / file
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None

    def gpu_stats(self, file: str = "gpu_usage.csv") -> dict:
        """Aggregate GPU utilization from the nvidia-smi csv.

        Returns ``{"avg", "peak", "samples"}``. Tolerates trailing-space headers
        (``utilization.gpu [%]``), ``[N/A]`` cells, and a missing/empty file
        (returns zeros so a check can still fail cleanly rather than crash).
        """
        path = self.eval_dir / file
        empty = {"avg": 0.0, "peak": 0.0, "samples": 0}
        if not path.exists():
            return empty

        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return empty

            col = self._find_column(header, "utilization.gpu")
            if col is None:
                return empty

            values: list[float] = []
            for row in reader:
                if len(row) <= col:
                    continue
                # Leading number; tolerates a trailing unit ("98 %") from older
                # non-nounits captures. Skips [N/A], repeated headers, blanks.
                m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", row[col])
                if m:
                    values.append(float(m.group(1)))

        if not values:
            return empty
        return {
            "avg": sum(values) / len(values),
            "peak": max(values),
            "samples": len(values),
        }

    @staticmethod
    def _find_column(header: list[str], name: str) -> Optional[int]:
        """Index of the header cell whose normalized name starts with ``name``.

        nvidia-smi labels the column ``utilization.gpu`` (csv,nounits) or
        ``utilization.gpu [%]``; match the prefix, ignoring case/whitespace.
        """
        name = name.lower()
        for i, cell in enumerate(header):
            if cell.strip().lower().startswith(name):
                return i
        return None

    # -- stage helpers -------------------------------------------------------

    def exit_stage(self, name: str, status: Any, fatal: bool = False) -> StageResult:
        """Build a StageResult from a ``facts.json`` exit code (0 == pass)."""
        passed = status == 0
        return StageResult(
            name=name,
            passed=passed,
            value=status,
            detail=f"exit code {status}",
            fatal=fatal,
        )

    # -- runner --------------------------------------------------------------

    def stages(self) -> list[Callable[[], StageResult]]:
        raise NotImplementedError

    def evaluate(self) -> Verdict:
        """Execute stages in order with fatal short-circuit.

        Named ``evaluate`` (not ``run``) so a problem is free to define a ``run``
        *stage* method without shadowing the runner.

        A failing fatal stage stops execution; remaining stages are recorded as
        ``skipped``. Overall pass requires every non-skipped stage to pass and no
        fatal break.
        """
        results: list[StageResult] = []
        stages = self.stages()
        broke = False
        for i, stage in enumerate(stages):
            result = stage()
            results.append(result)
            if result.fatal and not result.passed:
                # Mark the rest as skipped without executing them.
                for skipped_stage in stages[i + 1:]:
                    results.append(
                        StageResult(
                            name=skipped_stage.__name__,
                            passed=False,
                            detail="skipped (earlier fatal stage failed)",
                            skipped=True,
                        )
                    )
                broke = True
                break

        passed = not broke and all(
            s.passed for s in results if not s.skipped
        )
        return Verdict(passed=passed, stages=results)
