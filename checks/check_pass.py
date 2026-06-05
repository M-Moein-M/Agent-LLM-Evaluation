#!/usr/bin/env python3
"""Host-side pass-check driver.

Walks the trace tree for every round (a dir containing ``.eval/facts.json``),
loads that problem's ``checks/<PROBLEM>/check.py``, runs it against the round's
``.eval/`` evidence, and writes ``.eval/pass_result.json``.

Pure stdlib, no Docker, no toolchain — it only reads the evidence files captured
by the container. Idempotent: re-run any time after editing a threshold to
recompute verdicts with zero GPU cost.

    python checks/check_pass.py --traces traces
    python checks/check_pass.py --traces traces --problems RabbitCT
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="traces", help="Trace root directory")
    parser.add_argument(
        "--checks-root",
        default=None,
        help="Directory holding <PROBLEM>/check.py (default: this script's own directory)",
    )
    parser.add_argument(
        "--problems",
        default=None,
        help="Comma-separated problem names to limit to (default: all found)",
    )
    return parser.parse_args()


def find_rounds(traces_root: Path):
    """Yield (problem_name, eval_dir) for every round under the trace tree.

    Trace layout: traces/<PROBLEM>/<MODEL>/<DATE>/round-N/.eval/facts.json
    The problem name is the path segment directly under the trace root.
    """
    for facts in traces_root.glob("**/.eval/facts.json"):
        eval_dir = facts.parent
        rel = eval_dir.relative_to(traces_root)
        if not rel.parts:
            continue
        problem = rel.parts[0]
        yield problem, eval_dir


_CHECK_CLASS_CACHE: dict[str, type] = {}


def load_check_class(checks_root: Path, problem: str):
    """Import ``checks/<problem>/check.py`` and return its ``Check`` class.

    ``checks_root`` is placed on sys.path so the per-problem module's
    ``from eval_lib import ...`` resolves to the shared library.
    """
    if problem in _CHECK_CLASS_CACHE:
        return _CHECK_CLASS_CACHE[problem]

    module_path = checks_root / problem / "check.py"
    if not module_path.exists():
        _CHECK_CLASS_CACHE[problem] = None
        return None

    root_str = str(checks_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    spec = importlib.util.spec_from_file_location(f"checks_{problem}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_cls = getattr(module, "Check", None)
    _CHECK_CLASS_CACHE[problem] = check_cls
    return check_cls


def main():
    args = parse_args()
    traces_root = Path(args.traces)
    # Per-problem checks live next to this script (checks/<PROBLEM>/check.py),
    # so default to its own directory — robust to the caller's CWD.
    checks_root = Path(args.checks_root) if args.checks_root else Path(__file__).resolve().parent
    only = set(args.problems.split(",")) if args.problems else None

    if not traces_root.exists():
        print(f"ERROR: traces root not found: {traces_root}", file=sys.stderr)
        return 1

    checked = 0
    skipped = 0
    for problem, eval_dir in find_rounds(traces_root):
        if only and problem not in only:
            continue

        check_cls = load_check_class(checks_root, problem)
        if check_cls is None:
            print(f"WARN: no checks/{problem}/check.py — skipping {eval_dir}", file=sys.stderr)
            skipped += 1
            continue

        verdict = check_cls(eval_dir).evaluate()
        out = eval_dir / "pass_result.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(verdict.to_dict(), fh, indent=2)

        status = "PASS" if verdict.passed else "FAIL"
        failed = [s.name for s in verdict.stages if not s.passed and not s.skipped]
        tail = f" ({', '.join(failed)})" if failed else ""
        print(f"[{status}]{tail} {eval_dir}")
        checked += 1

    print(f"\nChecked {checked} round(s); skipped {skipped}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
