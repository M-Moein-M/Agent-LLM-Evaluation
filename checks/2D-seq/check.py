"""Pass check for the 2D-seq problem (2D Poisson / SOR, CPU).

Stages (in order, build/run are fatal):
    build → run → integrity → digest → metric

The workload is a *fixed* number of SOR sweeps (``poisson.par`` sets
``eps 0.0``, so the residual test never fires). Every variant therefore does
exactly the same arithmetic and the only free variable is wall time.

Correctness is graded on the ``DIGEST`` line that ``src/verify.c`` prints after
the timed region — five reductions over the interior field. Because SOR is a
contraction, honest floating-point differences (vectorization, reassociation,
``-ffast-math``, a different ISA) stay around 1e-15 relative, while a change of
*sweep ordering* — red/black being the obvious one — moves the digest by
~6.4e-7 at this grid and sweep count. TOL_REL = 1e-9 sits six orders above the
noise and three below the red/black signal (verified: red/black trips l2, l1,
min and max at 5e-7..6.4e-7).

``osc`` is a cancelling alternating sum, so a relative test on it is
meaningless; it is compared with an absolute tolerance scaled by the reference
``l1`` instead. At 300 sweeps both orderings leave ``osc`` near zero, so it is
a redundant guard here rather than the discriminating field.

Reference values live in ``reference.json`` next to this file. Regenerate them
by building the pristine problem and running ``./exe-GCC poisson.par``.

Tune the thresholds below; re-running ``checks/check_pass.py`` recomputes
verdicts over existing traces with no re-run of the workload.
"""

import json
import math
from pathlib import Path

from eval_lib import ProblemCheck, StageResult

_REFERENCE = json.loads((Path(__file__).parent / "reference.json").read_text())

# Files the agent is told not to touch. host.sh rsyncs the work dir into the
# trace with --compare-dest against the pristine problem, so a file only lands
# in the trace if it differs from the original. Presence == modified.
#
# src/main.c is frozen because it owns BOTH ends of the measurement: it brackets
# solve() with the timer and prints the SOLVE_TIME line this check ranks on. An
# editable driver lets a run shrink the timed region or hoist sweeps into setup
# and still produce a correct DIGEST — i.e. a better score for no faster solver.
# Everything that legitimately needs changing (solver, allocation, build) is in
# other files.
_FROZEN = ("poisson.par", "src/main.c", "src/verify.c", "src/verify.h")


class Check(ProblemCheck):
    # --- tunable thresholds -------------------------------------------------
    TOL_REL = 1e-9        # relative tolerance on l2, l1, min, max
    TOL_OSC_REL_L1 = 1e-9  # absolute tolerance on osc, in units of reference l1

    def stages(self):
        return [self.build, self.run, self.integrity, self.digest, self.metric]

    def build(self) -> StageResult:
        return self.exit_stage("build", self.facts()["build"]["status"], fatal=True)

    def run(self) -> StageResult:
        return self.exit_stage("run", self.facts()["run"]["status"], fatal=True)

    def integrity(self) -> StageResult:
        round_dir = self.eval_dir.parent
        touched = [f for f in _FROZEN if (round_dir / f).exists()]
        if touched:
            return StageResult(
                "integrity", False, touched,
                f"frozen file(s) modified: {', '.join(touched)}",
            )
        return StageResult("integrity", True, [], "frozen files unmodified")

    def _digest(self) -> dict:
        keys = ("l2", "l1", "osc", "min", "max")
        return {
            k: self.log_value(rf"DIGEST\b.*?\b{k}=([0-9.eE+-]+)")
            for k in keys
        }

    def digest(self) -> StageResult:
        got = self._digest()
        missing = [k for k, v in got.items() if v is None]
        if missing:
            return StageResult(
                "digest", False, got,
                f"no usable DIGEST line in run.log (missing: {', '.join(missing)})",
            )

        osc_tol = self.TOL_OSC_REL_L1 * abs(_REFERENCE["l1"])
        bad = []
        for k, v in got.items():
            ref = _REFERENCE[k]
            if k == "osc":
                ok = abs(v - ref) <= osc_tol
                err = abs(v - ref)
            else:
                err = abs(v - ref) / abs(ref) if ref else abs(v - ref)
                ok = err <= self.TOL_REL
            if not (ok and math.isfinite(v)):
                bad.append(f"{k}={v:.9e} (ref {ref:.9e}, err {err:.2e})")

        if bad:
            return StageResult("digest", False, got, "; ".join(bad))
        return StageResult(
            "digest", True, got,
            f"all 5 digest fields within {self.TOL_REL:g} of reference",
        )

    def metric(self) -> StageResult:
        # Ranking measurement (non-gating): solve() wall time in seconds, from
        # the SOLVE_TIME line frozen src/main.c prints around the timed region.
        # The tag matters: log_value takes the FIRST re.search match, so an
        # untagged "<number>s" pattern would be hijacked by any timing the
        # agent's own code happens to print earlier in run.log.
        v = self.log_value(r"SOLVE_TIME\s+([0-9.]+)s")
        return self.metric_stage("solve_time", v, "s", higher_better=False)
