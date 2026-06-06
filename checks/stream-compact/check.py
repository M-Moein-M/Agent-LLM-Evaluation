"""Pass check for the stream-compact problem.

Stages (in order, build/run are fatal):
    build → run → mse → gpu

The agent implements order-preserving stream compaction. The fixed verifier in
src/main.cu runs it many times against a CPU reference and reports the
worst-case error as a single ``Mean Squared Error`` line. A correct,
order-preserving implementation scores exactly 0 on every repetition, so the
threshold is set tight — any reordering, wrong count, or nondeterministic run
produces a non-zero worst case. The ``gpu`` stage guards against a CPU-only
implementation.

Tune the thresholds below; re-running ``checks/check_pass.py`` recomputes
verdicts over existing traces with no GPU re-run.
"""

from eval_lib import ProblemCheck, StageResult


class Check(ProblemCheck):
    # --- tunable thresholds -------------------------------------------------
    # A correct solution yields an exact 0 worst-case error, so this only needs
    # to sit above floating noise (there is none here — values are integers).
    MSE_MAX = 1e-9
    GPU_MIN_AVG = 15.0    # mean utilization.gpu (%) over the run
    GPU_MIN_PEAK = 40.0   # peak utilization.gpu (%) — guards against CPU fallback

    def stages(self):
        return [self.build, self.run, self.mse, self.gpu]

    def build(self) -> StageResult:
        return self.exit_stage("build", self.facts()["build"]["status"], fatal=True)

    def run(self) -> StageResult:
        return self.exit_stage("run", self.facts()["run"]["status"], fatal=True)

    def mse(self) -> StageResult:
        v = self.log_value(r"Mean Squared Error:\s*([0-9.eE+-]+)")
        passed = v is not None and v <= self.MSE_MAX
        if v is None:
            detail = "no 'Mean Squared Error' line found in run.log"
        else:
            detail = f"worst-case error {v:g} <= {self.MSE_MAX:g}"
        return StageResult("mse", passed, v, detail)

    def gpu(self) -> StageResult:
        g = self.gpu_stats()
        passed = (
            g["samples"] > 0
            and g["avg"] >= self.GPU_MIN_AVG
            and g["peak"] >= self.GPU_MIN_PEAK
        )
        detail = (
            f"avg {g['avg']:.1f}% (>= {self.GPU_MIN_AVG}), "
            f"peak {g['peak']:.0f}% (>= {self.GPU_MIN_PEAK}), "
            f"{g['samples']} samples"
        )
        return StageResult("gpu", passed, g, detail)
