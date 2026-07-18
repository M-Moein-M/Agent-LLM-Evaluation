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
        return [self.build, self.run, self.mse, self.gpu, self.metric]

    def metric(self) -> StageResult:
        # Ranking measurement (non-gating): the verifier's worst-case error over
        # its repetitions. Compared only WITHIN this table; across problems,
        # pass@1 is the common axis.
        v = self.log_value(r"Mean Squared Error:\s*([0-9.eE+-]+)")
        return self.metric_stage("mse", v, "", higher_better=False)

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
        # Gate on peak only; avg is reported but not gated. See
        # ProblemCheck.gpu_stage for the rationale (correct GPU kernels with a
        # long non-GPU tail have low avg yet high peak).
        return self.gpu_stage(self.GPU_MIN_PEAK, self.GPU_MIN_AVG)
