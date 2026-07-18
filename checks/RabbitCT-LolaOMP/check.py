"""Pass check for the RabbitCT-LolaOMP problem.

Stages (in order, build/run are fatal):
    build → run → mse → gpu

Same contract as RabbitCT-LolaBunny: the in-binary verifier prints
``Mean Squared Error: <x> HU^2`` when run with ``-c``, and nvidia-smi sampling
captured by the harness must show real GPU utilization (guards against a CPU
fallback passing the correctness bar). The GPU port must match the LolaBunny
baseline against ``Reference256.vol``; line-range clipping only skips voxels
whose rays miss the detector (zero contribution), so the result is numerically
equivalent and the 1e-6 MSE bar is the same as LolaBunny.

Tune the thresholds below; re-running ``scripts/check_pass.py`` recomputes
verdicts over existing traces with no GPU re-run.
"""

from eval_lib import ProblemCheck, StageResult


class Check(ProblemCheck):
    # --- tunable thresholds -------------------------------------------------
    MSE_MAX = 1e-6        # run.log "Mean Squared Error" must be <= this
    GPU_MIN_AVG = 20.0    # mean utilization.gpu (%) over the run
    GPU_MIN_PEAK = 50.0   # peak utilization.gpu (%) — guards against CPU fallback

    def stages(self):
        return [self.build, self.run, self.mse, self.gpu, self.metric]

    def metric(self) -> StageResult:
        # Ranking measurement (non-gating): reconstruction error against the
        # reference volume, same as LolaBunny. Compared only WITHIN this table.
        v = self.log_value(r"Mean Squared Error:\s*([0-9.eE+-]+)")
        return self.metric_stage("mse", v, "HU^2", higher_better=False)

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
            detail = f"mse {v:g} <= {self.MSE_MAX:g}"
        return StageResult("mse", passed, v, detail)

    def gpu(self) -> StageResult:
        # Gate on peak only; avg is reported but not gated. See
        # ProblemCheck.gpu_stage for the rationale (correct GPU kernels with a
        # long non-GPU tail have low avg yet high peak).
        return self.gpu_stage(self.GPU_MIN_PEAK, self.GPU_MIN_AVG)
