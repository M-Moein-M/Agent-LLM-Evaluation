"""Pass check for the RabbitCT-LolaBunny problem.

Stages (in order, build/run are fatal):
    build → run → mse → gpu

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
            detail = f"mse {v:g} <= {self.MSE_MAX:g}"
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
