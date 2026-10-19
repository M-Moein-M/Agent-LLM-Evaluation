"""Pass check template. Copy to checks/<problem>/check.py and edit.

The checker never compiles or runs anything. It reads the evidence the container wrote
to <round>/.eval/ (facts.json with build/run exit codes, run.log with the program's
output) and returns ordered stages. A failing `fatal` stage skips the rest.

    build -> run -> mse -> metric

The class must be named `Check`. checks/check_pass.py finds it by problem name, and
host.sh runs it after every round (re-run any time to recompute verdicts without
rerunning agents: python3 checks/check_pass.py --traces traces).
"""

from eval_lib import ProblemCheck, StageResult


class Check(ProblemCheck):
    # --- tunable thresholds -------------------------------------------------
    MSE_MAX = 1e-9
    # For GPU problems add a stage `return self.gpu_stage(40.0)` to reject CPU-only solutions.

    def stages(self):
        return [self.build, self.run, self.mse, self.metric]

    def build(self) -> StageResult:
        return self.exit_stage("build", self.facts()["build"]["status"], fatal=True)

    def run(self) -> StageResult:
        return self.exit_stage("run", self.facts()["run"]["status"], fatal=True)

    def mse(self) -> StageResult:
        v = self.log_value(r"Mean Squared Error:\s*([0-9.eE+-]+)")
        passed = v is not None and v <= self.MSE_MAX
        detail = ("no 'Mean Squared Error' line in run.log" if v is None
                  else f"error {v:g} <= {self.MSE_MAX:g}")
        return StageResult("mse", passed, v, detail)

    def metric(self) -> StageResult:
        # Non-gating ranking number (it never flips pass/fail). Shown by tools/summarize.py
        # and collect_results.py. higher_better=False: lower time ranks first.
        v = self.log_value(r"Time:\s*([0-9.eE+-]+)\s*ms")
        return self.metric_stage("time", v, "ms", higher_better=False)
