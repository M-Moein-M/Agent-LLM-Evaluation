"""Pass check for the tiled-gemm problem.

Stages (in order, build/run are fatal):
    build → run → mse → gpu → perf

tiled-gemm is an OPTIMIZATION task: correctness is the floor (near-zero
worst-case MSE across the verifier's fixed size sweep, which mixes tile
multiples and non-multiples), and the graded goal is throughput.

The throughput grade does NOT come from the agent's code. At measure time the
harness compiles its own ``checks/tiled-gemm/bench.cu`` against the agent's
kernel, runs the agent kernel AND cuBLAS on one fixed, identical workload, and
times both with nsys (``.eval/kern_agent.csv`` / ``.eval/kern_cublas.csv``;
cudaEvent fallback in ``.eval/perf.log``). The ``perf`` stage grades on the
RATIO — agent throughput as a fraction of cuBLAS on the same device. That ratio
assumes no operation count (any 2*N^3 constant cancels) and self-calibrates to
whatever GPU runs it, so there is no brittle per-GPU GFLOP/s number. It also
re-checks correctness at the graded size (bench MSE vs cuBLAS). The ``gpu`` stage
guards against a CPU-only re-implementation.

Tune the thresholds below; re-running ``checks/check_pass.py`` recomputes
verdicts over existing traces with no GPU re-run.
"""

import csv
import re

from eval_lib import ProblemCheck, StageResult


class Check(ProblemCheck):
    # --- tunable thresholds -------------------------------------------------
    MSE_MAX = 1e-2          # worst-case MSE in run.log must be <= this
    GPU_MIN_AVG = 15.0      # mean utilization.gpu (%) over the run
    GPU_MIN_PEAK = 40.0     # peak utilization.gpu (%) — guards against CPU fallback
    BENCH_MSE_MAX = 1e-2    # agent-vs-cuBLAS MSE at the graded size (bench.cu)
    # Minimum throughput as a FRACTION of cuBLAS on the same workload+device
    # (1.0 == matches cuBLAS). PROVISIONAL floor that rejects naive/unoptimized
    # kernels; being a ratio it is GPU-independent (unlike the old fixed GFLOP/s
    # threshold). Tune against real runs.
    PERF_MIN_FRAC = 0.15
    # Paired bench (.eval/perf_paired.log): cuBLAS is timed before and after the
    # agent kernel. A spread (max/min - 1) of the cuBLAS segments above this
    # marks the round "contended" in the detail. Reported, never gated.
    PAIRED_MAX_SPREAD = 0.05

    def stages(self):
        return [self.build, self.run, self.mse, self.gpu, self.perf, self.metric]

    def metric(self) -> StageResult:
        # Ranking measurement (non-gating): the harness bench's agent-vs-cuBLAS
        # kernel-time ratio — the same number the perf stage gates on, but kept
        # as a rankable scalar. Preferred over run.log's self-reported GFLOP/s
        # because the bench is harness-owned (bound read-only at /grade) and the
        # ratio self-calibrates to the device.
        v = self._cublas_frac()[0]
        return self.metric_stage("cublas_frac", v, "x cuBLAS", higher_better=True)

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
            detail = f"mse {v:g} {'<=' if passed else '>'} {self.MSE_MAX:g}"
        return StageResult("mse", passed, v, detail)

    def gpu(self) -> StageResult:
        # Gate on peak only; avg is reported but not gated. See
        # ProblemCheck.gpu_stage for the rationale (correct GPU kernels with a
        # long non-GPU tail have low avg yet high peak).
        return self.gpu_stage(self.GPU_MIN_PEAK, self.GPU_MIN_AVG)

    # -- perf: agent throughput as a fraction of cuBLAS on a fixed workload --

    def _kernel_total_ns(self, filename: str):
        """Sum 'Total Time (ns)' over all kernels in an nsys cuda_gpu_kern_sum
        CSV. Returns None if the file is missing/empty/unparseable (so the perf
        stage can fall back to the cudaEvent timing)."""
        path = self.eval_dir / filename
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return None
            col = self._find_column(header, "total time")
            if col is None:
                return None
            total = 0.0
            seen = False
            for row in reader:
                if len(row) <= col:
                    continue
                cell = row[col].replace(",", "").strip()
                try:
                    total += float(cell)
                    seen = True
                except ValueError:
                    continue
        return total if seen else None

    def _paired(self):
        """Paired bench result: ``(agent_ms, cublas_ms, spread)`` or None.

        agent_ms is the minimum over the repetitions, cublas_ms the minimum over
        all cuBLAS segments, spread is max/min - 1 over the cuBLAS segments.
        """
        path = self.eval_dir / "perf_paired.log"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        reps = re.findall(r"BENCH paired rep=\d+ cublas_pre=([0-9.eE+-]+) "
                          r"agent=([0-9.eE+-]+) cublas_post=([0-9.eE+-]+)", text)
        try:
            agent = [float(r[1]) for r in reps]
            cublas = [float(x) for r in reps for x in (r[0], r[2])]
        except ValueError:
            return None
        if not reps or min(agent) <= 0 or min(cublas) <= 0:
            return None
        return min(agent), min(cublas), max(cublas) / min(cublas) - 1.0

    def _cublas_frac(self):
        """Agent throughput as a fraction of cuBLAS: ``(frac, source)``.

        Prefers the paired bench when present, then nsys kernel time; falls back to the bench's cudaEvent ms_total
        if the nsys CSVs are absent. ``frac`` is None when neither is available
        (bench never built/ran). Shared by the ``perf`` gate and the ``metric``
        ranking stage so both always report the same number.
        """
        paired = self._paired()
        if paired is not None:
            agent_ms, cublas_ms, spread = paired
            source = "paired"
            if spread > self.PAIRED_MAX_SPREAD:
                source += f", contended: cuBLAS spread {spread * 100:.0f}%"
            return cublas_ms / agent_ms, source

        agent_ns = self._kernel_total_ns("kern_agent.csv")
        cublas_ns = self._kernel_total_ns("kern_cublas.csv")
        source = "nsys"
        if agent_ns is None or cublas_ns is None:
            agent_ns = self.log_value(r"BENCH agent .*ms_total=\s*([0-9.eE+-]+)",
                                      file="perf.log")
            cublas_ns = self.log_value(r"BENCH cublas .*ms_total=\s*([0-9.eE+-]+)",
                                       file="perf.log")
            source = "cudaEvent"

        if not agent_ns or not cublas_ns or agent_ns <= 0:
            return None, source
        return cublas_ns / agent_ns, source

    def perf(self) -> StageResult:
        # Correctness at the graded size: the bench compares the agent kernel to
        # cuBLAS on identical inputs. A missing line means the bench never ran
        # (build failed / nvcc or bench absent) — no perf credit.
        bench_mse = self.log_value(r"BENCH mse=\s*([0-9.eE+-]+)", file="perf.log")
        if bench_mse is None:
            return StageResult("perf", False, None,
                               "no perf bench result (.eval/perf.log) — bench "
                               "did not build/run")
        if bench_mse > self.BENCH_MSE_MAX:
            return StageResult("perf", False, bench_mse,
                               f"bench MSE {bench_mse:g} > {self.BENCH_MSE_MAX:g} "
                               "(kernel incorrect at graded size)")

        frac, source = self._cublas_frac()
        if frac is None:
            return StageResult("perf", False, None,
                               "perf timing unavailable (no nsys CSVs and no "
                               "BENCH ms_total in perf.log)")

        passed = frac >= self.PERF_MIN_FRAC
        detail = (f"{frac * 100:.1f}% of cuBLAS ({source}; "
                  f">= {self.PERF_MIN_FRAC * 100:.0f}%), bench MSE {bench_mse:g}")
        return StageResult("perf", passed, frac, detail)
