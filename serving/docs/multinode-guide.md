# Serving Kimi K2.7 Code on 16×H100

Background notes for the model in `multi-node/models/`. The launch commands below are the plain
SGLang cookbook form; `multi-node/serve.sbatch` builds the same kind of command (inside apptainer,
one task per node) from the model files. Sizes and layouts were checked on 2026-07-04, re-check the
cookbooks before relying on them.

**Cluster:** 2 nodes × 8× NVIDIA H100 80GB (SXM assumed) = 1,280 GB total HBM, sm90 (Hopper).
**Workload profile:** batch serving, latency-tolerant, high utilization and stability prioritized.
**Precision ceiling:** FP8 (per requirement). Sources: SGLang `docs_new/cookbook/autoregressive` (Kimi-K2.7-Code cookbook), `vllm-project/recipes` (Kimi-K2.5), inspected 2026-07-04.

---

## 0. Summary matrix

| Model | Checkpoint | Precision | Weights | Layout | KV headroom | Recipe status |
|---|---|---|---|---|---|---|
| Kimi K2.7 Code | `moonshotai/Kimi-K2.7-Code` | **Native INT4 (only release)** | ~600 GB | TP16 across 2 nodes | ~500 GB | K2.6-compatible deployment; single-node needs ≥140 GB/GPU, so H100 requires multi-node |

One point that is easy to get wrong:

1. **Kimi K2.7 Code has no FP8 release.** The only published checkpoint is native INT4 (quantization-aware, K2 lineage, Modified MIT license). INT4 *is* the ceiling precision for this model everywhere. A "native FP8 ~1 TB" figure would refer to a hypothetical dequantized form that does not exist as a checkpoint.

---

## 1. Prerequisites (both nodes)

### 1.1 Interconnect — the critical dependency

TP16 across nodes puts every all-reduce on the inter-node fabric.

- Required: RDMA — InfiniBand (NDR/HDR) or RoCE v2, ideally ≥400 Gb/s aggregate. GPUDirect RDMA enabled.
- Verify before anything else:
  ```bash
  ibstat                       # links up, rate as expected
  nvidia-smi topo -m           # NICs adjacent to GPUs (PIX/PXB, not SYS)
  # NCCL bandwidth sanity across both nodes:
  # all_reduce_perf -b 8 -e 1G -f 2 -g 1  (via nccl-tests, mpirun across 16 ranks)
  ```
- If you only have Ethernet/TCP between nodes: do **not** run TP16. Use pipeline parallelism between nodes (TP8 intra-node + PP2) — see §5.3. Latency worsens; acceptable for your batch profile.

Recommended NCCL env (both nodes, adjust HCA names to yours):

```bash
export NCCL_IB_HCA=mlx5           # or explicit list: mlx5_0,mlx5_1,...
export NCCL_IB_GID_INDEX=3        # RoCE v2 only
export NCCL_SOCKET_IFNAME=<mgmt-if>   # bootstrap interface
export NCCL_DEBUG=INFO            # first launches only; remove for production
```

### 1.2 Software

Framework: **SGLang** primary (the cookbook lives there); a vLLM alternative is noted below.

```bash
apptainer pull sglang_<tag>.sif docker://lmsysorg/sglang:<tag>
```

Pin the image tag once validated, do not track `:latest`.

Kimi K2.7 Code: `transformers>=4.57.1,<5.0.0` (model card requirement).

### 1.3 Weights

Pre-download to identical local paths on all nodes (or a shared parallel FS, which is what `MODEL_DIR` assumes). Size: ~600 GB (Kimi INT4).

```bash
huggingface-cli download moonshotai/Kimi-K2.7-Code --local-dir /models/Kimi-K2.7-Code
```

### 1.4 Multi-node launch mechanics (SGLang)

Identical command on both nodes except `--node-rank`:

```bash
--nnodes 2 --node-rank 0 --dist-init-addr <NODE0_IP>:5000   # node 0
--nnodes 2 --node-rank 1 --dist-init-addr <NODE0_IP>:5000   # node 1
```

The HTTP API is served from node 0. The host `/dev` is visible already: do **not** `--bind /dev/infiniband` (the explicit bind is mounted nodev and every `open()` of a uverbs device fails with EACCES, NCCL silently falls back to sockets); raise the memlock rlimit instead (`ulimit -l unlimited`, done in `multi-node/lib.sh`). Without IB exposure, comms silently fall back to TCP — the cookbook flags this failure mode explicitly for the PD-disaggregation path, and it applies equally here.

---

## 2. Kimi K2.7 Code — native INT4, TP16

**Basis:** SGLang Kimi-K2.7-Code cookbook. Architecture identical to K2.5/K2.6 (MLA attention, 64 heads; MoonViT 400M vision tower; 256K context; forces thinking mode — instant mode unsupported). Cookbook single-node configs require ≥140 GB/GPU (H200/B300 TP8, GB300 TP4); H100 80 GB forces the multi-node layout.

### 2.1 Launch (per node)

Cookbook generator base command, extended to TP16 multi-node, DP-attention enabled (the cookbook's stated production-throughput option):

```bash
python3 -m sglang.launch_server \
  --model-path /models/Kimi-K2.7-Code \
  --served-model-name kimi-k2.7-code \
  --tp 16 \
  --nnodes 2 --node-rank $RANK --dist-init-addr $NODE0_IP:5000 \
  --dp 16 --enable-dp-attention \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

### 2.2 Notes

- **INT4 is not a compromise here.** It is the release format (quantization-aware trained); there is nothing higher-precision to serve. This also makes Kimi the lightest fit on the cluster: ~37 GB weights/GPU, biggest KV headroom of the three.
- **Head divisibility:** 64 attention heads / TP16 = 4 heads per rank — valid on NVIDIA. (The cookbook's `heads_per_gpu % 16 == 0` constraint is AITER/AMD-specific.)
- **Context:** 256K supported. With ~40 GB/GPU KV headroom you can serve full context with deep batches; use `--context-length 128000` only if you want more concurrent requests than context.
- **Utilization caveat:** at TP16 the INT4 weights are thin per GPU; per-GPU compute utilization will be lower than the other two models. If Kimi becomes a permanent resident, a TP8+PP2 layout or even dedicating a future ≥140 GB/GPU node is more efficient — but TP16 is the correct config on this hardware.
- **vLLM alternative:** K2.5 recipe pattern applies (same architecture): `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --enable-auto-tool-choice --mm-encoder-tp-mode data --trust-remote-code`, multi-node via `--master-addr/--nnodes/--node-rank` (`--headless` on the follower).

---

## 5. Cluster operations

### 5.1 One model at a time

The layout uses all 16 GPUs. Serve one model per deployment window; swap by killing the server and launching the next (weights pre-staged per §1.3 make swaps a load-time cost only, ~5–15 min each at NVMe/parallel-FS speeds).

### 5.2 Rollout

Launch → single-request smoke test → GSM8K-class accuracy spot check (`sgl-eval run gsm8k --base-url ...`) → ramp batch depth while watching `--enable-metrics` output (KV usage, running requests, token throughput) → pin image tag + full command in version control.

### 5.3 If inter-node RDMA is unavailable

Replace TP16 with TP8 intra-node + pipeline parallelism across nodes: `--tp 8 --pp 2` (SGLang PP support varies by model — verify per model in your pinned version; vLLM: `-tp 8 -pp 2`). All-reduce stays on NVLink; only activations cross nodes. Throughput drops moderately; for latency-tolerant batch this is a legitimate permanent configuration, not just a fallback.

### 5.4 Known failure modes to watch

| Symptom | Likely cause | Action |
|---|---|---|
| Launch hangs at init on both nodes | NCCL bootstrap can't traverse; wrong `NCCL_SOCKET_IFNAME` or firewall on `--dist-init-addr` port | Fix interface/port; `NCCL_DEBUG=INFO` |
| Throughput far below single-node ×2 expectations | Comms fell back to TCP (IB not exposed in container) | Expose `/dev/infiniband`, memlock ulimit |

### 5.5 What not to carry over from sm120 (RTX PRO 6000 class) debugging

None of the sm120 workarounds apply on sm90: no `SGLANG_ENABLE_JIT_DEEPGEMM=0` / `SGLANG_ENABLE_DEEP_GEMM=0` (DeepGEMM is wanted here), no tilelang DSA override, no forced `--kv-cache-dtype bfloat16` (auto-selection is correct on Hopper), no NVFP4 anything (no FP4 hardware on H100). Clean-slate launch commands only.

---

## 6. Source map

| Claim | Source |
|---|---|
| K2.7 INT4-only release, ≥140 GB/GPU single-node, parsers, transformers pin, 256K context, DP-attention guidance | SGLang `Moonshotai/Kimi-K2.7-Code.mdx` + `kimi-k27-code-deployment.jsx` |
| K2.5 vLLM multi-node mechanics (`--master-addr/--nnodes/--node-rank/--headless`) | `vllm-project/recipes/moonshotai/Kimi-K2.5.md` |

Re-check both repos before deployment: TP16 on H100 may gain a verified recipe. Adopt it over the adaptations here.
