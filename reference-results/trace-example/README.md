# Sample traces

One round per agent (Claude Code, Kimi Code, pi, mini-swe-agent), all `tiled-gemm` with
`DeepSeek-V4-Flash-0731`. Each round passed. Layout: `traces/<problem>/<agent>/<model>/<campaign>/round-N/`.
A round holds the agent's trace, `solution.json`, `changed_diff.txt` and the checker output in `.eval/`.

Browse them:

```bash
tools/trace-vis/run.sh -w reference-results/trace-example --scan-fs
```

## Full traces

The agent sessions behind the pass@1 matrix are on Hugging Face:
[m-moein-m/Agent-LLM-Evaluation-Traces](https://huggingface.co/datasets/m-moein-m/Agent-LLM-Evaluation-Traces).
