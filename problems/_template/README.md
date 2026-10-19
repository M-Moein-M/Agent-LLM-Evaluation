# _template

Copy this directory to `problems/<name>/` and the checker in `checks/_template/` to
`checks/<name>/`, then replace the toy problem. What the harness needs from a problem:

| File | Purpose |
|---|---|
| `PROMPT.md` | The task the agent reads. Must end with the `solution.json` contract (build commands, run commands, `variant_name`). |
| `Makefile`, `src/` | Starter code that compiles. The agent edits it. Put the fixed verifier in a file the prompt forbids editing. |
| `download-input.sh` (optional) | Fetches bulk input into `data/<name>/` (never committed). `scripts/run_eval.sh` bind-mounts `data/<name>/RabbitInput` only for the RabbitCT family, so extend the data bind in the agent `host.sh` files if your problem needs one. |

The verifier must print the numbers the checker reads from `run.log`. This template prints
`Mean Squared Error: <v>` (gating) and `Time: <ms>` (ranking only).

The grader lives in `checks/<name>/`, outside this directory, because `problems/<name>/`
is copied into the agent's workspace and the agent must not see the grader.
