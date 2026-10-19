# claude-pro — Claude Code on a personal subscription

Same agent as `agents/apptainer/claude/`, but authenticated with a **personal
Claude subscription** (OAuth) against `api.anthropic.com` instead of the
gateway. It is a full fork so the gateway agent stays untouched.

Everything lives in this directory — nothing is written to `$HOME`.
`agent.sif`, `.secrets/`, `.setup-home/`, `.cache/` and `.work/` are gitignored.

## 1. Build the image (once)

```bash
cd <repo root>
APPTAINER_CACHEDIR=$PWD/agents/apptainer/claude-pro/.cache \
apptainer build --fakeroot \
  agents/apptainer/claude-pro/agent.sif agents/apptainer/claude-pro/agent.def
```

The recipe is identical to the gateway agent's, so if you already built
`agents/apptainer/claude/agent.sif` you can just link it instead:

```bash
ln -s ../claude/agent.sif agents/apptainer/claude-pro/agent.sif
```

## 2. Mint the subscription token (once, interactive)

`claude setup-token` runs the OAuth flow headlessly: it prints a URL, you open
it in your browser, paste the code back, and it prints an `sk-ant-oat01-…` token.
Needs a TTY and outbound HTTPS — run it on a login node, not in a batch job.

```bash
mkdir -p agents/apptainer/claude-pro/.setup-home agents/apptainer/claude-pro/.secrets
chmod 700 agents/apptainer/claude-pro/.secrets

apptainer exec --home $PWD/agents/apptainer/claude-pro/.setup-home \
  agents/apptainer/claude-pro/agent.sif claude setup-token
```

`--home` points at this directory's own staging home, **not** `:/root` — the
`claude` binary lives under `/root/.local`, and binding over `/root` shadows it.

Store the printed token, readable only by you:

```bash
( umask 077; printf '%s\n' 'sk-ant-oat01-...' \
    > agents/apptainer/claude-pro/.secrets/oauth_token )
```

This is a long-lived credential for your Anthropic account, sitting on a
group-readable workspace filesystem. `.secrets/` is chmod 700 and gitignored, but
check `getfacl` on the repo if others share the group — and revoke the token at
console.anthropic.com if it ever leaks.

## 3. Run

```bash
EVAL_MODELS=claude-opus-4-8 EVAL_ROUNDS=1 EVAL_SOLVE_PAR=1 \
  ./agents/apptainer/claude-pro/host.sh
```

Traces land in `traces/<problem>/claude-pro/<model>/<date>/round-N/`, the same
layout `checks/check_pass.py` and `checks/trace_stats.py` already consume.

## Before you scale it up

- **Rate limits.** A subscription has per-account limits; `EVAL_SOLVE_PAR` defaults
  to 1 for that reason. A 429 mid-solve is not retried — it just becomes a failed
  round in the results. Watch `agent.stderr.log` on the first wave.
- **Comparability.** These runs are not measuring the same thing as the gateway
  models (different model, different quota, different context window). Fine as a
  reference ceiling; footnote it if it shares a table with the gateway rows.
