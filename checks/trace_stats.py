#!/usr/bin/env python3
"""Host-side run statistics: tokens, iteration steps, and context compaction.

Walks the trace tree and, for every round (a dir holding ``.eval/facts.json``),
parses the agent's raw event/trajectory file that sits NEXT TO ``.eval/`` and
writes ``.eval/stats.json``.

Four agent formats are supported. The first three record per-model-call token
usage (so the same metrics come out the other side); kimi does not — see below.

  * **pi**  → ``agent.jsonl`` (one JSON event per line). Assistant turns are
    ``message_end`` events whose ``message.role == "assistant"``; the per-call
    usage lives at ``message.usage`` (``input`` / ``output`` / ``totalTokens``).
    Tool actions are ``tool_execution_start`` events.
  * **swe-mini** → ``agent.traj.json`` (one JSON object). Per-call usage lives at
    ``messages[].extra.response.usage`` (``prompt_tokens`` / ``completion_tokens``
    / ``total_tokens``); tool actions are ``messages[]`` with ``role == "tool"``.
  * **claude** → ``agent.stream.jsonl`` (Claude Code ``--output-format
    stream-json``, one JSON event per line). Assistant turns are ``type ==
    "assistant"`` events; per-call usage lives at ``message.usage``
    (``input_tokens`` + ``cache_read_input_tokens`` + ``cache_creation_input_tokens``
    is the effective prompt context, ``output_tokens`` the completion). Tool
    actions are ``tool_use`` content blocks inside those assistant messages.
  * **kimi** → ``agent.stream.jsonl`` TOO (Kimi Code CLI). Same filename as
    claude but a different schema: flat chat messages carrying a top-level
    ``role`` (``assistant`` / ``tool`` / ``meta``) and NO ``message`` wrapper.
    One ``assistant`` line == one model call; ``tool`` lines are tool results.
    Because the filename collides, the two are told apart by sniffing content
    (``_sniff_stream_agent``), never by name — dispatching on the name alone is
    what previously mislabelled every kimi round as ``agent: "claude"`` and
    reported 0 turns / 0 tokens for it.

    Kimi Code emits NO token usage anywhere in this stream, so ``total_input`` /
    ``total_output`` / ``total_tokens`` / ``peak_context`` are reported as
    ``None`` (unmeasurable), NOT 0 — a real 0 would silently drag wave averages
    down, whereas ``None`` is filtered out by the ``_mean`` helpers in
    collect_results.py / report_html.py and shows as blank. Recovering kimi
    token counts needs a harness change (the CLI would have to be made to log
    per-call usage); it cannot be back-filled from existing traces. For the same
    reason kimi compaction cannot be detected: with no context series there is
    nothing to spot a drop in, so ``compacted`` is always False for kimi and
    should be read as "unknown", not "confirmed no compaction".

Definitions, kept identical across agents so models are comparable:

  * ``turns``  — number of model calls (assistant responses). The "iteration
    steps" the agent took.
  * ``steps``  — number of tool executions performed.
  * ``total_input`` / ``total_output`` / ``total_tokens`` — summed over every
    model call (prompt tokens are re-sent each call, so ``total_input`` is the
    real billed prompt cost, not a single snapshot).
  * ``peak_context`` — high-water mark of the per-call context size.
  * ``compacted`` / ``compaction_points`` — context compaction is detected purely
    from the per-call context-size series: in a normal loop it grows
    monotonically (history is re-sent), so a sharp DROP means the runtime
    summarized/pruned the history. A "compaction point" is a turn index where the
    context shrank by more than ``DROP_FRAC`` vs. the previous call.

Pure stdlib, no Docker, no GPU. Idempotent — re-run any time:

    python checks/trace_stats.py --traces traces
    python checks/trace_stats.py --traces traces --problems tiled-gemm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# A per-call context drop larger than this fraction vs. the previous call is
# treated as a compaction event. Real compaction typically cuts context by far
# more than this (often >50%); the margin only guards against minor jitter.
DROP_FRAC = 0.2


def _compaction_points(context_series: list[int]) -> list[int]:
    """Indices in ``context_series`` where context dropped > DROP_FRAC."""
    points = []
    for i in range(1, len(context_series)):
        prev = context_series[i - 1]
        if prev > 0 and context_series[i] < prev * (1.0 - DROP_FRAC):
            points.append(i)
    return points


def _finish(agent, source, total_input, total_output, total_tokens,
            turns, steps, context_series, usage_available=True) -> dict:
    """Assemble a stats dict. ``usage_available=False`` marks an agent whose
    trace carries no token usage at all (kimi): the token and context fields
    then come out as ``None`` rather than a misleading 0."""
    points = _compaction_points(context_series)
    if not usage_available:
        total_input = total_output = total_tokens = None
        peak_context = None
    else:
        peak_context = max(context_series) if context_series else 0
    return {
        "agent": agent,
        "source": source,
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "turns": turns,
        "steps": steps,
        "peak_context": peak_context,
        "compacted": bool(points),
        "compaction_points": points,
    }


def parse_pi(path: Path) -> dict:
    """Parse a pi ``agent.jsonl`` event stream."""
    total_input = total_output = total_tokens = turns = steps = 0
    context_series: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "tool_execution_start":
                steps += 1
            elif etype == "message_end":
                msg = ev.get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                turns += 1
                usage = msg.get("usage") or {}
                inp = int(usage.get("input") or 0)
                out = int(usage.get("output") or 0)
                tot = int(usage.get("totalTokens") or (inp + out))
                total_input += inp
                total_output += out
                total_tokens += tot
                if inp:
                    context_series.append(inp)
    return _finish("pi", path.name, total_input, total_output, total_tokens,
                   turns, steps, context_series)


def parse_swe_mini(path: Path) -> dict:
    """Parse a swe-mini ``agent.traj.json`` trajectory object."""
    with open(path, "r", encoding="utf-8") as fh:
        traj = json.load(fh)

    total_input = total_output = total_tokens = turns = steps = 0
    context_series: list[int] = []
    for msg in traj.get("messages") or []:
        role = msg.get("role")
        if role == "tool":
            steps += 1
            continue
        if role != "assistant":
            continue
        turns += 1
        usage = (((msg.get("extra") or {}).get("response") or {}).get("usage")) or {}
        inp = int(usage.get("prompt_tokens") or 0)
        out = int(usage.get("completion_tokens") or 0)
        tot = int(usage.get("total_tokens") or (inp + out))
        total_input += inp
        total_output += out
        total_tokens += tot
        if inp:
            context_series.append(inp)
    return _finish("swe-mini", path.name, total_input, total_output,
                   total_tokens, turns, steps, context_series)


def parse_claude(path: Path) -> dict:
    """Parse a Claude Code ``agent.stream.jsonl`` (--output-format stream-json).

    Claude Code emits ONE ``assistant`` event per content block, all sharing the
    same ``message.id`` and repeating that message's ``usage``. So we count a
    *turn* (and its tokens) once per unique message id, while *steps* (tool
    executions) are counted per ``tool_use`` block across all events.
    """
    total_input = total_output = total_tokens = turns = steps = 0
    context_series: list[int] = []
    seen_msgs: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message") or {}
            # Count tool_use content blocks as tool executions (steps). These are
            # per-block events, so count them every time (no dedup).
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    steps += 1
            # Token usage + turn count are per-MESSAGE (the same usage is
            # repeated on every block event of one response) — dedup by id.
            mid = msg.get("id")
            if mid is not None and mid in seen_msgs:
                continue
            if mid is not None:
                seen_msgs.add(mid)
            turns += 1
            usage = msg.get("usage") or {}
            # Effective prompt context = fresh input + both cache tiers; Claude
            # re-sends (cached) history each call, so this is the real per-call
            # context size the compaction heuristic keys off.
            inp = (int(usage.get("input_tokens") or 0)
                   + int(usage.get("cache_read_input_tokens") or 0)
                   + int(usage.get("cache_creation_input_tokens") or 0))
            out = int(usage.get("output_tokens") or 0)
            total_input += inp
            total_output += out
            total_tokens += inp + out
            if inp:
                context_series.append(inp)
    return _finish("claude", path.name, total_input, total_output, total_tokens,
                   turns, steps, context_series)


def parse_kimi(path: Path) -> dict:
    """Parse a Kimi Code CLI ``agent.stream.jsonl``.

    Flat chat messages, one per line, discriminated by a top-level ``role``:

      * ``assistant`` — one model call (may carry ``content``, ``tool_calls``,
        or both). Counted as a turn. Kimi emits ONE line per response, not one
        per content block as Claude Code does, so no message-id dedup is needed.
      * ``tool``      — one tool result. Counted as a step, matching swe-mini
        (steps are executions that came back, not calls that were requested; an
        assistant line may request several at once).
      * ``meta``      — session banner and API retry records. Neither a turn nor
        a step.

    No usage anywhere in the format, hence ``usage_available=False``.
    """
    turns = steps = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = ev.get("role")
            if role == "assistant":
                turns += 1
            elif role == "tool":
                steps += 1
    return _finish("kimi", path.name, 0, 0, 0, turns, steps, [],
                   usage_available=False)


# Top-level "type" values that only Claude Code's stream-json emits.
_CLAUDE_EVENT_TYPES = {"assistant", "user", "result", "system"}
# Top-level "role" values that only the Kimi Code CLI stream emits.
_KIMI_ROLES = {"assistant", "tool", "meta"}


def _sniff_stream_agent(path: Path, max_lines: int = 40) -> str:
    """Return "kimi" or "claude" for an ``agent.stream.jsonl``.

    Both agents write that same filename, so the format has to be identified by
    shape. Claude Code events wrap the payload: a top-level ``type`` drawn from
    _CLAUDE_EVENT_TYPES plus a ``message`` object. Kimi lines are bare chat
    messages: a top-level ``role``, no ``message`` key. Only the first few lines
    are read; ties fall back to claude, preserving the old behaviour for any
    stream this misses.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            if "message" in ev and ev.get("type") in _CLAUDE_EVENT_TYPES:
                return "claude"
            if ev.get("role") in _KIMI_ROLES:
                return "kimi"
    return "claude"


def parse_round(round_dir: Path) -> Optional[dict]:
    """Locate and parse the agent trace file in ``round_dir`` (parent of .eval).

    Returns the stats dict, or ``None`` if no recognized trace file is present.
    """
    pi_trace = round_dir / "agent.jsonl"
    swe_trace = round_dir / "agent.traj.json"
    stream_trace = round_dir / "agent.stream.jsonl"
    if pi_trace.exists():
        return parse_pi(pi_trace)
    if swe_trace.exists():
        return parse_swe_mini(swe_trace)
    if stream_trace.exists():
        # claude and kimi share this filename — pick the parser by content.
        if _sniff_stream_agent(stream_trace) == "kimi":
            return parse_kimi(stream_trace)
        return parse_claude(stream_trace)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="traces", help="Trace root directory")
    parser.add_argument(
        "--problems",
        default=None,
        help="Comma-separated problem names to limit to (default: all found)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    traces_root = Path(args.traces)
    only = set(args.problems.split(",")) if args.problems else None

    if not traces_root.exists():
        print(f"ERROR: traces root not found: {traces_root}", file=sys.stderr)
        return 1

    written = 0
    missing = 0
    errored = 0
    for facts in traces_root.glob("**/.eval/facts.json"):
        eval_dir = facts.parent
        round_dir = eval_dir.parent
        rel = eval_dir.relative_to(traces_root)
        problem = rel.parts[0] if rel.parts else None
        if only and problem not in only:
            continue

        try:
            stats = parse_round(round_dir)
        except Exception as exc:  # one bad trace must not abort the rest
            print(f"[ERROR] {round_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            errored += 1
            continue

        if stats is None:
            print(f"WARN: no agent.jsonl / agent.traj.json / agent.stream.jsonl in {round_dir}", file=sys.stderr)
            missing += 1
            continue

        with open(eval_dir / "stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        flag = "  [COMPACTED]" if stats["compacted"] else ""
        # None = the agent's trace carries no usage (kimi), not a measured zero.
        tok = "n/a" if stats["total_tokens"] is None else stats["total_tokens"]
        peak = "n/a" if stats["peak_context"] is None else stats["peak_context"]
        print(
            f"[stats] {round_dir}: {stats['turns']} turns, {stats['steps']} steps, "
            f"{tok} tok, peak ctx {peak}{flag}"
        )
        written += 1

    print(f"\nWrote {written} stats.json; missing-trace {missing}; errored {errored}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
