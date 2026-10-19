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
    per-call usage); it cannot be back-filled from existing traces. kimi
    compaction is only logged in the wire log, so without it ``compacted`` is
    always False and should be read as "unknown", not "confirmed no compaction".

    Exception: rounds run after 2026-09-17 keep kimi's session wire log as
    ``agent.kimi-wire/main.wire.jsonl``. Its ``step.end`` records carry per-call
    ``usage`` and ``finishReason``, so those rounds get real token counts, a
    context series, ``capped_calls``, and ``output_limit`` when the last call hit
    the server output cap (kimi ends the round there and the stream alone looks
    ``completed``).

Definitions, kept identical across agents so models are comparable:

  * ``turns``  — number of model calls (assistant responses). The "iteration
    steps" the agent took.
  * ``steps``  — number of tool executions performed.
  * ``total_input`` / ``total_output`` / ``total_tokens`` — summed over every
    model call (prompt tokens are re-sent each call, so ``total_input`` is the
    real billed prompt cost, not a single snapshot).
  * ``peak_context`` — high-water mark of the per-call context size.
  * ``end_reason`` — WHY the session stopped, in the shared vocabulary of
    ``checks/failure_class.py`` (``completed``, ``context_overflow``,
    ``format_error``, ``max_turns``, ``cut_off``, ...). Every schema records
    this somewhere different — pi in a trailing ``agent_end`` envelope, swe-mini
    in a ``role:"exit"`` message, claude in its ``result`` record, kimi only in
    the sibling ``agent.stderr.log`` — so each parser digs it out and hands it
    to the one classifier. ``end_severity`` says whether that ending is normal
    (``ok``), a cap the harness enforced (``warn``), or a provider failure
    (``error``); ``end_stop_reason`` keeps the harness's own raw word for it and
    ``end_detail`` the provider message. collect_results.py turns this into each
    round's ``fail_class``, which is what replaced the near-useless "failed at
    the build stage" as the reason a round is reported to have failed.
  * ``compacted`` / ``compaction_points`` — context compaction as the agent
    logged it: pi ``compaction_start`` events, claude ``compact_boundary``
    system events, kimi ``compaction.*`` wire records. A compaction point is the
    number of model calls made before the compaction. swe-mini has no
    compaction, and kimi rounds without a wire log cannot tell (always False).
    Context drops are not used: they miss a compaction at the end of a round
    and fire on cache accounting and on kimi dropping old reasoning.

Pure stdlib, no GPU. Idempotent — re-run any time:

    python checks/trace_stats.py --traces traces
    python checks/trace_stats.py --traces traces --problems tiled-gemm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import failure_class as fc

def _finish(agent, source, total_input, total_output, total_tokens,
            turns, steps, context_series, usage_available=True,
            end=("unknown", None, None, None), end_stop_reason=None,
            end_detail="", capped_calls=None, capped_ctx=None,
            empty_reply_nudges=None, compaction_points=()) -> dict:
    """Assemble a stats dict. ``usage_available=False`` marks an agent whose
    trace carries no token usage at all (kimi): the token and context fields
    then come out as ``None`` rather than a misleading 0.

    ``end`` is the ``(reason, tokens used, token limit, noun)`` tuple from the
    matching ``failure_class.*_end_reason`` — why the agent's session stopped.
    It is the same slug the trace viewer shows on the timeline's last node, and
    collect_results.py turns it into the round's ``fail_class``.

    ``capped_calls`` is the number of model calls stopped by the token limit
    (finish reason ``length`` / ``max_tokens``), ``None`` when the trace cannot
    tell. ``capped_ctx`` lists input + output tokens of each such call, in
    order: a value at the server window means the context filled up, a value
    well below it with output near a cap means an output cap.

    ``empty_reply_nudges`` (claude only, ``None`` elsewhere) counts the calls
    Claude Code adds after a thinking-only reply. They are real model calls
    in ``turns`` that ``--max-turns`` does not count, so ``turns`` can pass 120."""
    points = list(compaction_points)
    if not usage_available:
        total_input = total_output = total_tokens = None
        peak_context = None
    else:
        peak_context = max(context_series) if context_series else 0
    reason, used, limit, _noun = end
    return {
        "agent": agent,
        "source": source,
        # why the run stopped (failure_class vocabulary), not whether it passed
        "end_reason": reason,
        "end_severity": fc.END_SEVERITY.get(reason, "error"),
        "end_stop_reason": end_stop_reason,
        "end_detail": (end_detail or "")[:500] or None,
        "end_tokens_used": used,
        "end_tokens_limit": limit,
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "turns": turns,
        "steps": steps,
        "capped_calls": capped_calls,
        "capped_ctx": capped_ctx,
        "empty_reply_nudges": empty_reply_nudges,
        "peak_context": peak_context,
        "compacted": bool(points),
        "compaction_points": points,
    }


def parse_pi(path: Path) -> dict:
    """Parse a pi ``agent.jsonl`` event stream."""
    total_input = total_output = total_tokens = turns = steps = 0
    context_series: list[int] = []
    end_stop = end_err = None   # no agent_end at all = the process was killed
    capped_ctx: list[int] = []
    points: list[int] = []
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
            if etype == "agent_end":
                # pi can emit one per retry attempt; the last one is the ending
                msgs = ev.get("messages") or []
                last = msgs[-1] if msgs else {}
                end_stop, end_err = last.get("stopReason"), last.get("errorMessage")
            elif etype == "tool_execution_start":
                steps += 1
            elif etype == "compaction_start":
                points.append(turns)
            elif etype == "message_end":
                msg = ev.get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                turns += 1
                usage = msg.get("usage") or {}
                # full prompt = fresh input + both cache tiers, as for claude
                inp = (int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
                       + int(usage.get("cacheWrite") or 0))
                out = int(usage.get("output") or 0)
                tot = int(usage.get("totalTokens") or (inp + out))
                total_input += inp
                total_output += out
                total_tokens += tot
                if inp:
                    context_series.append(inp)
                # only message_end counts: turn_end and agent_end repeat the message
                if msg.get("stopReason") == "length":
                    capped_ctx.append(inp + out)
    return _finish("pi", path.name, total_input, total_output, total_tokens,
                   turns, steps, context_series,
                   capped_calls=len(capped_ctx), capped_ctx=capped_ctx,
                   compaction_points=points,
                   end=fc.pi_end_reason(end_stop, end_err),
                   end_stop_reason=end_stop,
                   end_detail=fc.provider_message(end_err))


def parse_swe_mini(path: Path) -> dict:
    """Parse a swe-mini ``agent.traj.json`` trajectory object."""
    with open(path, "r", encoding="utf-8") as fh:
        traj = json.load(fh)

    total_input = total_output = total_tokens = turns = steps = 0
    context_series: list[int] = []
    exit_status = None      # extra.exit_status of the trailing role:"exit" record
    exit_msg = ""           # ...and the exception text that came with it
    capped_ctx: list[int] = []
    messages = traj.get("messages") or []
    for msg in messages:
        role = msg.get("role")
        if role == "exit":
            extra = msg.get("extra") or {}
            extra = extra if isinstance(extra, dict) else {}
            exit_status = extra.get("exit_status")
            exit_msg = extra.get("exception_str") or msg.get("content") or ""
            continue
        if role == "tool":
            steps += 1
            continue
        if role != "assistant":
            continue
        turns += 1
        response = (msg.get("extra") or {}).get("response") or {}
        usage = response.get("usage") or {}
        inp = int(usage.get("prompt_tokens") or 0)
        out = int(usage.get("completion_tokens") or 0)
        tot = int(usage.get("total_tokens") or (inp + out))
        total_input += inp
        total_output += out
        total_tokens += tot
        if inp:
            context_series.append(inp)
        choices = response.get("choices") or [{}]
        if choices[0].get("finish_reason") == "length":
            capped_ctx.append(inp + out)
    reason, used, limit, what, note = fc.swemini_end_reason(
        exit_status, exit_msg, messages)
    return _finish("swe-mini", path.name, total_input, total_output,
                   total_tokens, turns, steps, context_series,
                   end=(reason, used, limit, what),
                   end_stop_reason=exit_status, end_detail=note or exit_msg,
                   capped_calls=len(capped_ctx), capped_ctx=capped_ctx)


# Claude Code's re-prompt after a reply with no visible output.
EMPTY_REPLY_NUDGE = "[Your previous response had no visible output"


def parse_claude(path: Path) -> dict:
    """Parse a Claude Code ``agent.stream.jsonl`` (--output-format stream-json).

    Claude Code emits ONE ``assistant`` event per content block, all sharing the
    same ``message.id`` and repeating that message's ``usage``. So we count a
    *turn* (and its tokens) once per unique message id, while *steps* (tool
    executions) are counted per ``tool_use`` block across all events.

    Token totals: for rounds routed through the LiteLLM gateway to a
    self-hosted model, the per-message ``usage`` above comes back all-zero on
    every single event (the gateway doesn't relay it), even though the round
    ran for real. The terminal ``type: "result"`` event's own ``usage`` is a
    session total computed by the Claude Code CLI itself, independent of
    whether the gateway relayed per-call usage — so when it's present and
    nonzero, prefer it over the per-message sum. Falls back to the
    per-message sum when the result event is missing (process killed
    mid-run) or itself reports all-zero usage. Note: this doesn't help
    ``context_series`` (used for peak-context/compaction detection) — the
    result event has only a final total, not a per-call series, so that
    stays 0/unflagged for these rounds.

    ``capped_calls`` stays ``None``: the stream records ``stop_reason: null`` on
    every call, and the session log that has it is deleted with the round's
    home dir. Tool-input JSON errors are no substitute, since models also emit
    broken tool JSON without any cap.
    """
    total_input = total_output = total_tokens = turns = steps = 0
    context_series: list[int] = []
    seen_msgs: set[str] = set()
    result_rec = None   # no result record = the process was killed mid-run
    nudges = 0
    points: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                result_rec = ev
                continue
            if ev.get("type") == "system" and ev.get("subtype") == "compact_boundary":
                points.append(turns)
                continue
            if ev.get("type") == "user":
                content = (ev.get("message") or {}).get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                for block in content or []:
                    if (isinstance(block, dict) and block.get("type") == "text"
                            and (block.get("text") or "").startswith(EMPTY_REPLY_NUDGE)):
                        nudges += 1
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
    res = result_rec or {}
    result_usage = res.get("usage") or {}
    result_inp = (int(result_usage.get("input_tokens") or 0)
                  + int(result_usage.get("cache_read_input_tokens") or 0)
                  + int(result_usage.get("cache_creation_input_tokens") or 0))
    result_out = int(result_usage.get("output_tokens") or 0)
    if result_inp + result_out > 0:
        total_input, total_output = result_inp, result_out
        total_tokens = result_inp + result_out
    detail = res.get("result") if res.get("is_error") else ""
    return _finish("claude", path.name, total_input, total_output, total_tokens,
                   turns, steps, context_series,
                   end=fc.claude_end_reason(result_rec),
                   end_stop_reason=res.get("terminal_reason") or res.get("subtype"),
                   end_detail=detail if isinstance(detail, str) else "",
                   empty_reply_nudges=nudges, compaction_points=points)


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

    No usage anywhere in the format, hence ``usage_available=False``, unless the
    round kept its wire log (``failure_class.read_kimi_wire``).
    """
    turns = steps = 0
    last_role = last_meta_type = None
    last_has_toolcalls = final_has_toolcalls = backgrounded = False
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
            last_role, last_meta_type = role, None
            last_has_toolcalls = False
            if role == "assistant":
                turns += 1
                last_has_toolcalls = final_has_toolcalls = bool(ev.get("tool_calls"))
            elif role == "tool":
                steps += 1
                backgrounded = backgrounded or fc.KIMI_BACKGROUND_TASK in line
            elif role == "meta":
                last_meta_type = ev.get("type")

    # A harness-side stop (step cap, dropped connection, provider error) is
    # never written into the stream — it only reaches the sibling stderr log.
    wire = fc.read_kimi_wire(str(path.parent))
    reason, msg, used, limit, what = fc.kimi_end_reason(
        str(path.parent), last_role, last_has_toolcalls, last_meta_type,
        final_has_toolcalls, backgrounded, wire[0] if wire else None)
    end = (reason, used, limit, what)
    if wire is None:
        return _finish("kimi", path.name, 0, 0, 0, turns, steps, [],
                       usage_available=False, end=end, end_detail=msg)
    wire, points = wire
    context_series = [context for context, _stop, _out in wire]
    total_input = sum(context_series)
    total_output = sum(out for _context, _stop, out in wire)
    capped_ctx = [context + out for context, stop, out in wire
                  if stop == "max_tokens"]
    return _finish("kimi", path.name, total_input, total_output,
                   total_input + total_output, turns, steps, context_series,
                   end=end, end_detail=msg, capped_calls=len(capped_ctx),
                   capped_ctx=capped_ctx, compaction_points=points)


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
    are read.

    An EMPTY stream decides nothing — kimi writes none when it dies before the
    first response (a rate limit on the opening request), and calling those
    rounds "claude" both mislabelled the agent and lost the real ending. The
    sibling ``agent.stderr.log`` settles it: its terminal ``error: failed to run
    prompt: ...`` line is the kimi CLI's own format, which no other harness
    emits. Everything still undecided falls back to claude, as before.
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
    if fc.kimi_terminal_reason(fc.read_kimi_stderr_tail(str(path.parent))):
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
    parser.add_argument(
        "--campaign",
        default=None,
        help="Only rounds of this campaign dir name; avoids walking the whole trace tree",
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
    pattern = f"*/*/*/{args.campaign}/round-*/.eval/facts.json" if args.campaign else "**/.eval/facts.json"
    for facts in traces_root.glob(pattern):
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

        stats_path = eval_dir / "stats.json"
        # Preserve any keys this parse doesn't produce (e.g. kimi's
        # est_total_input/est_total_output/est_total_tokens/est_tokenizer_source,
        # backfilled separately by checks/kimi_token_estimate.py) so re-running
        # this script doesn't silently wipe them out.
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as fh:
                    prior = json.load(fh)
                for key, val in prior.items():
                    stats.setdefault(key, val)
            except (json.JSONDecodeError, OSError):
                pass
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        flag = "  [COMPACTED]" if stats["compacted"] else ""
        # None = the agent's trace carries no usage (kimi), not a measured zero.
        tok = "n/a" if stats["total_tokens"] is None else stats["total_tokens"]
        peak = "n/a" if stats["peak_context"] is None else stats["peak_context"]
        print(
            f"[stats] {round_dir}: {stats['turns']} turns, {stats['steps']} steps, "
            f"{tok} tok, peak ctx {peak}, end {stats['end_reason']}{flag}"
        )
        written += 1

    print(f"\nWrote {written} stats.json; missing-trace {missing}; errored {errored}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
