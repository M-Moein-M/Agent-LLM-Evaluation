#!/usr/bin/env python3
"""Backfill estimated token counts for kimi agent rounds via local tokenization.

Kimi Code CLI never emits token usage in agent.stream.jsonl (confirmed in
checks/trace_stats.py's parse_kimi() — total_input/total_output/total_tokens
are always None for kimi). This script tokenizes the actually-logged text
(the real per-problem PROMPT.md, an optional captured CLI system+tools
prefix, and a cumulative replay of the stream itself) with each round's real
backend-model tokenizer, and writes the result into new est_* keys in that
round's .eval/stats.json — additive, read-modify-write, leaves every
existing field (including the real, null, total_tokens) untouched.

Requires the reference-results/analysis/tokenizers/ files and the .venv-tokenize venv (see the
plan this script implements — transformers/tiktoken/sentencepiece, no
torch/tensorflow/flax).

Usage: python3 reference-results/analysis/kimi_token_estimate.py [--dry-run] [glob]
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizers"   # gitignored, see fetch_tokenizers.md
PREFIX_CAPTURE_PATH = TOKENIZER_DIR / "kimi_system_prefix.json"

# Trace model-dir name -> tokenizer directory name under reference-results/analysis/tokenizers/.
# Two entries are substitutions (no separate tokenizer copy — same family /
# same weights under quantization); est_tokenizer_source records the real
# tokenizer dir used so this is visible in the output.
TOKENIZER_MAP = {
    "Qwen_Qwen3.6-27B": "Qwen_Qwen3.6-27B",
    "Qwen_Qwen3.6-35B-A3B-FP8": "Qwen_Qwen3.6-35B-A3B-FP8",
    "Qwen_Qwen3.8-27B": "Qwen_Qwen3.8-27B",
    "Qwen_Qwen3.8-27B-think": "Qwen_Qwen3.8-27B",
    "deepseek-ai_DeepSeek-V4-Flash": "deepseek-ai_DeepSeek-V4-Flash",
    "deepseek-ai_DeepSeek-V4-Flash-0731": "deepseek-ai_DeepSeek-V4-Flash-0731",
    "meta-llama_Llama-3.3-70B-Instruct": "meta-llama_Llama-3.3-70B-Instruct",
    "google_gemma-4-31B-it": "google_gemma-4-31B-it",
    "gemmaselfhosted_google_gemma-4-31B-it": "google_gemma-4-31B-it",
    "moonshotai_Kimi-K2.7-Code": "moonshotai_Kimi-K2.7-Code",
    "moonshotai_Kimi-K3": "moonshotai_Kimi-K2.7-Code",
    "nvidia_Kimi-K2.7-Code-NVFP4": "moonshotai_Kimi-K2.7-Code",
    "gpt-oss-120b": "openai_gpt-oss-120b",
    "openai_gpt-oss-120b": "openai_gpt-oss-120b",
}

_tokenizer_cache = {}


def get_tokenizer(tokenizer_dir_name):
    if tokenizer_dir_name not in _tokenizer_cache:
        path = TOKENIZER_DIR / tokenizer_dir_name
        _tokenizer_cache[tokenizer_dir_name] = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True
        )
    return _tokenizer_cache[tokenizer_dir_name]


def count_tokens(tok, text):
    if not text:
        return 0
    return len(tok.encode(text))


def load_cli_prefix():
    if not PREFIX_CAPTURE_PATH.exists():
        return None
    with open(PREFIX_CAPTURE_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    system_text = data.get("system", "")
    tools_text = json.dumps(data.get("tools", []))
    return system_text, tools_text


def parse_stream_events(stream_path):
    events = []
    with open(stream_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                events.append(ev)
    return events


def event_text(ev):
    """Text a single stream event contributes, regardless of role."""
    role = ev.get("role")
    parts = []
    if role == "assistant":
        content = ev.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in ev.get("tool_calls") or []:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            parts.append(name)
            parts.append(args if isinstance(args, str) else json.dumps(args))
    elif role == "tool":
        content = ev.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(p for p in parts if p)


def estimate_round(stream_path, prompt_text, cli_prefix, tok):
    events = parse_stream_events(stream_path)

    output_text_parts = []
    for ev in events:
        if ev.get("role") == "assistant":
            content = ev.get("content")
            if isinstance(content, str):
                output_text_parts.append(content)
            for call in ev.get("tool_calls") or []:
                fn = call.get("function", {})
                args = fn.get("arguments", "")
                output_text_parts.append(args if isinstance(args, str) else json.dumps(args))
    est_output_tokens = count_tokens(tok, "\n".join(output_text_parts))

    base_prefix_text = prompt_text
    if cli_prefix is not None:
        system_text, tools_text = cli_prefix
        base_prefix_text = system_text + "\n" + tools_text + "\n" + prompt_text
    base_tokens = count_tokens(tok, base_prefix_text)

    # Cumulative replay: at each assistant turn, the model sees the base
    # prefix plus everything logged before it in the stream (prior assistant
    # content/tool-calls + prior tool results) — mirroring the "prompt resent
    # each call" definition trace_stats.py uses for other agents.
    total_input_tokens = 0
    running_history_tokens = 0
    for ev in events:
        if ev.get("role") == "assistant":
            total_input_tokens += base_tokens + running_history_tokens
        running_history_tokens += count_tokens(tok, event_text(ev))

    return est_output_tokens, total_input_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print results, do not write stats.json")
    ap.add_argument(
        "glob",
        nargs="?",
        default="traces/*/kimi/*/*/round-*/agent.stream.jsonl",
        help="glob (relative to repo root) selecting agent.stream.jsonl files to process",
    )
    args = ap.parse_args()

    cli_prefix = load_cli_prefix()
    missing_prefix = cli_prefix is None
    if missing_prefix:
        print("WARNING: no reference-results/analysis/tokenizers/kimi_system_prefix.json — "
              "est_total_input will only cover PROMPT.md + stream replay "
              "(est_input_missing_cli_prefix=true)", file=sys.stderr)

    paths = sorted(REPO_ROOT.glob(args.glob))
    if not paths:
        print(f"no files matched glob: {args.glob}", file=sys.stderr)
        return 1

    prompt_cache = {}
    n_ok = n_skip = 0

    for stream_path in paths:
        round_dir = stream_path.parent
        model_dir_name = round_dir.parent.parent.name
        problem_name = round_dir.parents[3].name

        tokenizer_dir_name = TOKENIZER_MAP.get(model_dir_name)
        if tokenizer_dir_name is None:
            print(f"SKIP {stream_path}: no tokenizer mapping for model dir '{model_dir_name}'",
                  file=sys.stderr)
            n_skip += 1
            continue

        if problem_name not in prompt_cache:
            prompt_path = REPO_ROOT / "problems" / problem_name / "PROMPT.md"
            prompt_cache[problem_name] = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
            if not prompt_cache[problem_name]:
                print(f"WARNING: no PROMPT.md for problem '{problem_name}'", file=sys.stderr)

        try:
            tok = get_tokenizer(tokenizer_dir_name)
        except Exception as exc:
            print(f"SKIP {stream_path}: failed to load tokenizer '{tokenizer_dir_name}': {exc}",
                  file=sys.stderr)
            n_skip += 1
            continue

        est_output_tokens, est_total_input = estimate_round(
            stream_path, prompt_cache[problem_name], cli_prefix, tok
        )
        est_total_tokens = est_total_input + est_output_tokens

        stats_path = round_dir / ".eval" / "stats.json"
        if not stats_path.exists():
            print(f"SKIP {stream_path}: no .eval/stats.json next to it", file=sys.stderr)
            n_skip += 1
            continue

        with open(stats_path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)

        stats["est_total_input"] = est_total_input
        stats["est_total_output"] = est_output_tokens
        stats["est_total_tokens"] = est_total_tokens
        stats["est_tokenizer_source"] = tokenizer_dir_name
        stats["est_input_missing_cli_prefix"] = missing_prefix

        if args.dry_run:
            print(f"{stream_path}: est_total_input={est_total_input} "
                  f"est_total_output={est_output_tokens} est_total_tokens={est_total_tokens} "
                  f"tokenizer={tokenizer_dir_name}")
        else:
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, indent=2)

        n_ok += 1

    print(f"done: {n_ok} rounds updated, {n_skip} skipped" + (" (dry-run, nothing written)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
