# gpt-oss parsers, SGLang 0.5.20

Target, in `sglang_v0.5.20.sif` (pristine files compared byte for byte with the image):
- `python/sglang/srt/function_call/gpt_oss_detector.py`
- `python/sglang/srt/parser/harmony_parser.py`

Used by `single-node/models/gpt-oss-120b.sh`, which binds the patched files read-only over the
image copies.

`gpt_oss_detector.py`
- sglang's grammar constraint for harmony ends generation right after the analysis message, so
  `tool_choice` required/named and `strict=true` tools returned an empty reply with no tool call.
  The patch skips the constraint and parses the model's own tool call, as `auto` does. A forced
  call is therefore not grammar-enforced.
- A message ending in `<|call|>` that names no function (no recipient, unknown function, bad JSON)
  is returned as content or reasoning instead of being dropped.
- Accepts `functions.X.json` as recipient and a JSON body `{"name": X, "arguments": {...}}`.

`harmony_parser.py`
- The stock parser accepts a tool call only as
  `<|channel|>commentary to=functions.X <|constrain|>json`. The model also writes it on the
  analysis channel, with `code` or no content type, with the recipient before the channel, without
  a `<|message|>` token, or with the arguments in parentheses. The patch rewrites every header with
  a `functions.*` recipient into the stock shape before the stock code sees it.
