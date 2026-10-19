# gpt-oss chat template

Target: the `chat_template.jinja` that ships with `openai/gpt-oss-120b` (pristine file; the patch
was made against the copy in the 120b checkpoint). Used by `single-node/models/gpt-oss-120b.sh`
through `--chat-template`.

Changes:
- Past reasoning is read from `message.reasoning_content` as well as `message.thinking`. sglang keeps
  reasoning in `reasoning_content`, so the analysis a client echoed back never reached the model.
- A tool-call turn with both content and reasoning is rendered (analysis message, then a commentary
  preamble) where the original raises an exception.
- The original's `inner_type|length > 50` cut-off is removed. It rendered every array of objects in
  a tool schema as `any[]` and hid the item fields from the model.

## Harmony vocab for /v1/responses

`/v1/responses` loads the o200k harmony vocabulary over HTTPS on init. Compute nodes often have no
route out and sglang then disables that API. Pre-fetch it on a login node into `$HARMONY_CACHE`
(default `$MODEL_DIR/harmony-tiktoken-cache`); the file name is the SHA-1 of the download URL:

    mkdir -p "$HARMONY_CACHE"
    curl -L https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken \
      -o "$HARMONY_CACHE/fb374d419588a4632f3f557e76b4b70aebbca790"
    sha256sum "$HARMONY_CACHE"/fb374d41*   # 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d

The model file binds the directory to `/harmony-cache` and sets `TIKTOKEN_RS_CACHE_DIR`.
