# sglang patches

Local fixes that are bound over files inside the SGLang images (apptainer `--bind file:target:ro`).
Each patch is a unified diff against the pristine file of one image version, see the README in
its directory for the exact target.

| dir | target | what |
|---|---|---|
| `v0.5.16/` | image `sglang_v0.5.16.sif` | `deepseek_v2.py`: Kimi VL wrapper + pipeline parallelism crash |
| `v0.5.20/` | image `sglang_v0.5.20.sif` | `gpt_oss_detector.py`, `harmony_parser.py`: gpt-oss tool-call parsing |
| `gpt-oss/` | the model's own `chat_template.jinja` | reasoning echo, content + reasoning on tool turns, tool schema rendering |

Build the patched files once per image or model checkout:

    sglang-patches/apply_patches.sh v0.5.16     # reads the pristine file out of $IMAGE_DIR/sglang_v0.5.16.sif
    sglang-patches/apply_patches.sh v0.5.20
    sglang-patches/apply_patches.sh gpt-oss     # reads $MODEL_DIR/openai/gpt-oss-120b/chat_template.jinja

Output goes to `$PATCH_DIR/sglang-v0.5.NN/` and `$PATCH_DIR/gpt-oss/`, next to a `.orig` copy of each
pristine file for diffing. The model files (`single-node/models/gpt-oss-120b.sh`,
`multi-node/models/Kimi-K2.7-Code-TP4xPP3.sh`) point at those paths.

If a patch does not apply, the image or checkpoint is not the version it targets. Do not force it:
the files change between SGLang releases.
