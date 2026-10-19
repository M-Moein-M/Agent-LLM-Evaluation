# deepseek_v2.py, SGLang 0.5.16

Target: `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py` in `sglang_v0.5.16.sif`.
The pristine file was compared byte for byte with the one in that image.

Bug (0.5.14 and 0.5.16): Kimi (VL wrapper, `kimi_k25`) with pipeline parallelism crashes on PP
ranks above 0 at CUDA-graph capture, `'NoneType' object has no attribute 'shape'` in the
`forward` of `deepseek_v2.py`. On non-first PP ranks the wrapper passes `input_ids=None` and
`input_embeds=None`, the stage input is `pp_proxy_tensors["hidden_states"]`. The patch takes the
length from there.

Not needed on 0.5.20 (the file differs and no longer has this bug according to the notes in
`multi-node/models/Kimi-K2.7-Code-TP4xPP3.sh`). Only used with that model when running on 0.5.16:
enable the `EXTRA_BINDS` line in the model file.
