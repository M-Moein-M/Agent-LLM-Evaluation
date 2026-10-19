#### RabbitCT-LolaBunny — ranking metric: `mse` [HU^2] (lower is better)

| Agent | Model | pass@1 | mse [HU^2] (mean ± sd) | best | n | limits |
|---|---|:--:|:--:|:--:|:--:|:--|
| claude-pro | claude-opus-4-8 | 5/5 | 0 ± 0 | 0 | 5 | — |
| swe-mini | Kimi-K2.7-Code | 10/10 | 0 ± 0 | 0 | 10 | — |
| pi | DeepSeek-V4-Flash-0731 | 13/15 | 0 ± 0 | 0 | 13 | — |
| swe-mini | Qwen3.8-27B | 6/10 | 0 ± 0 | 0 | 7 | 1 capped |
| claude | gpt-oss-120b | 1/10 | 0 ± n/a | 0 | 1 | 2 nudges |
| swe-mini | DeepSeek-V4-Flash-0731 | 12/15 | 4.13e-08 ± 1.49e-07 | 0 | 13 | — |
| pi | Qwen3.8-27B | 12/20 | 1.65e-07 ± 2.58e-07 | 0 | 13 | 22 capped |
| swe-mini | Qwen3.6-27B | 9/10 | 2.98e-07 ± 2.83e-07 | 0 | 9 | 1 capped |
| kimi | DeepSeek-V4-Flash-0731 | 16/20 | 4.77e-07 ± 1.83e-06 | 0 | 17 | 3 >120 calls |
| pi | gpt-oss-120b | 1/5 | 5.36e-07 ± n/a | 5.36e-07 | 1 | — |
| kimi | Qwen3.6-35B-A3B-FP8 | 11/19 | 7.47e-07 ± 2.07e-06 | 0 | 13 | 2 hung |
| pi | Qwen3.6-27B | 18/20 | 1.03e-06 ± 2.37e-06 | 0 | 20 | — |
| pi | Qwen3.6-35B-A3B-FP8 | 11/20 | 1.06e-06 ± 2.17e-06 | 0 | 12 | 2 capped |
| claude | Qwen3.6-27B | 16/20 | 1.08e-06 ± 2.78e-06 | 0 | 20 | 6 nudges |
| kimi | Qwen3.6-27B | 18/20 | 1.3e-06 ± 2.55e-06 | 0 | 20 | 1 hung |
| claude | Qwen3.8-27B | 6/10 | 1.3e-06 ± 3.26e-06 | 0 | 8 | 1 >120 calls |
| pi | Kimi-K2.7-Code | 15/50 | 1.41e-06 ± 3.2e-06 | 0 | 18 | — |
| swe-mini | DeepSeek-V4-Flash | 5/10 | 1.59e-06 ± 3.12e-06 | 0 | 6 | — |
| claude | Qwen3.6-35B-A3B-FP8 | 10/30 | 1.86e-06 ± 4.3e-06 | 0 | 13 | 8 >120 calls, 59 nudges |
| pi | DeepSeek-V4-Flash | 10/25 | 4.48e-06 ± 5.34e-06 | 0 | 17 | — |
| claude | gemma-4-31B-it | 0/20 | 4e-05 ± 0.000111 | 7.57e-06 | 12 | — |
| kimi | DeepSeek-V4-Flash | 7/10 | 4.49e-05 ± 0.000131 | 0 | 9 | — |
| swe-mini | Qwen3.6-35B-A3B-FP8 | 5/10 | 5.63e-05 ± 0.000148 | 0 | 7 | — |
| claude | DeepSeek-V4-Flash | 6/10 | 5.67e-05 ± 0.000149 | 5.36e-07 | 7 | — |
| claude | DeepSeek-V4-Flash-0731 | 15/20 | 0.000124 ± 0.000351 | 0 | 17 | — |
| kimi | Qwen3.8-27B | 5/15 | 0.000135 ± 0.000329 | 0 | 6 | 2 capped |
| kimi | gemma-4-31B-it | 1/10 | 0.894 ± 2.68 | 5.36e-07 | 9 | — |
| claude | Kimi-K2.7-Code | 7/20 | 7.43e+03 ± 2.57e+04 | 0 | 12 | — |
| pi | gemma-4-31B-it | 1/10 | 9.58e+03 ± 2.88e+04 | 5.36e-07 | 9 | — |
| swe-mini | gemma-4-31B-it-FP8-block | 1/10 | 5.75e+04 ± 4.98e+04 | 5.36e-07 | 3 | — |
| claude | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| claude | Llama-3.3-70B-Instruct | 0/10 | — | — | 0 | 1 >120 calls |
| kimi | Kimi-K2.7-Code-NVFP4 | 0/5 | — | — | 0 | — |
| kimi | gpt-oss-120b | 0/5 | — | — | 0 | — |
| kimi | Llama-3.3-70B-Instruct | 0/5 | — | — | 0 | — |
| kimi | Kimi-K2.7-Code | 0/5 | — | — | 0 | — |
| pi | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| pi | Kimi-K2.7-Code-NVFP4 | 0/10 | — | — | 0 | — |
| pi | Llama-3.3-70B-Instruct | 0/10 | — | — | 0 | — |
| swe-mini | gpt-oss-120b | 0/20 | — | — | 0 | — |
| swe-mini | Llama-3.3-70B-Instruct | 0/20 | — | — | 0 | — |

> `n` = rounds that produced a measurement — every round that built and ran, **passing or failing**. Rounds that failed at build/run never measured anything and are in pass@1's denominator only, so `n < runs` there. The metric is problem-local: it ranks models *within* this table and is not comparable across problems — pass@1 is the cross-problem axis.

> `limits` = model calls stopped by the output cap (`capped`, summed over rounds; unknown for claude and old kimi rounds), rounds over 120 model calls, rounds where kimi hung after its final reply (`hung`, pass verdict kept), and claude's re-prompts after a thinking-only reply (`nudges`, not counted by its turn limit).

#### RabbitCT-LolaOMP — ranking metric: `mse` [HU^2] (lower is better)

| Agent | Model | pass@1 | mse [HU^2] (mean ± sd) | best | n | limits |
|---|---|:--:|:--:|:--:|:--:|:--|
| claude | Qwen3.6-27B | 8/10 | 0 ± 0 | 0 | 10 | 6 nudges |
| pi | Kimi-K2.7-Code | 20/50 | 0 ± 0 | 0 | 20 | — |
| claude | gpt-oss-120b | 1/10 | 7.19e-06 ± 6.23e-06 | 0 | 3 | 1 >120 calls, 5 nudges |
| swe-mini | DeepSeek-V4-Flash-0731 | 14/15 | 4.15e-05 ± 0.000161 | 0 | 15 | — |
| swe-mini | Kimi-K2.7-Code | 9/10 | 6.23e-05 ± 0.000197 | 0 | 10 | — |
| swe-mini | Qwen3.8-27B | 9/10 | 6.24e-05 ± 0.000197 | 0 | 10 | — |
| pi | Qwen3.8-27B | 15/20 | 6.93e-05 ± 0.000202 | 0 | 18 | — |
| claude | Qwen3.8-27B | 7/10 | 7.79e-05 ± 0.00022 | 0 | 8 | — |
| kimi | Qwen3.8-27B | 14/20 | 7.79e-05 ± 0.000213 | 0 | 16 | — |
| kimi | Qwen3.6-27B | 15/20 | 9.63e-05 ± 0.000227 | 0 | 20 | — |
| claude | DeepSeek-V4-Flash | 6/10 | 0.000106 ± 0.000219 | 0 | 10 | — |
| pi | DeepSeek-V4-Flash-0731 | 11/15 | 0.000134 ± 0.000265 | 0 | 14 | — |
| swe-mini | Qwen3.6-27B | 7/10 | 0.000141 ± 0.000235 | 0 | 10 | — |
| kimi | DeepSeek-V4-Flash-0731 | 11/20 | 0.000156 ± 0.000279 | 0 | 16 | 2 >120 calls |
| pi | Qwen3.6-35B-A3B-FP8 | 6/20 | 0.000171 ± 0.00029 | 0 | 11 | — |
| kimi | DeepSeek-V4-Flash | 5/10 | 0.000183 ± 0.00028 | 0 | 9 | — |
| pi | DeepSeek-V4-Flash | 14/25 | 0.000189 ± 0.000278 | 0 | 24 | — |
| claude | Kimi-K2.7-Code | 13/20 | 0.000197 ± 0.000333 | 0 | 18 | — |
| swe-mini | DeepSeek-V4-Flash | 5/10 | 0.000249 ± 0.000322 | 0 | 10 | — |
| claude | DeepSeek-V4-Flash-0731 | 10/20 | 0.000271 ± 0.000338 | 0 | 17 | — |
| swe-mini | Qwen3.6-35B-A3B-FP8 | 1/10 | 0.000426 ± 0.000602 | 0 | 2 | — |
| kimi | Qwen3.6-35B-A3B-FP8 | 2/20 | 0.000492 ± 0.000234 | 0 | 13 | — |
| kimi | gemma-4-31B-it | 0/10 | 0.000565 ± 0.000115 | 0.000393 | 4 | — |
| claude-pro | claude-opus-4-8 | 0/5 | 0.000623 ± 0 | 0.000623 | 5 | — |
| claude | Qwen3.6-35B-A3B-FP8 | 1/30 | 633 ± 1.79e+03 | 0 | 8 | 7 >120 calls, 68 nudges |
| pi | Qwen3.6-27B | 17/20 | 4.46e+03 ± 1.99e+04 | 0 | 20 | — |
| claude | gemma-4-31B-it | 1/10 | 2.88e+04 ± 4.46e+04 | 0 | 6 | — |
| swe-mini | gpt-oss-120b | 0/20 | 4.31e+04 ± 4.98e+04 | 1.02e-05 | 4 | — |
| pi | gemma-4-31B-it | 0/10 | 2.16e+05 ± 5.19e+05 | 0.000394 | 9 | — |
| swe-mini | gemma-4-31B-it-FP8-block | 1/10 | 2.23e+06 ± 4.65e+06 | 0 | 10 | — |
| claude | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| claude | Llama-3.3-70B-Instruct | 0/10 | — | — | 0 | — |
| kimi | Kimi-K2.7-Code-NVFP4 | 0/5 | — | — | 0 | — |
| kimi | gpt-oss-120b | 0/5 | — | — | 0 | — |
| kimi | Llama-3.3-70B-Instruct | 0/5 | — | — | 0 | — |
| kimi | Kimi-K2.7-Code | 0/5 | — | — | 0 | — |
| pi | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| pi | Kimi-K2.7-Code-NVFP4 | 0/10 | — | — | 0 | — |
| pi | gpt-oss-120b | 0/5 | — | — | 0 | — |
| pi | Llama-3.3-70B-Instruct | 0/10 | — | — | 0 | — |
| swe-mini | Llama-3.3-70B-Instruct | 0/20 | — | — | 0 | — |

> `n` = rounds that produced a measurement — every round that built and ran, **passing or failing**. Rounds that failed at build/run never measured anything and are in pass@1's denominator only, so `n < runs` there. The metric is problem-local: it ranks models *within* this table and is not comparable across problems — pass@1 is the cross-problem axis.

> `limits` = model calls stopped by the output cap (`capped`, summed over rounds; unknown for claude and old kimi rounds), rounds over 120 model calls, rounds where kimi hung after its final reply (`hung`, pass verdict kept), and claude's re-prompts after a thinking-only reply (`nudges`, not counted by its turn limit).

#### stream-compact — ranking metric: `mse` (lower is better)

| Agent | Model | pass@1 | mse (mean ± sd) | best | n | limits |
|---|---|:--:|:--:|:--:|:--:|:--|
| claude-pro | claude-opus-4-8 | 5/5 | 0 ± 0 | 0 | 5 | — |
| claude | DeepSeek-V4-Flash | 10/10 | 0 ± 0 | 0 | 10 | — |
| claude | gpt-oss-120b | 10/10 | 0 ± 0 | 0 | 10 | — |
| claude | Qwen3.8-27B | 10/10 | 0 ± 0 | 0 | 10 | — |
| kimi | DeepSeek-V4-Flash | 10/10 | 0 ± 0 | 0 | 10 | — |
| pi | DeepSeek-V4-Flash | 20/20 | 0 ± 0 | 0 | 20 | — |
| pi | DeepSeek-V4-Flash-0731 | 15/15 | 0 ± 0 | 0 | 15 | — |
| swe-mini | DeepSeek-V4-Flash | 10/10 | 0 ± 0 | 0 | 10 | — |
| swe-mini | Kimi-K2.7-Code | 10/10 | 0 ± 0 | 0 | 10 | — |
| swe-mini | DeepSeek-V4-Flash-0731 | 15/15 | 0 ± 0 | 0 | 15 | — |
| swe-mini | Qwen3.8-27B | 10/10 | 0 ± 0 | 0 | 10 | — |
| claude | Qwen3.6-27B | 19/20 | 0 ± 0 | 0 | 19 | 1 nudges |
| claude | DeepSeek-V4-Flash-0731 | 19/20 | 0 ± 0 | 0 | 19 | — |
| kimi | Qwen3.6-27B | 19/20 | 0 ± 0 | 0 | 19 | 1 hung |
| kimi | Qwen3.6-35B-A3B-FP8 | 19/20 | 0 ± 0 | 0 | 20 | — |
| pi | Qwen3.6-27B | 19/20 | 0 ± 0 | 0 | 19 | 1 capped |
| claude | Kimi-K2.7-Code | 18/20 | 0 ± 0 | 0 | 18 | — |
| kimi | DeepSeek-V4-Flash-0731 | 18/20 | 0 ± 0 | 0 | 18 | — |
| pi | Qwen3.8-27B | 18/20 | 0 ± 0 | 0 | 18 | — |
| swe-mini | Qwen3.6-35B-A3B-FP8 | 9/10 | 0 ± 0 | 0 | 9 | — |
| kimi | Qwen3.8-27B | 17/20 | 0 ± 0 | 0 | 17 | 1 hung |
| kimi | gpt-oss-120b | 3/5 | 0 ± 0 | 0 | 3 | — |
| swe-mini | Qwen3.6-27B | 6/10 | 0 ± 0 | 0 | 6 | — |
| pi | Qwen3.6-35B-A3B-FP8 | 9/20 | 0 ± 0 | 0 | 10 | 9 capped |
| claude | Qwen3.6-35B-A3B-FP8 | 13/30 | 0 ± 0 | 0 | 13 | 2 >120 calls, 24 nudges |
| pi | Kimi-K2.7-Code | 18/50 | 0 ± 0 | 0 | 18 | — |
| swe-mini | gpt-oss-120b | 5/20 | 0 ± 0 | 0 | 6 | — |
| kimi | Kimi-K2.7-Code-NVFP4 | 1/5 | 0 ± n/a | 0 | 1 | — |
| pi | gpt-oss-120b | 1/5 | 2.41e+10 ± 3.4e+10 | 0 | 2 | — |
| kimi | gemma-4-31B-it | 6/10 | 6.67e+10 ± 1.26e+11 | 0 | 9 | — |
| swe-mini | gemma-4-31B-it-FP8-block | 5/10 | 7.4e+10 ± 1.29e+11 | 0 | 9 | — |
| claude | gemma-4-31B-it | 4/10 | 8.4e+10 ± 1.37e+11 | 0 | 7 | — |
| pi | gemma-4-31B-it | 5/10 | 1.24e+11 ± 1.97e+11 | 0 | 9 | — |
| claude | Llama-3.3-70B-Instruct | 0/10 | 6.41e+11 ± 0 | 6.41e+11 | 8 | — |
| kimi | Llama-3.3-70B-Instruct | 0/5 | 6.41e+11 ± n/a | 6.41e+11 | 1 | — |
| pi | Llama-3.3-70B-Instruct | 0/10 | 6.41e+11 ± 0 | 6.41e+11 | 10 | — |
| swe-mini | Llama-3.3-70B-Instruct | 0/20 | 6.41e+11 ± 0 | 6.41e+11 | 4 | — |
| claude | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| kimi | Kimi-K2.7-Code | 0/5 | — | — | 0 | — |
| pi | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| pi | Kimi-K2.7-Code-NVFP4 | 0/10 | — | — | 0 | — |

> `n` = rounds that produced a measurement — every round that built and ran, **passing or failing**. Rounds that failed at build/run never measured anything and are in pass@1's denominator only, so `n < runs` there. The metric is problem-local: it ranks models *within* this table and is not comparable across problems — pass@1 is the cross-problem axis.

> `limits` = model calls stopped by the output cap (`capped`, summed over rounds; unknown for claude and old kimi rounds), rounds over 120 model calls, rounds where kimi hung after its final reply (`hung`, pass verdict kept), and claude's re-prompts after a thinking-only reply (`nudges`, not counted by its turn limit).

#### tiled-gemm — ranking metric: `cublas_frac` [x cuBLAS] (higher is better)

| Agent | Model | pass@1 | cublas_frac [x cuBLAS] (mean ± sd) | best | n | limits |
|---|---|:--:|:--:|:--:|:--:|:--|
| claude | gpt-oss-120b | 1/10 | 1.34 ± 1.61 | 3.16 | 3 | — |
| pi | Kimi-K2.7-Code | 6/50 | 1.25 ± 0.608 | 2.41 | 6 | — |
| claude | Qwen3.8-27B | 2/10 | 1.14 ± 0.793 | 1.7 | 2 | — |
| claude-pro | claude-opus-4-8 | 5/5 | 0.843 ± 0.0491 | 0.893 | 5 | — |
| swe-mini | Kimi-K2.7-Code | 10/10 | 0.824 ± 0.209 | 1.21 | 10 | — |
| claude | Kimi-K2.7-Code | 10/20 | 0.813 ± 0.548 | 2.36 | 11 | 1 nudges |
| swe-mini | DeepSeek-V4-Flash-0731 | 9/15 | 0.751 ± 0.0739 | 0.848 | 9 | — |
| pi | DeepSeek-V4-Flash-0731 | 10/15 | 0.74 ± 0.127 | 0.948 | 10 | 3 capped |
| kimi | DeepSeek-V4-Flash-0731 | 8/20 | 0.685 ± 0.159 | 1 | 8 | 5 >120 calls, 3 hung |
| claude | DeepSeek-V4-Flash-0731 | 9/20 | 0.666 ± 0.173 | 0.856 | 9 | — |
| claude | DeepSeek-V4-Flash | 4/10 | 0.573 ± 0.136 | 0.772 | 5 | 2 >120 calls |
| swe-mini | DeepSeek-V4-Flash | 9/10 | 0.548 ± 0.13 | 0.728 | 9 | — |
| claude | Qwen3.6-27B | 15/20 | 0.542 ± 0.162 | 0.904 | 16 | 1 >120 calls, 7 nudges |
| pi | DeepSeek-V4-Flash | 12/20 | 0.541 ± 0.0745 | 0.645 | 12 | — |
| swe-mini | Qwen3.6-35B-A3B-FP8 | 5/10 | 0.535 ± 0.957 | 3.21 | 10 | — |
| kimi | DeepSeek-V4-Flash | 9/10 | 0.53 ± 0.126 | 0.647 | 9 | — |
| kimi | Qwen3.6-27B | 14/25 | 0.495 ± 0.0735 | 0.65 | 14 | 1 hung |
| pi | Qwen3.6-27B | 11/20 | 0.429 ± 0.116 | 0.65 | 12 | 9 capped |
| kimi | gemma-4-31B-it | 10/10 | 0.425 ± 0.175 | 0.702 | 10 | — |
| swe-mini | gemma-4-31B-it-FP8-block | 9/10 | 0.415 ± 0.157 | 0.634 | 10 | — |
| kimi | Qwen3.6-35B-A3B-FP8 | 7/20 | 0.393 ± 0.14 | 0.587 | 7 | 2 >120 calls |
| pi | Qwen3.6-35B-A3B-FP8 | 5/20 | 0.392 ± 0.216 | 0.604 | 7 | 16 capped |
| swe-mini | Qwen3.6-27B | 7/10 | 0.36 ± 0.16 | 0.624 | 8 | 1 capped |
| pi | gemma-4-31B-it | 8/10 | 0.338 ± 0.184 | 0.639 | 10 | — |
| claude | gemma-4-31B-it | 6/10 | 0.296 ± 0.181 | 0.574 | 10 | — |
| swe-mini | gpt-oss-120b | 0/20 | 0.289 ± 0.331 | 0.955 | 8 | — |
| claude | Qwen3.6-35B-A3B-FP8 | 1/29 | 0.204 ± 0.121 | 0.29 | 2 | 13 >120 calls, 71 nudges |
| claude | Llama-3.3-70B-Instruct | 0/10 | 0.126 ± 0.000421 | 0.127 | 9 | — |
| swe-mini | Llama-3.3-70B-Instruct | 0/20 | 0.126 ± 0.000265 | 0.126 | 6 | — |
| pi | Llama-3.3-70B-Instruct | 0/10 | 0.126 ± 0.000173 | 0.126 | 10 | — |
| kimi | Llama-3.3-70B-Instruct | 0/5 | 0.123 ± 0.00318 | 0.126 | 5 | — |
| claude | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| kimi | Kimi-K2.7-Code-NVFP4 | 0/5 | — | — | 0 | — |
| kimi | Kimi-K2.7-Code | 0/5 | — | — | 0 | — |
| kimi | gpt-oss-120b | 0/5 | — | — | 0 | — |
| kimi | Qwen3.8-27B | 0/15 | — | — | 0 | 6 capped |
| pi | gemma-4-31B-it-FP8-block | 0/10 | — | — | 0 | — |
| pi | Kimi-K2.7-Code-NVFP4 | 0/10 | — | — | 0 | — |
| pi | gpt-oss-120b | 0/5 | — | — | 0 | — |
| pi | Qwen3.8-27B | 0/20 | — | — | 0 | 15 capped |
| swe-mini | Qwen3.8-27B | 0/10 | — | — | 0 | 2 capped |

> `n` = rounds that produced a measurement — every round that built and ran, **passing or failing**. Rounds that failed at build/run never measured anything and are in pass@1's denominator only, so `n < runs` there. The metric is problem-local: it ranks models *within* this table and is not comparable across problems — pass@1 is the cross-problem axis.

> `limits` = model calls stopped by the output cap (`capped`, summed over rounds; unknown for claude and old kimi rounds), rounds over 120 model calls, rounds where kimi hung after its final reply (`hung`, pass verdict kept), and claude's re-prompts after a thinking-only reply (`nudges`, not counted by its turn limit).

