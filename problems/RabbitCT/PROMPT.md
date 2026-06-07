Implement a GPU version of the RabbitCT LolaBunny backprojection.

Add a new GPU algorithm named `LolaBunnyGPU` that produces the same output as LolaBunny and works with the existing runner and `-c` verification. Keep LolaBunny intact for baseline comparison.

Build, run, and verify your implementation. Debug until it produces correct results. Use size 256 and `./RabbitInput/Reference256.vol` for verification with the `-c` flag.

When finished, write a file `solution.json` in the problem root with the exact commands you used:

```json
{
  "build": ["make TOOLCHAIN=CLANG"],
  "run": ["./run-bench.sh -s 256 -v LolaBunnyGPU -c ./RabbitInput/Reference256.vol"],
  "variant_name": "LolaBunnyGPU"
}
```

Requirements:
- All commands must be relative to the problem root directory.
- `variant_name` is the name passed to `-v` in the run command.
- These commands will be independently re-run by the evaluation harness for verification.
- If your solution does not compile or produces incorrect results, the harness will fail — write commands that actually work.
