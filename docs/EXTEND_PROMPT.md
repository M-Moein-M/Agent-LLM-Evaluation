# Prompt for your coding agent

Copy the text below, replace the part in angle brackets, and give it to your agent in this repo.

```text
Read AGENT.md first. Then extend this benchmark to evaluate: <one or two sentences describing the
problem, for example "a CPU task: optimize a sparse matrix-vector multiply in C for a given matrix
set; correctness by comparing against a reference output, speed measured as wall time">.

Do the following:
1. Create problems/<name>/ from problems/_template: PROMPT.md (with the solution.json contract),
   Makefile, and initial source code in src/ that compiles and is a correct but slow starting point.
2. Create checks/<name>/check.py from checks/_template: stages for build, run and correctness, and a
   non-gating metric. Keep the checker outside problems/<name>/.
3. Put any input-data download in problems/<name>/download-input.sh. Do not commit data.
4. Run the starter code through the checker with scripts/check_starter.sh and show the result.
5. Run tests/endpoint/run.sh against my endpoint with --problem <name> and report PASS/WARN/FAIL and the
   token cost.
6. Update AGENT.md (directory map, problem list, environment variables, new output fields) and
   docs/results-schema.md in the same change. Do not leave AGENT.md stale.
Do not modify reference-results/ or other problems.
```
