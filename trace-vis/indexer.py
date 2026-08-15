"""Builds the browse index over traces/.

Deliberately seeded from `results/runs.jsonl` rather than walking `traces/`:
the tree is ~19 GB / ~1450 round dirs on a cluster filesystem, while runs.jsonl
is a single file with one row per measured round carrying everything the
navigation needs (problem, agent, model, campaign, round, passed, stages, gpu,
turns, steps, token totals) plus `__path` = <round>/.eval/pass_result.json.

A directory scan is kept only as an opt-in fallback for rounds that never
reached the checker and therefore have no row.
"""
import json
import os
import threading
import time

import curation

ROUND_PREFIX = "round-"


def _round_dir_from_path(eval_path):
    """`traces/P/A/M/TS/round-N/.eval/pass_result.json` -> `traces/P/A/M/TS/round-N`."""
    return os.path.dirname(os.path.dirname(eval_path))


def _run_id(problem, agent, model, campaign_ts, rnd):
    return "/".join((problem, agent, model, campaign_ts, f"{ROUND_PREFIX}{rnd}"))


def _metric_from_stages(stages):
    """The non-gating `metric` stage carries each problem's rankable scalar."""
    for st in stages or []:
        if st.get("name") == "metric" and isinstance(st.get("value"), dict):
            v = st["value"]
            return {"name": v.get("metric"), "value": v.get("value"),
                    "unit": v.get("unit"), "higher_better": v.get("higher_better")}
    return None


def _fail_stage(stages):
    for st in stages or []:
        if not st.get("passed") and not st.get("skipped"):
            return st.get("name")
    return None


class TraceIndex:
    def __init__(self, workspace, traces_dirname="traces", runs_jsonl=None):
        self.workspace = os.path.abspath(workspace)
        self.traces_root = os.path.join(self.workspace, traces_dirname)
        self.runs_jsonl = runs_jsonl or os.path.join(self.workspace, "results", "runs.jsonl")
        self._lock = threading.Lock()
        self.runs = {}          # run_id -> run dict
        self.by_problem = {}    # problem -> agent -> [run_id]
        self.by_agent = {}      # agent -> problem -> [run_id]
        self.campaigns = {}     # campaign key -> meta
        self.curated = set()    # run_ids in the "important runs" scope (curation.py)
        self.built_at = None
        self.source_mtime = None
        self.warnings = []
        self.scanned_fs = False

    # ------------------------------------------------------------ build ---

    def build(self, scan_fs=False):
        with self._lock:
            runs = {}
            warnings = []
            mtime = None
            if os.path.isfile(self.runs_jsonl):
                mtime = os.path.getmtime(self.runs_jsonl)
                for row in self._read_runs_jsonl(warnings):
                    runs[row["run_id"]] = row
            else:
                warnings.append(f"{self.runs_jsonl} not found — falling back to a filesystem scan")
                scan_fs = True

            if scan_fs:
                self._scan_filesystem(runs, warnings)
                self.scanned_fs = True

            self.runs = runs
            self.source_mtime = mtime
            self.warnings = warnings
            self._build_pivots()
            self.curated = curation.curated_run_ids(runs.values())
            if runs and not self.curated:
                warnings.append("no run matched the curated scope — showing all runs instead")
            self.built_at = time.time()
        return self

    def _read_runs_jsonl(self, warnings):
        with open(self.runs_jsonl, "r", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    warnings.append(f"runs.jsonl:{lineno} unparseable ({e.msg})")
                    continue
                campaign = d.get("campaign") or ""
                campaign_ts = os.path.basename(campaign)
                problem = d.get("problem")
                agent = d.get("agent")
                model = d.get("model") or d.get("model_raw")
                rnd = d.get("round")
                if not all((problem, agent, model, campaign_ts)) or rnd is None:
                    warnings.append(f"runs.jsonl:{lineno} missing identity fields")
                    continue

                eval_path = d.get("__path")
                if eval_path:
                    round_rel = _round_dir_from_path(eval_path)
                else:
                    round_rel = os.path.join(campaign, f"{ROUND_PREFIX}{rnd}")

                stages = d.get("stages") or []
                rid = _run_id(problem, agent, model, campaign_ts, rnd)
                yield {
                    "run_id": rid,
                    "problem": problem,
                    "agent": agent,
                    "model": model,
                    "model_raw": d.get("model_raw"),
                    "campaign": campaign_ts,
                    "campaign_rel": campaign,
                    "round": rnd,
                    "round_rel": round_rel,
                    "passed": bool(d.get("passed")),
                    "failed_stage": _fail_stage(stages),
                    "metric": _metric_from_stages(stages),
                    "stages": stages,
                    "gpu": d.get("gpu"),
                    "turns": d.get("turns"),
                    "steps": d.get("steps"),
                    "total_input": d.get("total_input"),
                    "total_output": d.get("total_output"),
                    "total_tokens": d.get("total_tokens"),
                    "peak_context": d.get("peak_context"),
                    "compacted": d.get("compacted"),
                    "has_eval": True,
                }

    def _scan_filesystem(self, runs, warnings):
        """Fallback: pick up round dirs that never produced a pass_result row."""
        if not os.path.isdir(self.traces_root):
            warnings.append(f"{self.traces_root} does not exist")
            return
        added = 0
        for problem in sorted(self._subdirs(self.traces_root)):
            for agent in sorted(self._subdirs(os.path.join(self.traces_root, problem))):
                apath = os.path.join(self.traces_root, problem, agent)
                for model in sorted(self._subdirs(apath)):
                    mpath = os.path.join(apath, model)
                    for ts in sorted(self._subdirs(mpath)):
                        tpath = os.path.join(mpath, ts)
                        for rd in sorted(self._subdirs(tpath)):
                            if not rd.startswith(ROUND_PREFIX):
                                continue
                            try:
                                rnd = int(rd[len(ROUND_PREFIX):])
                            except ValueError:
                                continue
                            rid = _run_id(problem, agent, model, ts, rnd)
                            if rid in runs:
                                continue
                            runs[rid] = {
                                "run_id": rid, "problem": problem, "agent": agent,
                                "model": model, "model_raw": model, "campaign": ts,
                                "campaign_rel": os.path.relpath(tpath, self.workspace),
                                "round": rnd,
                                "round_rel": os.path.relpath(os.path.join(tpath, rd), self.workspace),
                                "passed": None, "failed_stage": None, "metric": None,
                                "stages": [], "gpu": None, "turns": None, "steps": None,
                                "total_input": None, "total_output": None,
                                "total_tokens": None, "peak_context": None,
                                "compacted": None, "has_eval": False,
                            }
                            added += 1
        if added:
            warnings.append(f"filesystem scan added {added} round(s) with no eval result")

    @staticmethod
    def _subdirs(path):
        try:
            with os.scandir(path) as it:
                return [e.name for e in it if e.is_dir()]
        except OSError:
            return []

    def _build_pivots(self):
        by_problem, by_agent, campaigns = {}, {}, {}
        for rid, r in self.runs.items():
            by_problem.setdefault(r["problem"], {}).setdefault(r["agent"], []).append(rid)
            by_agent.setdefault(r["agent"], {}).setdefault(r["problem"], []).append(rid)
            ckey = "/".join((r["problem"], r["agent"], r["model"], r["campaign"]))
            c = campaigns.setdefault(ckey, {
                "key": ckey, "problem": r["problem"], "agent": r["agent"],
                "model": r["model"], "campaign": r["campaign"],
                "rel": r["campaign_rel"], "run_ids": [],
            })
            c["run_ids"].append(rid)

        sort_key = lambda rid: self.runs[rid]["round"]
        for d in by_problem.values():
            for k in d:
                d[k].sort(key=sort_key)
        for d in by_agent.values():
            for k in d:
                d[k].sort(key=sort_key)
        for c in campaigns.values():
            c["run_ids"].sort(key=sort_key)
            c.update(self._tally(c["run_ids"]))
        self.by_problem, self.by_agent, self.campaigns = by_problem, by_agent, campaigns

    # ---------------------------------------------------------- queries ---

    def _scoped(self, run_ids, scope):
        """Narrow a run_id list to the requested scope.

        An empty curated set (a workspace whose runs.jsonl matches none of the
        rules) would leave the UI blank, so it degrades to showing everything;
        `summary()` reports which scope was actually served.
        """
        if scope == "all" or not self.curated:
            return list(run_ids)
        return [rid for rid in run_ids if rid in self.curated]

    def _tally(self, run_ids):
        total = len(run_ids)
        graded = [self.runs[r] for r in run_ids if self.runs[r]["passed"] is not None]
        passed = sum(1 for r in graded if r["passed"])
        return {"n_runs": total, "n_graded": len(graded), "n_passed": passed,
                "pass_rate": (passed / len(graded)) if graded else None}

    def maybe_refresh(self):
        """Rebuild if runs.jsonl changed under us (cheap stat, no tree walk)."""
        try:
            m = os.path.getmtime(self.runs_jsonl)
        except OSError:
            return
        if self.source_mtime is not None and m != self.source_mtime:
            self.build(scan_fs=self.scanned_fs)

    def summary(self, scope=curation.DEFAULT_SCOPE):
        problems = []
        for p in sorted(self.by_problem):
            ids = self._scoped([rid for a in self.by_problem[p].values() for rid in a], scope)
            if not ids:
                continue
            agents = sorted({self.runs[rid]["agent"] for rid in ids})
            problems.append(dict(name=p, agents=agents, **self._tally(ids)))
        agents = []
        for a in sorted(self.by_agent):
            ids = self._scoped([rid for p in self.by_agent[a].values() for rid in p], scope)
            if not ids:
                continue
            probs = sorted({self.runs[rid]["problem"] for rid in ids})
            agents.append(dict(name=a, problems=probs, **self._tally(ids)))
        ids = self._scoped(self.runs, scope)
        models = sorted({self.runs[rid]["model"] for rid in ids})
        n_campaigns = len({self._campaign_key(rid) for rid in ids})
        return {
            "problems": problems,
            "agents": agents,
            "models": models,
            "totals": dict(self._tally(ids), n_campaigns=n_campaigns,
                           n_problems=len(problems), n_agents=len(agents),
                           n_models=len(models)),
            "scope": self.scope_info(scope),
            "built_at": self.built_at,
            "warnings": self.warnings,
        }

    def scope_info(self, scope):
        """What the caller asked for, what it got, and the size of each scope."""
        served = "all" if (scope == "all" or not self.curated) else "curated"
        return {"requested": scope, "served": served,
                "n_curated": len(self.curated), "n_all": len(self.runs)}

    def problem_view(self, problem, scope=curation.DEFAULT_SCOPE):
        """problem -> its agents -> their campaigns (the 'pick a problem first' pivot)."""
        if problem not in self.by_problem:
            return None
        agents = []
        for agent in sorted(self.by_problem[problem]):
            ids = self._scoped(self.by_problem[problem][agent], scope)
            if not ids:
                continue
            agents.append(dict(name=agent, **self._tally(ids),
                               campaigns=self._campaigns_for(ids)))
        ids = self._scoped([r for a in self.by_problem[problem].values() for r in a], scope)
        return dict(problem=problem, agents=agents, scope=self.scope_info(scope),
                    **self._tally(ids))

    def agent_view(self, agent, scope=curation.DEFAULT_SCOPE):
        """agent -> its problems -> their campaigns (the 'pick an agent first' pivot)."""
        if agent not in self.by_agent:
            return None
        problems = []
        for problem in sorted(self.by_agent[agent]):
            ids = self._scoped(self.by_agent[agent][problem], scope)
            if not ids:
                continue
            problems.append(dict(name=problem, **self._tally(ids),
                                 campaigns=self._campaigns_for(ids)))
        ids = self._scoped([r for p in self.by_agent[agent].values() for r in p], scope)
        return dict(agent=agent, problems=problems, scope=self.scope_info(scope),
                    **self._tally(ids))

    def _campaign_key(self, run_id):
        r = self.runs[run_id]
        return "/".join((r["problem"], r["agent"], r["model"], r["campaign"]))

    def _campaigns_for(self, run_ids):
        """Group an already-scoped run_id list into campaigns.

        Tallies come from the ids passed in, not from the campaign's full round
        list, so a campaign whose artifact rounds were dropped reports the
        counts the curated scope actually shows.
        """
        groups = {}
        for rid in run_ids:
            groups.setdefault(self._campaign_key(rid), []).append(rid)
        out = []
        for k, ids in groups.items():
            c = self.campaigns[k]
            ids = sorted(ids, key=lambda rid: self.runs[rid]["round"])
            out.append({**{kk: vv for kk, vv in c.items()
                           if kk not in ("run_ids", "n_runs", "n_graded",
                                         "n_passed", "pass_rate")},
                        **self._tally(ids),
                        "n_runs_all": len(c["run_ids"]),
                        "runs": [self.run_brief(rid) for rid in ids]})
        out.sort(key=lambda c: (c["model"], c["campaign"]))
        return out

    def run_brief(self, run_id):
        r = self.runs[run_id]
        d = {k: r[k] for k in ("run_id", "problem", "agent", "model", "campaign",
                               "round", "passed", "failed_stage", "metric",
                               "turns", "steps", "total_tokens", "has_eval")}
        d["curated"] = run_id in self.curated
        return d

    def get(self, run_id):
        return self.runs.get(run_id)

    def round_path(self, run_id):
        """Absolute round dir for a run, guarded against path traversal."""
        r = self.runs.get(run_id)
        if r is None:
            return None
        p = os.path.abspath(os.path.join(self.workspace, r["round_rel"]))
        if p != self.traces_root and not p.startswith(self.traces_root + os.sep):
            return None
        return p if os.path.isdir(p) else None
