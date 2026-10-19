#!/usr/bin/env python3
"""Resolver unit test: endpoints.yaml.example resolves for every agent. Run: python3 tests/test_resolve_endpoint.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "lib"))
import resolve_endpoint as rl  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    env = os.path.join(d, ".env")
    open(env, "w").write("EVAL_API_KEY=secret123\n")
    eps, _ = rl.load_config(os.path.join(HERE, "..", "endpoints.yaml.example"), env)
    out = os.path.join(d, "models.json")
    rl.write_models(eps, out)
    mj = json.load(open(out))
    assert mj["providers"]["default"]["apiKey"] == "secret123"
    assert mj["providers"]["default"]["models"][0]["id"] == "org/Model-Name"
    assert oct(os.stat(out).st_mode & 0o777) == "0o600"
    for agent in ("claude", "kimi", "pi", "swe-mini"):
        ep, chosen = rl.select(eps, None, None, agent)
        ex = rl.exports(ep, chosen, agent, out)
        want = "default/org/Model-Name" if agent == "pi" else "org/Model-Name"
        assert ex["EVAL_MODELS"] == want, (agent, ex)
        assert ex["EVAL_PROVIDER"] == "default" and ex["MODELS_CONFIG"] == out
    ep, chosen = rl.select(eps, "default", ["my-model"], "pi")  # by key
    ep, chosen = rl.select(eps, "default", ["default/org/Model-Name"], "pi")  # pi-style provider/id
    try:
        rl.select(eps, "nope", None, "pi")
        raise SystemExit("unknown endpoint accepted")
    except rl.ConfigError:
        pass
print("ok")
