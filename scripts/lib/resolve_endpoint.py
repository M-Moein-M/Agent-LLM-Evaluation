#!/usr/bin/env python3
"""Resolve endpoints.yaml + .env into what the agents consume.

One source of truth (endpoints.yaml, secrets in .env) feeds all four agents:

  --out-models PATH   write a pi-schema models.json containing every endpoint as a
                      provider (the format the agent solve.sh files already read
                      with jq: .providers.<endpoint>.{baseUrl,apiKey,models[]})
  --emit sh           print `export ...` lines for the selected endpoint/models
  --emit json         print the resolved selection as JSON (used by tests/endpoint)
  --preflight         send a 1-token request per selected model using the protocol
                      the agent speaks; exit 1 when any model is unreachable
  --list              show configured endpoints and models (keys never printed)

Selection comes from flags or the environment:
  EVAL_ENDPOINT (alias EVAL_PROVIDER)  endpoint name
  EVAL_MODELS                          comma list of model keys or wire ids
  EVAL_AGENT                           claude | kimi | pi | swe-mini | claude-pro

Importable: load_config(), select(), probe().  Python 3.8+, needs PyYAML.
"""
import argparse
import json
import os
import re
import shlex
import sys
import urllib.error
import urllib.request

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("resolve_endpoint.py: PyYAML is required (pip install pyyaml)")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")  # used as a jq path: .providers.<name>
AGENTS = ("claude", "kimi", "pi", "swe-mini", "claude-pro")
ANTHROPIC_AGENTS = ("claude",)  # speak /v1/messages; the others speak /chat/completions


class ConfigError(Exception):
    pass


def read_env_file(path):
    """KEY=VALUE lines, optional `export `, optional quotes, # comments."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out[k.strip()] = v
    return out


def load_config(config_path=None, env_path=None):
    """Return (endpoints dict, secrets dict). Process env wins over .env."""
    config_path = config_path or os.environ.get("EVAL_ENDPOINTS_FILE") or os.path.join(ROOT, "endpoints.yaml")
    env_path = env_path or os.path.join(ROOT, ".env")
    if not os.path.isfile(config_path):
        raise ConfigError(
            "%s not found. Run: cp endpoints.yaml.example endpoints.yaml  (and cp .env.example .env)" % config_path)
    with open(config_path) as f:
        doc = yaml.safe_load(f) or {}
    eps = doc.get("endpoints")
    if not isinstance(eps, dict) or not eps:
        raise ConfigError("%s: top-level `endpoints:` mapping is missing or empty" % config_path)
    secrets = dict(read_env_file(env_path))
    secrets.update(os.environ)
    out = {}
    for name, ep in eps.items():
        if not NAME_RE.match(str(name)):
            raise ConfigError("endpoint name %r must match %s (it becomes a jq path)" % (name, NAME_RE.pattern))
        if not isinstance(ep, dict) or not ep.get("base_url"):
            raise ConfigError("endpoint %s: base_url is required" % name)
        models = ep.get("models")
        if not isinstance(models, dict) or not models:
            raise ConfigError("endpoint %s: `models:` mapping is missing or empty" % name)
        norm = {}
        for key, m in models.items():
            m = dict(m or {})
            m.setdefault("id", str(key))
            m.setdefault("context_window", 128000)
            m.setdefault("max_output_tokens", 32768)
            norm[str(key)] = m
        key_env = ep.get("api_key_env")
        if key_env:
            api_key = secrets.get(key_env, "")
            if not api_key:
                api_key = None  # reported lazily so --list works without secrets
        else:
            api_key = str(ep.get("api_key", "EMPTY"))
        out[str(name)] = {
            "name": str(name),
            "base_url": str(ep["base_url"]).rstrip("/"),
            "api_key": api_key,
            "api_key_env": key_env,
            "protocols": list(ep.get("protocols") or ["openai", "anthropic"]),
            "models": norm,
        }
    return out, secrets


def require_key(ep):
    if ep["api_key"] is None:
        raise ConfigError("endpoint %s: environment variable %s is not set (put it in .env)"
                          % (ep["name"], ep["api_key_env"]))
    return ep["api_key"]


def models_json(endpoints, strict=True):
    """pi-schema models.json with every endpoint as a provider."""
    providers = {}
    for name, ep in endpoints.items():
        if ep["api_key"] is None and strict:
            continue  # endpoint without a key cannot be used; do not emit a broken provider
        providers[name] = {
            "baseUrl": ep["base_url"],
            "api": "openai-completions",
            "apiKey": ep["api_key"] if ep["api_key"] is not None else "",
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [
                {"id": m["id"], "contextWindow": int(m["context_window"]), "maxTokens": int(m["max_output_tokens"])}
                for m in ep["models"].values()
            ],
        }
    return {"providers": providers}


def write_models(endpoints, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # holds API keys
    with os.fdopen(fd, "w") as f:
        json.dump(models_json(endpoints), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def select(endpoints, endpoint=None, models=None, agent=None):
    """Pick an endpoint and models. `models` is a list of keys or wire ids."""
    if not endpoint:
        if len(endpoints) == 1:
            endpoint = next(iter(endpoints))
        else:
            raise ConfigError("several endpoints configured (%s): set EVAL_ENDPOINT" % ", ".join(endpoints))
    if endpoint not in endpoints:
        raise ConfigError("endpoint %r not in endpoints.yaml (have: %s)" % (endpoint, ", ".join(endpoints)))
    ep = endpoints[endpoint]
    require_key(ep)
    if agent and agent in ANTHROPIC_AGENTS and "anthropic" not in ep["protocols"]:
        raise ConfigError("endpoint %s does not list the `anthropic` protocol but agent %s needs /v1/messages"
                          % (endpoint, agent))
    wanted = []
    for tok in models or list(ep["models"]):
        tok = tok.strip()
        if not tok:
            continue
        if "/" in tok and tok.split("/", 1)[0] == endpoint:  # pi-style provider/id
            tok = tok.split("/", 1)[1]
        hit = next((k for k, m in ep["models"].items() if tok in (k, m["id"])), None)
        if hit is None:
            raise ConfigError("model %r not defined under endpoint %s (have: %s)" % (tok, endpoint, ", ".join(ep["models"])))
        wanted.append(hit)
    chosen = []
    for key in wanted:
        m = ep["models"][key]
        wire = (m.get("agent_ids") or {}).get(agent, m["id"]) if agent else m["id"]
        chosen.append({"key": key, "id": wire, "trace_tag": m.get("trace_tag"),
                       "reasoning": bool(m.get("reasoning", False)),
                       "context_window": int(m["context_window"]),
                       "max_output_tokens": int(m["max_output_tokens"])})
    return ep, chosen


def exports(ep, chosen, agent, models_path):
    ids = ["%s/%s" % (ep["name"], c["id"]) if agent == "pi" else c["id"] for c in chosen]
    env = {
        "EVAL_ENDPOINT": ep["name"],
        "EVAL_PROVIDER": ep["name"],
        "EVAL_MODELS": ",".join(ids),
        "MODELS_CONFIG": models_path,
    }
    if len(chosen) == 1 and chosen[0]["trace_tag"] and not os.environ.get("EVAL_MODEL_TAG"):
        env["EVAL_MODEL_TAG"] = chosen[0]["trace_tag"]
    return env


def probe(ep, model_id, agent=None, timeout=30):
    """1-token request using the agent's protocol. Returns (ok, http_status, detail)."""
    key = require_key(ep)
    if agent in ANTHROPIC_AGENTS:
        base = ep["base_url"][:-3] if ep["base_url"].endswith("/v1") else ep["base_url"]
        url = base + "/v1/messages"
        body = {"model": model_id, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
        headers = {"Authorization": "Bearer " + key, "x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url = ep["base_url"] + "/chat/completions"
        body = {"model": model_id, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
        headers = {"Authorization": "Bearer " + key}
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200, r.status, url
    except urllib.error.HTTPError as e:
        return False, e.code, "%s: %s" % (url, e.read(200).decode("utf-8", "replace").strip())
    except Exception as e:  # DNS, refused, timeout
        return False, 0, "%s: %s" % (url, e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--env-file")
    ap.add_argument("--endpoint", default=os.environ.get("EVAL_ENDPOINT") or os.environ.get("EVAL_PROVIDER"))
    ap.add_argument("--models", default=os.environ.get("EVAL_MODELS"), help="comma list of keys or wire ids")
    ap.add_argument("--agent", default=os.environ.get("EVAL_AGENT"))
    ap.add_argument("--out-models", help="write pi-schema models.json here")
    ap.add_argument("--emit", choices=["sh", "json"])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    try:
        eps, _ = load_config(a.config, a.env_file)
        if a.list:
            for n, ep in eps.items():
                print("%s  %s  key=%s  protocols=%s" % (n, ep["base_url"],
                      "set" if ep["api_key"] is not None else "MISSING(%s)" % ep["api_key_env"], ",".join(ep["protocols"])))
                for k, m in ep["models"].items():
                    print("    %-28s id=%s ctx=%s out=%s" % (k, m["id"], m["context_window"], m["max_output_tokens"]))
            return 0
        models_path = a.out_models or os.environ.get("MODELS_CONFIG") or os.path.join(ROOT, ".run", "models.json")
        if a.out_models:
            write_models(eps, a.out_models)
        if not (a.emit or a.preflight):
            return 0
        ep, chosen = select(eps, a.endpoint, a.models.split(",") if a.models else None, a.agent)
        if a.emit == "sh":
            for k, v in exports(ep, chosen, a.agent, models_path).items():
                print("export %s=%s" % (k, shlex.quote(v)))
        elif a.emit == "json":
            print(json.dumps({"endpoint": ep["name"], "base_url": ep["base_url"], "api_key": ep["api_key"],
                              "protocols": ep["protocols"], "models": chosen}, indent=2))
        if a.preflight:
            failed = []
            sys.stderr.write("Preflight: %s (%s protocol)\n" % (ep["base_url"], "anthropic" if a.agent in ANTHROPIC_AGENTS else "openai"))
            for c in chosen:
                ok, status, detail = probe(ep, c["id"], a.agent)
                sys.stderr.write("  [%s] %s (HTTP %s)%s\n" % ("ok" if ok else "FAIL", c["id"], status, "" if ok else "  " + detail))
                if not ok:
                    failed.append(c["id"])
            if failed:
                sys.stderr.write("ERROR: unreachable: %s. Check base_url, api key and that the model is served "
                                 "(skip with EVAL_SKIP_PREFLIGHT=1).\n" % ", ".join(failed))
                return 1
        return 0
    except ConfigError as e:
        sys.stderr.write("ERROR: %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
