"""LLM extraction pipeline (T06a): input building, cached API calls, JSON validation, cost ledger.

Keys come from paperB/.env (DEEPSEEK_API_KEY; Gemini via Vertex AI: GOOGLE_CLOUD_PROJECT [+ VERTEX_API_KEY
or ADC], or AI Studio: GEMINI_API_KEY) and are never logged.
Every call is cached at data/llm/{model}/{sha256(model|prompt_version|input)}.json; a cached
result is returned without calling the API, so interrupted runs resume where they stopped.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from src import paths
from src.extract_rules import normalize
from src.segment import find_spans

PROMPTS = paths.ROOT / "prompts"
LLM_DIR = paths.DATA / "llm"
MODEL_A = "deepseek-flash"      # DeepSeek-V4.1-Flash (api-docs.deepseek.com, checked 2026-09-30)
MODEL_B = "gemini-3.8-flash"    # newest stable Gemini Flash (ai.google.dev pricing, checked 2026-09-30)
# USD per 1M tokens. DeepSeek off-peak / peak; Gemini paid standard tier (upper bound: free tier is $0)
PRICES = {
    MODEL_A: {"offpeak": {"hit": 0.003, "miss": 0.15, "out": 0.60},
              "peak": {"hit": 0.006, "miss": 0.30, "out": 1.20}},
    MODEL_B: {"standard": {"in": 0.75, "out": 3.75}, "batch": {"in": 0.375, "out": 1.875}},
}
PILOT_BUDGET_USD = 1.0
# Generation settings (part of the cache key). Thinking tokens are billed as output on both APIs;
# the extraction is a fixed-format JSON task, so DeepSeek thinking is disabled and Gemini uses the
# lowest level offered for gemini-3.8-flash ("low"; there is no "minimal"/off for this model).
GEN_CONFIG = {MODEL_A: "temp0|json|thinking=disabled", MODEL_B: "temp0|json|thinking_level=low"}

# ----------------------------------------------------------------------------- field specs
YN = {"1", "0"}
FIELDS = {
    "origin_text": "text", "residence_text": "text",
    "nonlocal_origin_model": {"1", "0", "NA"}, "residence_local_model": {"1", "0", "NA"},
    "status_at_judgment": {"在押", "取保", "监视居住", "其他", "NA"},
    "ever_bail": YN, "plea_formal": YN, "jiejie": YN,
    "procedure": {"速裁", "简易", "普通", "NA"}, "counsel_any": YN,
    "penalty_type": {"有期徒刑", "拘役", "管制", "单处罚金", "免予刑事处罚", "其他"},
    "term_months": "number", "probation": YN,
}
# v2 (T06a2): full-text word fields (plea_formal, jiejie, speedy, counsel, procedure) are rules only
FIELDS_V2 = {k: FIELDS[k] for k in ("origin_text", "residence_text", "status_at_judgment", "ever_bail",
                                    "penalty_type", "term_months", "probation")}
ORIGIN_FIELDS ={"origin_text": "text", "residence_text": "text", "segment": {"当事人", "指控", "查明", "NA"}}
EVIDENCE_MAX = 20


def load_prompt(name):
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    return text, name  # the file name carries the version (…_v1)


# ----------------------------------------------------------------------------- inputs
def build_input(text, mode="core"):
    """Segments sent to the model. core: header<=1200, reasoning<=800, verdict<=600.
    origin: header<=1200, charge<=800, facts<=800. Fallback: first 2000 chars."""
    t = normalize(text)
    sp = dict(find_spans(t))
    if sp.get("header"):
        # the party block runs up to the next segment, so the procedural sentences
        # ("适用简易程序", "辩护人…到庭") that follow the defendant paragraphs are included
        nxt = [v[0] for k, v in sp.items() if v and k != "header" and v[0] >= sp["header"][1]]
        sp["header"] = (0, min(nxt) if nxt else len(t))
    plan =([("当事人段", "header", 1200), ("本院认为段", "reasoning", 800), ("判决段", "verdict", 600)]
            if mode == "core" else
            [("当事人段", "header", 1200), ("指控段", "charge", 800), ("查明段", "facts", 800)])
    parts = [f"【{lab}】\n{t[sp[k][0]:sp[k][1]][:n]}" for lab, k, n in plan if sp.get(k)]
    if mode == "core" and not (sp.get("header") and sp.get("verdict")):
        parts = [f"【全文开头】\n{t[:2000]}"]
    if mode == "origin" and not parts:
        parts = [f"【全文开头】\n{t[:2000]}"]
    body = "\n\n".join(parts)
    return body, len(body)


# ----------------------------------------------------------------------------- parsing & validation
def parse_json(raw):
    """Parse a model reply (tolerates ```json fences). Raises ValueError on failure."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"json: {e}") from e


def _norm_value(v):
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(v)
    v = str(v).strip()
    return "NA" if v.upper() in ("NA", "N/A", "NULL", "NONE", "") else v


def validate(obj, spec=FIELDS):
    """Return ({field: value}, {field: evidence}, [errors]). Invalid values become None."""
    vals, evid, errs = {}, {}, []
    for f, allowed in spec.items():
        item = obj.get(f)
        if isinstance(item, dict):
            v, e = item.get("value"), item.get("evidence", "")
        else:
            v, e = item, ""
        if f not in obj:
            errs.append(f"missing:{f}")
            vals[f], evid[f] = None, ""
            continue
        v = _norm_value(v)
        if allowed == "number":
            if v != "NA":
                try:
                    v = str(float(re.sub(r"[^\d.]", "", v)))
                except ValueError:
                    errs.append(f"invalid:{f}")
                    v = None
        elif allowed != "text" and v not in allowed:
            errs.append(f"invalid:{f}")
            v = None
        vals[f] = v
        evid[f] = (e or "")[:EVIDENCE_MAX * 2]
    return vals, evid, errs


# ----------------------------------------------------------------------------- cache
def cache_key(model, prompt_version, user_input):
    cfg = GEN_CONFIG.get(model, "")
    return hashlib.sha256(f"{model}|{prompt_version}|{cfg}|{user_input}".encode("utf-8")).hexdigest()


def cache_path(model, key):
    return LLM_DIR / model / f"{key}.json"


# ----------------------------------------------------------------------------- clients
class MissingKey(RuntimeError):
    pass


def _env(name):
    from dotenv import load_dotenv
    load_dotenv(paths.ROOT / ".env")
    v = os.environ.get(name)
    if not v:
        raise MissingKey(f"{name} is not set. Add it to paperB/.env (see T06a task file); the script will not fill it in.")
    return v


def is_peak(ts=None):
    t = time.gmtime(ts)
    return t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


def _call_deepseek(system, user):
    from openai import OpenAI
    client = OpenAI(api_key=_env("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=120)
    r = client.chat.completions.create(
        model=MODEL_A, temperature=0, response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    u = r.usage
    hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    if miss is None:
        miss = u.prompt_tokens - hit
    return r.choices[0].message.content, {"in_hit": hit, "in_miss": miss, "out": u.completion_tokens,
                                          "finish": r.choices[0].finish_reason}


_GEMINI_CLIENT = None


def gemini_backend():
    """'vertex' (Vertex AI / Gemini Enterprise Agent Platform, paid by Cloud billing incl. the $300 trial)
    or 'aistudio' (Gemini API key). Chosen by GEMINI_BACKEND in .env; defaults to vertex when a
    Google Cloud project or Vertex key is configured."""
    from dotenv import load_dotenv
    load_dotenv(paths.ROOT / ".env")
    b = (os.environ.get("GEMINI_BACKEND") or "").strip().lower()
    if b in {"vertex", "aistudio"}:
        return b
    return "vertex" if (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_API_KEY")) else "aistudio"


def _gemini_client():
    """Build the google-genai client once.

    vertex:   VERTEX_API_KEY set  -> genai.Client(vertexai=True, api_key=...)
              otherwise           -> Application Default Credentials (gcloud auth application-default login)
                                     with GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (default 'global').
    aistudio: GEMINI_API_KEY.
    """
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    from google import genai
    if gemini_backend() == "vertex":
        key = os.environ.get("VERTEX_API_KEY")
        if key:
            _GEMINI_CLIENT = genai.Client(vertexai=True, api_key=key)
        else:
            _GEMINI_CLIENT = genai.Client(vertexai=True, project=_env("GOOGLE_CLOUD_PROJECT"),
                                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    else:
        _GEMINI_CLIENT = genai.Client(api_key=_env("GEMINI_API_KEY"))
    return _GEMINI_CLIENT


def _call_gemini(system, user):
    from google.genai import types
    client = _gemini_client()
    r = client.models.generate_content(
        model=MODEL_B, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0,
                                           response_mime_type="application/json",
                                           thinking_config=types.ThinkingConfig(thinking_level="low"),
                                           automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
    um = r.usage_metadata
    out = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    finish = r.candidates[0].finish_reason.name if r.candidates and r.candidates[0].finish_reason else None
    return r.text or "", {"in_hit": getattr(um, "cached_content_token_count", 0) or 0,
                          "in_miss": (um.prompt_token_count or 0) - (getattr(um, "cached_content_token_count", 0) or 0),
                          "out": out, "finish": finish, "backend": gemini_backend()}


CALLERS = {MODEL_A: _call_deepseek, MODEL_B: _call_gemini}


def cost_usd(model, usage, peak=False):
    if model == MODEL_A:
        p = PRICES[MODEL_A]["peak" if peak else "offpeak"]
        return (usage["in_hit"] * p["hit"] + usage["in_miss"] * p["miss"] + usage["out"] * p["out"]) / 1e6
    p = PRICES[MODEL_B]["standard"]
    return ((usage["in_hit"] + usage["in_miss"]) * p["in"] + usage["out"] * p["out"]) / 1e6


class Ledger:
    """Running cost (priced at paid rates; Gemini free tier makes the real cost lower)."""

    def __init__(self, budget=PILOT_BUDGET_USD):
        self.budget, self.spent, self.calls = budget, 0.0, 0

    def check(self):
        if self.spent >= self.budget:
            raise RuntimeError(f"budget reached: ${self.spent:.4f} >= ${self.budget}")


def call_cached(model, prompt_name, user_input, ledger=None, retries=3, caller=None):
    """Return the cached record, or call the API, store and return it.

    record = {model, prompt, key, raw, usage, latency_s, peak, cost_usd, error, ts}
    """
    system, version = load_prompt(prompt_name)
    key = cache_key(model, version, user_input)
    path = cache_path(model, key)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["cached"] = True
        return rec
    if ledger is not None:
        ledger.check()
    caller = caller or CALLERS[model]
    err, raw, usage, t0 = None, None, None, time.time()
    for attempt in range(retries):
        try:
            raw, usage = caller(system, user_input)
            err = None
            break
        except MissingKey:
            raise
        except Exception as e:  # timeouts, 429, 5xx: back off and retry
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2 * (attempt + 1) ** 2)
    peak = is_peak()
    rec = {"model": model, "prompt": version, "config": GEN_CONFIG.get(model), "key": key, "raw": raw, "usage": usage,
           "latency_s": round(time.time() - t0, 2), "peak": peak,
           "cost_usd": cost_usd(model, usage, peak) if usage else 0.0, "error": err,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if ledger is not None:
        ledger.spent += rec["cost_usd"]
        ledger.calls += 1
    if err is None:  # failed calls are not cached, so a rerun retries them
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    rec["cached"] = False
    return rec
