"""T06v: one minimal call to model B through the configured backend (Vertex or AI Studio).

Prints backend, model, auth mode and token usage. Never prints keys or the project id.
If the call fails, lists the Flash models the backend offers (to pick the newest stable one).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import llm_extract as L  # noqa: E402


def main():
    backend = L.gemini_backend()
    auth = ("api_key" if os.environ.get("VERTEX_API_KEY") else "ADC") if backend == "vertex" else "api_key"
    print(f"backend={backend}  auth={auth}  location={os.environ.get('GOOGLE_CLOUD_LOCATION', 'global') if backend == 'vertex' else '-'}")
    print(f"model={L.MODEL_B}  config={L.GEN_CONFIG[L.MODEL_B]}")
    t0 = time.time()
    try:
        raw, usage = L._call_gemini('只输出 JSON。', '回复 {"ok": 1}')
    except L.MissingKey as e:
        sys.exit(f"missing configuration: {e}")
    except Exception as e:  # show the error type and message without secrets
        msg = str(e)
        for k in ("VERTEX_API_KEY", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT"):
            if os.environ.get(k):
                msg = msg.replace(os.environ[k], f"<{k}>")
        print(f"CALL FAILED: {type(e).__name__}: {msg[:300]}")
        try:
            names = [m.name for m in L._gemini_client().models.list() if "flash" in (m.name or "")]
            print("flash models on this backend:", sorted(names))
        except Exception as e2:
            print("model list failed:", type(e2).__name__)
        sys.exit(1)
    print(f"reply={raw.strip()[:80]!r}")
    print(f"usage in={usage['in_miss'] + usage['in_hit']} out={usage['out']} (incl. thinking) finish={usage['finish']}")
    print(f"latency={time.time() - t0:.2f}s  cost_at_list_price=${L.cost_usd(L.MODEL_B, usage):.6f}")


if __name__ == "__main__":
    main()
