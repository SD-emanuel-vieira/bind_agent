import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os

# -------------------------
# # Helper functions (text cleaning + parsing)
# -------------------------
def strip_chunk_prefix(text: str) -> str:
    """Remove [SOURCE]/[TOPIC] prefix (if your gold chunks included it)."""
    if not text:
        return ""
    return re.sub(r"(?s)^\[SOURCE:[^\]]*\]\s*\n\[TOPIC:[^\]]*\]\s*\n\s*", "", text).strip()

def shorten(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else (t[:n].rstrip() + "…")

def safe_json_load(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        s2 = s[i:j+1]
        try:
            return json.loads(s2)
        except Exception:
            return {}
    return {}

def extract_chat_content(resp: Any) -> str:
    """Best-effort extraction of chat content from serving response."""
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            msg = resp["choices"][0].get("message", {})
            return msg.get("content", "") or ""
        if "predictions" in resp and resp["predictions"]:
            p0 = resp["predictions"][0]
            if isinstance(p0, dict) and "content" in p0:
                return p0["content"]
            if isinstance(p0, str):
                return p0
    return str(resp)


def parse_vs_similarity_response(res: Any) -> List[Dict[str, Any]]:
    """Normalize Vector Search similarity_search response to list[dict]."""
    if isinstance(res, dict):
        r = res.get("result") or res
        cols = r.get("columns")
        data = r.get("data_array") or r.get("data") or []
        if cols and data:
            out = []
            for row in data:
                out.append({c: row[i] for i, c in enumerate(cols)})
            return out
    if isinstance(res, list):
        return res
    return []