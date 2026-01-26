import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import unicodedata
from rag_lib.config import SEGMENTS

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

#### --------- Funciones para filtrar hits por gates (actualmente usado para excluir gráficos)
def _norm_q(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def filter_hits_by_query_gates(query: str, hits: list[dict], gates: dict[str, list[str]]) -> list[dict]:
    """
    Si hit.chunk_type está en `gates`, solo se mantiene si la query contiene alguna keyword permitida.
    """
    qn = _norm_q(query)
    if not qn or not hits:
        return hits

    out = []
    for h in hits:
        ct = (h.get("chunk_type") or "").strip().lower()
        if ct in gates:
            allowed = gates.get(ct) or []
            allowed_norm = [_norm_q(k) for k in allowed if k]
            if not any(k and k in qn for k in allowed_norm):
                continue
        out.append(h)

    # opcional: no vaciar todo si el filtro fue demasiado agresivo
    return out or hits

#### Descarta hits que no contienen el query

def drop_segment_topics_if_query_general(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Si la query NO menciona segmentos (empresa/corporate/institucional/minorista/baas),
    se descartan hits cuyo topic contenga alguno de esos segmentos.
    """
    if not hits:
        return hits

    qn = _norm_q(query)
    query_mentions_segment = any(seg in qn for seg in SEGMENTS)

    # Si la query ya menciona un segmento, no filtramos nada.
    if query_mentions_segment:
        return hits

    out = []
    for h in hits:
        topic = _norm_q(h.get("topic_heuristic") or "")
        if any(seg in topic for seg in SEGMENTS):
            continue
        out.append(h)

    # fallback por si fue demasiado agresivo
    return out or hits