import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from rag_lib.config import *
from rag_lib.text_utils import safe_json_load
from rag_lib.text_utils import shorten
from rag_lib.llm import call_chat

def tie_break_by_date_in_blocks(hits, block_size=2):
    out = []
    for i in range(0, len(hits), block_size):
        block = hits[i:i+block_size]
        block.sort(key=lambda h: (h.get("file_date") is not None, h.get("file_date")), reverse=True)
        out.extend(block)
    return out

# ---- Funciones que permiten excluir chunks sin anchor
_STOP = {"cual","cuál","cuales","cuáles","son","es","de","del","la","el","los","las",
         "para","por","en","un","una","y","o","que","qué"}

_MONTHS = {"enero","febrero","marzo","abril","mayo","junio","julio","agosto",
           "septiembre","octubre","noviembre","diciembre"}

def _norm_simple(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def query_anchors(query: str) -> list[str]:
    qn = _norm_simple(query)
    toks = re.findall(r"[a-z]{4,}", qn)
    out, seen = [], set()
    for t in toks:
        if t in _STOP: 
            continue
        if t in _MONTHS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def hit_anchor_score(hit: dict, anchors: list[str]) -> int:
    txt = _norm_simple(hit.get("chunk_text_clean") or hit.get("chunk_text") or "")
    return sum(1 for a in anchors if a in txt)

# -------- Se refuerza aquellos chunks que tiene un anchor score más alto
def enforce_anchor_priority(query: str, hits: list[dict]) -> list[dict]:
    anchors = query_anchors(query)
    if not anchors or not hits:
        return hits

    # calcula score
    max_s = 0
    for h in hits:
        s = hit_anchor_score(h, anchors)
        h["_q_anchor_score"] = s
        if s > max_s:
            max_s = s

    # si nadie matchea anchors, no tocamos nada
    if max_s == 0:
        return hits

    # stable: conserva el orden relativo que el LLM ya decidió dentro de cada grupo
    pos = [h for h in hits if h["_q_anchor_score"] > 0]
    neg = [h for h in hits if h["_q_anchor_score"] == 0]
    return pos + neg
# -------------------------
# CELL 6: Reranking with LLM
# -------------------------
def rerank_with_llm(query: str, hits: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    if not hits:
        return []

    items = []
    for idx, h in enumerate(hits, start=1):
        sid = f"S{idx}"
        snippet = shorten(h.get("chunk_text_clean", ""), RERANK_SNIPPET_CHARS)
        meta = (
            f'file_date={h.get("file_date")}, '
            f'path="{h.get("path")}", page_num={h.get("page_num")}, topic="{h.get("topic")}"'
        )
        items.append({"sid": sid, "meta": meta, "snippet": snippet, "hit": h})

    system = (
        "Eres un motor de reranking para recuperación de información.\n"
        "Ordena extractos por relevancia para responder la pregunta.\n"
        "Devuelve SOLO JSON válido, sin texto adicional."
    )

    user_lines = [f"Pregunta:\n{query}\n", "Candidatos:"]
    for it in items:
        user_lines.append(f"{it['sid']} | {it['meta']}\n{it['snippet']}\n")

    user_lines.append(
        "Devuelve JSON EXACTO:\n"
        "{\n"
        '  "ranked_sids": ["S3","S1",...],\n'
        '  "reasons": {"S3":"...", "S1":"..."}\n'
        "}\n"
        f"- ranked_sids debe incluir como máximo {top_k} ids.\n"
        "- Prioriza coincidencia literal con palabras clave del query si existe.\n"
        "- En caso de empate de relevancia, prioriza file_date más reciente.\n"
    )

    content = call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role":"system","content":system},{"role":"user","content":"\n".join(user_lines)}],
        temperature=TEMPERATURE_RERANK,
        max_tokens=650
    )

    parsed = safe_json_load(content)
    ranked = parsed.get("ranked_sids", [])

    if not ranked or not isinstance(ranked, list):
        return hits[:top_k]

    sid_to_hit = {f"S{i+1}": items[i]["hit"] for i in range(len(items))}
    reranked = [sid_to_hit[sid] for sid in ranked if sid in sid_to_hit]

    # fill up if needed
    if len(reranked) < top_k:
        seen = set(h.get("chunk_id") for h in reranked)
        for h in hits:
            if h.get("chunk_id") not in seen:
                reranked.append(h)
                if len(reranked) >= top_k:
                    break
    
    # # Se desempatan valores igual de relevantes por fecha del archivo
    # reranked = tie_break_by_date_in_blocks(reranked, block_size=2)

    # # Se manda para abajo los chunks que no tienen anchor
    # reranked = enforce_anchor_priority(query, reranked)

    return reranked[:top_k]
