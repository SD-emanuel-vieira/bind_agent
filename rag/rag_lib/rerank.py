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
    
    # Se desempatan valores igual de relevantes por fecha del archivo
    reranked = tie_break_by_date_in_blocks(reranked, block_size=2)

    return reranked[:top_k]
