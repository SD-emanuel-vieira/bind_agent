import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Set
from rag_lib.config import *
from rag_lib.text_utils import safe_json_load
from rag_lib.text_utils import shorten
from rag_lib.llm import call_chat
from rag_lib.business_glossary import BUSINESS_GLOSSARY_V2

def tie_break_by_date_in_blocks(hits, block_size=2):
    out = []
    for i in range(0, len(hits), block_size):
        block = hits[i:i+block_size]
        block.sort(key=lambda h: (h.get("file_date") is not None, h.get("file_date")), reverse=True)
        out.extend(block)
    return out

# ---- Funciones que permiten excluir chunks sin anchor
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


_STOP = {
    "cual", "cuál", "cuales", "cuáles", "como", "cómo", 
    "que", "qué", "quien", "quién", "donde", "dónde", "cuando", "cuándo",
    "son", "es", "fue", "fueron", "ser", "estar", "sido", "siendo",
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "para", "por", "en", "con", "sin", "sobre", "entre", "hacia",
    "y", "o", "ni", "pero", "sino", "aunque",
    "se", "le", "lo", "les", "nos", "me", "te",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
    "al", "a", "ha", "han", "hay", "he", "has",
    "muy", "mas", "más", "menos", "tan", "tanto", "mucho", "poco",
    "si", "no", "ya", "aun", "todavia", "tambien", "solo", "sólo",
}

_MONTHS = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
}


def _detect_glossary_phrases_for_anchors(query: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
    qn = _norm(query)
    found_phrases = []
    consumed_words: Set[str] = set()
    
    all_aliases = []
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = [acronym.lower()] + [a.lower() for a in obj.get("aliases", [])]
        for alias in aliases:
            alias_norm = _norm(alias)
            if alias_norm and " " in alias_norm:
                all_aliases.append((alias_norm, acronym.lower(), len(alias_norm)))
    
    all_aliases.sort(key=lambda x: x[2], reverse=True)
    
    for alias_norm, acronym, _ in all_aliases:
        if alias_norm in qn:
            alias_words = set(alias_norm.split())
            if not alias_words.intersection(consumed_words):
                found_phrases.append((alias_norm, acronym))
                consumed_words.update(alias_words)
    
    return found_phrases, consumed_words

def query_anchors(query: str) -> List[str]:
    """
    Versión mejorada que respeta frases del glosario.
    Compatible con el trace_stage existente.
    """
    qn = _norm(query)
    
    anchors = []
    consumed_words: Set[str] = set()
    
    # 1) Detectar frases del glosario
    phrases, phrase_words = _detect_glossary_phrases_for_anchors(query)
    for phrase, acronym in phrases:
        anchors.append(phrase)
        anchors.append(acronym)
        consumed_words.update(phrase.split())
    
    # 2) Detectar tokens del glosario
    words_in_query = set(qn.split())
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        acr_norm = _norm(acronym)
        if acr_norm in words_in_query and acr_norm not in consumed_words:
            anchors.append(acr_norm)
            consumed_words.add(acr_norm)
    
    # 3) Palabras restantes
    words = re.findall(r"[a-z0-9]+", qn)
    for w in words:
        if w in consumed_words or w in _STOP or w in _MONTHS:
            continue
        if len(w) < 4 and not (w.isdigit() and len(w) == 4):
            continue
        anchors.append(w)
        consumed_words.add(w)
    
    # Deduplicar
    seen: Set[str] = set()
    unique = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    
    return unique

def hit_anchor_score(hit: Dict[str, Any], anchors: List[str]) -> int:
    """
    Versión mejorada con scoring diferenciado para frases vs palabras.
    """
    txt = _norm(hit.get("chunk_text_clean") or hit.get("chunk_text") or "")
    topic = _norm(hit.get("topic_heuristic") or "")
    combined = txt + " " + topic
    
    score = 0
    for anchor in anchors:
        if " " in anchor:
            if anchor in combined:
                score += 3
        else:
            pattern = rf"\b{re.escape(anchor)}\b"
            if re.search(pattern, combined):
                score += 1
    
    return score

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
        # "- Si la pregunta no contiene 'Empresas', 'Corporate', 'Institucional', 'BaaS' o 'Minorista' entonces prioriza cualquier información que no contenga estos valores explicitamente.\n"
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


    return reranked[:top_k]

### -------------- DEBUGGING

import os, re, unicodedata
from collections import Counter

DEBUG_TRACE = os.getenv("RAG_DEBUG_TRACE", "0") == "1"

def _short_id(cid: str, n: int = 10) -> str:
    cid = cid or ""
    return cid[:n]

def trace_stage(stage: str, query: str, hits: list[dict], top: int = 8):
    """Tracer resumido para debuggear orden sin spamear output."""
    if not DEBUG_TRACE:
        return
    if hits is None:
        hits = []

    anchors = query_anchors(query)
    total = len(hits)

    # stats rápidos
    anchor_scores = [hit_anchor_score(h, anchors) for h in hits] if anchors else [0] * total
    n_anchor_pos = sum(1 for s in anchor_scores if s > 0)

    ctype_counts = Counter((h.get("chunk_type") or "NA") for h in hits)
    top_ctypes = ", ".join([f"{k}:{v}" for k, v in ctype_counts.most_common(3)])

    print("\n" + "-" * 110)
    print(f"[TRACE] {stage} | total={total} | anchors={anchors} | anchor_hits>0={n_anchor_pos} | chunk_type={top_ctypes}")

    # ✅ si querés imprimir TODOS, setear env var RAG_DEBUG_TRACE_ALL=1
    trace_all = os.getenv("RAG_DEBUG_TRACE_ALL", "0") == "1"
    limit = total if trace_all else min(top, total)

    for i in range(limit):
        h = hits[i]
        a = anchor_scores[i] if i < len(anchor_scores) else 0
        fd = h.get("file_date") or ""
        pg = h.get("page_num")
        ct = h.get("chunk_type") or ""
        gb = h.get("_glossary_bonus")
        
        # NUEVO: Mostrar el anchor_score calculado internamente
        internal_anchor = h.get("_anchor_score", "?")
        
        cid = _short_id(h.get("chunk_id"))

        path = h.get("path") or ""
        tail = path.split("/")[-1] if path else ""

        # ACTUALIZADO: formato con anchor_score interno
        print(f"  {i+1:02d}) a={a} | a_int={internal_anchor} | g={gb} | {fd} | p={pg} | {ct} | {tail} | cid={cid}")