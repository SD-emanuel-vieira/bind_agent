"""
rerank.py - Módulo de reranking para RAG

ACTUALIZADO:
- El reranker LLM ahora recibe información del anchor_score
- Los chunks con alto anchor_score están "protegidos" y no pueden bajar demasiado
- Mejor balance entre relevancia semántica (LLM) y match exacto (anchors)
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Set
from rag_lib.config import *
from rag_lib.text_utils import safe_json_load, shorten

from rag_lib.vector_search.llm import call_chat
from rag_lib.vector_search.business_glossary import BUSINESS_GLOSSARY_V2


def tie_break_by_date_in_blocks(hits, block_size=2):
    out = []
    for i in range(0, len(hits), block_size):
        block = hits[i:i+block_size]
        block.sort(key=lambda h: (h.get("file_date") is not None, h.get("file_date")), reverse=True)
        out.extend(block)
    return out


# ---- Funciones de normalización y anchors ----
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


def enforce_anchor_priority(query: str, hits: list[dict]) -> list[dict]:
    """Refuerza chunks que tienen anchor score más alto."""
    anchors = query_anchors(query)
    if not anchors or not hits:
        return hits

    max_s = 0
    for h in hits:
        s = hit_anchor_score(h, anchors)
        h["_q_anchor_score"] = s
        if s > max_s:
            max_s = s

    if max_s == 0:
        return hits

    pos = [h for h in hits if h["_q_anchor_score"] > 0]
    neg = [h for h in hits if h["_q_anchor_score"] == 0]
    return pos + neg


# ============================================================
# RERANKING CON LLM - VERSIÓN MEJORADA
# ============================================================

def rerank_with_llm(query: str, hits: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    """
    Reranking con LLM que RESPETA el anchor score.
    
    Estrategia:
    1. Calcular anchor_score para cada hit
    2. Identificar hits "protegidos" (anchor_score >= umbral)
    3. Enviar al LLM para reranking semántico
    4. Post-procesar: asegurar que hits protegidos estén en top posiciones
    """
    if not hits:
        return []
    
    # Calcular anchor scores
    anchors = query_anchors(query)
    for h in hits:
        if "_anchor_score" not in h:
            h["_anchor_score"] = hit_anchor_score(h, anchors) if anchors else 0
    
    # Encontrar el máximo anchor score
    max_anchor = max((h.get("_anchor_score", 0) for h in hits), default=0)
    
    # Umbral para "protección": chunks con score >= 80% del máximo están protegidos
    protection_threshold = max(1, int(max_anchor * 0.8)) if max_anchor > 0 else 0
    
    # Identificar hits protegidos (alto anchor score)
    protected_hits = [h for h in hits if h.get("_anchor_score", 0) >= protection_threshold and protection_threshold > 0]
    protected_ids = {h.get("chunk_id") for h in protected_hits}
    
    # Preparar items para el LLM
    items = []
    for idx, h in enumerate(hits, start=1):
        sid = f"S{idx}"
        snippet = shorten(h.get("chunk_text_clean", ""), RERANK_SNIPPET_CHARS)
        anchor_score = h.get("_anchor_score", 0)
        
        # Incluir anchor_score en metadata para que el LLM lo considere
        meta = (
            f'file_date={h.get("file_date")}, '
            f'path="{h.get("path")}", page_num={h.get("page_num")}, '
            f'topic="{h.get("topic_heuristic") or h.get("topic")}", '
            f'keyword_match_score={anchor_score}'  # NUEVO: informar al LLM
        )
        items.append({"sid": sid, "meta": meta, "snippet": snippet, "hit": h})

    system = (
        "Eres un motor de reranking para recuperación de información corporativa.\n"
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
        "- IMPORTANTE: keyword_match_score indica coincidencia con términos del query.\n"
        "  Chunks con keyword_match_score ALTO deben priorizarse fuertemente.\n"
        "- Prioriza coincidencia literal con palabras clave del query.\n"
        "- En caso de empate de relevancia, prioriza file_date más reciente.\n"
    )

    content = call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role":"system","content":system},{"role":"user","content":"\n".join(user_lines)}],
        temperature=TEMPERATURE_RERANK,
        max_tokens=650
    )

    parsed = safe_json_load(content)
    ranked_sids = parsed.get("ranked_sids", [])

    if not ranked_sids or not isinstance(ranked_sids, list):
        return hits[:top_k]

    sid_to_hit = {f"S{i+1}": items[i]["hit"] for i in range(len(items))}
    llm_reranked = [sid_to_hit[sid] for sid in ranked_sids if sid in sid_to_hit]

    # ============================================================
    # POST-PROCESAMIENTO: Proteger chunks con alto anchor score
    # ============================================================
    if protected_hits:
        # Encontrar hits protegidos que el LLM relegó a posiciones bajas
        protected_in_result = []
        other_in_result = []
        
        for h in llm_reranked:
            if h.get("chunk_id") in protected_ids:
                protected_in_result.append(h)
            else:
                other_in_result.append(h)
        
        # Encontrar hits protegidos que el LLM excluyó completamente
        result_ids = {h.get("chunk_id") for h in llm_reranked}
        protected_excluded = [h for h in protected_hits if h.get("chunk_id") not in result_ids]
        
        # Combinar: protegidos primero, luego el resto del ranking del LLM
        # Ordenar protegidos por anchor_score descendente
        all_protected = protected_in_result + protected_excluded
        all_protected.sort(key=lambda h: h.get("_anchor_score", 0), reverse=True)
        
        # Cuántos protegidos garantizar en el top
        # Regla: al menos la mitad de top_k, máximo todos los protegidos
        min_protected_slots = min(len(all_protected), max(2, top_k // 2))
        
        # Construir resultado final
        final_result = []
        protected_added = set()
        
        # Agregar los top protegidos primero
        for h in all_protected[:min_protected_slots]:
            final_result.append(h)
            protected_added.add(h.get("chunk_id"))
        
        # Agregar el resto del ranking del LLM (excluyendo los ya agregados)
        for h in llm_reranked:
            if h.get("chunk_id") not in protected_added:
                final_result.append(h)
                if len(final_result) >= top_k:
                    break
        
        # Si aún faltan, agregar protegidos restantes
        if len(final_result) < top_k:
            for h in all_protected:
                if h.get("chunk_id") not in protected_added:
                    final_result.append(h)
                    if len(final_result) >= top_k:
                        break
        
        reranked = final_result
    else:
        reranked = llm_reranked

    # Fill up si es necesario
    if len(reranked) < top_k:
        seen = set(h.get("chunk_id") for h in reranked)
        for h in hits:
            if h.get("chunk_id") not in seen:
                reranked.append(h)
                if len(reranked) >= top_k:
                    break

    return reranked[:top_k]


# ============================================================
# DEBUGGING
# ============================================================

import os
from collections import Counter

DEBUG_TRACE = os.getenv("RAG_DEBUG_TRACE", "0") == "1"


def _short_id(cid: str, n: int = 10) -> str:
    cid = cid or ""
    return cid[:n]


def trace_stage(stage: str, query: str, hits: list[dict], top: int = 12):
    """Tracer resumido para debuggear orden."""
    if not DEBUG_TRACE:
        return
    if hits is None:
        hits = []

    anchors = query_anchors(query)
    total = len(hits)

    anchor_scores = [hit_anchor_score(h, anchors) for h in hits] if anchors else [0] * total
    n_anchor_pos = sum(1 for s in anchor_scores if s > 0)

    ctype_counts = Counter((h.get("chunk_type") or "NA") for h in hits)
    top_ctypes = ", ".join([f"{k}:{v}" for k, v in ctype_counts.most_common(3)])

    print("\n" + "-" * 110)
    print(f"[TRACE] {stage} | total={total} | anchors={anchors} | anchor_hits>0={n_anchor_pos} | chunk_type={top_ctypes}")

    trace_all = os.getenv("RAG_DEBUG_TRACE_ALL", "0") == "1"
    limit = total if trace_all else min(top, total)

    for i in range(limit):
        h = hits[i]
        a = anchor_scores[i] if i < len(anchor_scores) else 0
        fd = h.get("file_date") or ""
        pg = h.get("page_num")
        ct = h.get("chunk_type") or ""
        gb = h.get("_glossary_bonus")
        internal_anchor = h.get("_anchor_score", "?")
        cid = _short_id(h.get("chunk_id"))

        path = h.get("path") or ""
        tail = path.split("/")[-1] if path else ""

        print(f"  {i+1:02d}) a={a} | a_int={internal_anchor} | g={gb} | {fd} | p={pg} | {ct} | {tail} | cid={cid}")