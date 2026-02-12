"""
rerank.py - Módulo de reranking para RAG

ACTUALIZADO:
- El reranker LLM ahora recibe información del anchor_score
- Los chunks con alto anchor_score están "protegidos" y no pueden bajar demasiado
- Mejor balance entre relevancia semántica (LLM) y match exacto (anchors)

REFACTORIZADO:
- Las funciones de anchor (query_anchors, hit_anchor_score) ahora se importan
  desde glossary_helper.py para evitar duplicación de código
- glossary_helper tiene la implementación más completa que incluye:
  * Detección de entidades nombradas (Banco Santander → frase única)
  * Mayor score para frases (+5 vs +3)
  * Umbral de longitud más permisivo (3 chars vs 4)
"""

import os
import re
from collections import Counter
from typing import Any, Dict, List, Set

from bind_rag_agent.config import (
    TOP_K_FINAL,
    RERANK_SNIPPET_CHARS,
    TEMPERATURE_RERANK,
    LLM_ENDPOINT,
)
from bind_rag_agent.text_utils import safe_json_load, shorten
from bind_rag_agent.vector_search.llm import call_chat

# ============================================================
# IMPORTAR FUNCIONES DE ANCHOR DESDE GLOSSARY_HELPER
# Esto elimina la duplicación y usa la implementación más robusta
# que incluye detección de entidades nombradas
# ============================================================
from bind_rag_agent.vector_search.glossary_helper import (
    extract_query_anchors,
    compute_anchor_score,
)


# ============================================================
# TIE-BREAKING POR FECHA
# ============================================================

def tie_break_by_date_in_blocks(
    hits: List[Dict[str, Any]], 
    block_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Desempata hits dentro de bloques por fecha de archivo.
    
    Dentro de cada bloque de `block_size` hits consecutivos,
    ordena por file_date descendente (más reciente primero).
    
    Args:
        hits: Lista de hits a procesar
        block_size: Tamaño del bloque para desempate
        
    Returns:
        Lista de hits con desempate aplicado
    """
    out = []
    for i in range(0, len(hits), block_size):
        block = hits[i:i+block_size]
        block.sort(
            key=lambda h: (h.get("file_date") is not None, h.get("file_date")), 
            reverse=True
        )
        out.extend(block)
    return out


# ============================================================
# ENFORCE ANCHOR PRIORITY
# ============================================================

def enforce_anchor_priority(
    query: str, 
    hits: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Refuerza chunks que tienen anchor score más alto.
    
    Mueve los hits con anchor_score > 0 al principio de la lista,
    manteniendo el orden relativo dentro de cada grupo.
    
    Args:
        query: Query del usuario
        hits: Lista de hits a reordenar
        
    Returns:
        Lista con hits de alto anchor score primero
    """
    anchors = extract_query_anchors(query)
    
    if not anchors or not hits:
        return hits

    max_score = 0
    for h in hits:
        score = compute_anchor_score(h, anchors)
        h["_anchor_score"] = score
        max_score = max(max_score, score)

    if max_score == 0:
        return hits

    positive = [h for h in hits if h["_anchor_score"] > 0]
    zero = [h for h in hits if h["_anchor_score"] == 0]
    
    return positive + zero


# ============================================================
# RERANKING CON LLM
# ============================================================

def rerank_with_llm(
    query: str, 
    hits: List[Dict[str, Any]], 
    top_k: int = TOP_K_FINAL
) -> List[Dict[str, Any]]:
    """
    Reranking con LLM que RESPETA el anchor score.
    
    Estrategia:
    1. Calcular anchor_score para cada hit
    2. Identificar hits "protegidos" (anchor_score >= umbral)
    3. Enviar al LLM para reranking semántico
    4. Post-procesar: asegurar que hits protegidos estén en top posiciones
    
    Args:
        query: Query del usuario
        hits: Lista de hits candidatos
        top_k: Número máximo de hits a retornar
        
    Returns:
        Lista rerankeada de hits
    """
    if not hits:
        return []
    
    # ============================================================
    # Paso 1: Calcular anchor scores usando glossary_helper
    # ============================================================
    anchors = extract_query_anchors(query)
    
    for h in hits:
        if "_anchor_score" not in h:
            h["_anchor_score"] = compute_anchor_score(h, anchors) if anchors else 0
    
    # ============================================================
    # Paso 2: Determinar umbral de protección
    # ============================================================
    max_anchor = max((h.get("_anchor_score", 0) for h in hits), default=0)
    
    # Umbral: chunks con score >= 80% del máximo están protegidos
    protection_threshold = max(1, int(max_anchor * 0.8)) if max_anchor >= 1 else 0

    # Identificar hits protegidos
    protected_hits = [
        h for h in hits 
        if h.get("_anchor_score", 0) >= protection_threshold and protection_threshold > 0
    ]
    protected_ids = {h.get("chunk_id") for h in protected_hits}
    
    # ============================================================
    # Paso 3: Preparar items para el LLM
    # ============================================================
    items = []
    for idx, h in enumerate(hits, start=1):
        sid = f"S{idx}"
        snippet = shorten(h.get("chunk_text_clean", ""), RERANK_SNIPPET_CHARS)
        anchor_score = h.get("_anchor_score", 0)
        
        meta = (
            f'file_date={h.get("file_date")}, '
            f'path="{h.get("path")}", page_num={h.get("page_num")}, '
            f'topic="{h.get("topic_heuristic") or h.get("topic")}", '
            f'keyword_match_score={anchor_score}'
        )
        items.append({"sid": sid, "meta": meta, "snippet": snippet, "hit": h})

    # ============================================================
    # Paso 4: Llamar al LLM para reranking
    # ============================================================
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
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_lines)}
        ],
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
    # Paso 5: Post-procesamiento - Proteger hits con alto anchor score
    # ============================================================
    if protected_hits:
        protected_in_result = []
        other_in_result = []
        
        for h in llm_reranked:
            if h.get("chunk_id") in protected_ids:
                protected_in_result.append(h)
            else:
                other_in_result.append(h)
        
        result_ids = {h.get("chunk_id") for h in llm_reranked}
        protected_excluded = [
            h for h in protected_hits 
            if h.get("chunk_id") not in result_ids
        ]
        
        all_protected = protected_in_result + protected_excluded
        all_protected.sort(key=lambda h: h.get("_anchor_score", 0), reverse=True)
        
        min_protected_slots = min(len(all_protected), max(2, top_k // 2))
        
        final_result = []
        protected_added: Set[str] = set()
        
        for h in all_protected[:min_protected_slots]:
            final_result.append(h)
            protected_added.add(h.get("chunk_id"))
        
        for h in llm_reranked:
            if h.get("chunk_id") not in protected_added:
                final_result.append(h)
                if len(final_result) >= top_k:
                    break
        
        if len(final_result) < top_k:
            for h in all_protected:
                if h.get("chunk_id") not in protected_added:
                    final_result.append(h)
                    if len(final_result) >= top_k:
                        break
        
        reranked = final_result
    else:
        reranked = llm_reranked

    # ============================================================
    # Paso 6: Fill up si es necesario
    # ============================================================
    if len(reranked) < top_k:
        seen = {h.get("chunk_id") for h in reranked}
        for h in hits:
            if h.get("chunk_id") not in seen:
                reranked.append(h)
                if len(reranked) >= top_k:
                    break

    return reranked[:top_k]


# ============================================================
# DEBUGGING / TRACING
# ============================================================

DEBUG_TRACE = os.getenv("RAG_DEBUG_TRACE", "0") == "1"


def _short_id(cid: str, n: int = 10) -> str:
    """Trunca un chunk_id para display."""
    return (cid or "")[:n]


def trace_stage(
    stage: str, 
    query: str, 
    hits: List[Dict[str, Any]], 
    top: int = 12
) -> None:
    """
    Tracer resumido para debuggear orden de hits en cada etapa.
    
    Solo se ejecuta si RAG_DEBUG_TRACE=1.
    Muestra anchor scores, glossary bonus, y metadata de cada hit.
    
    Args:
        stage: Nombre de la etapa (ej: "1) retrieve_candidates")
        query: Query del usuario
        hits: Lista de hits a tracear
        top: Número máximo de hits a mostrar (default 12)
    """
    if not DEBUG_TRACE:
        return
    
    if hits is None:
        hits = []

    # Usar las funciones importadas de glossary_helper
    anchors = extract_query_anchors(query)
    total = len(hits)

    # Calcular anchor scores para cada hit
    anchor_scores = [
        compute_anchor_score(h, anchors) for h in hits
    ] if anchors else [0] * total
    
    n_anchor_pos = sum(1 for s in anchor_scores if s > 0)

    # Contar chunk_types
    ctype_counts = Counter((h.get("chunk_type") or "NA") for h in hits)
    top_ctypes = ", ".join([f"{k}:{v}" for k, v in ctype_counts.most_common(3)])

    # Header
    print("\n" + "-" * 110)
    print(
        f"[TRACE] {stage} | total={total} | "
        f"anchors={anchors} | anchor_hits>0={n_anchor_pos} | "
        f"chunk_type={top_ctypes}"
    )

    # Determinar límite de hits a mostrar
    trace_all = os.getenv("RAG_DEBUG_TRACE_ALL", "0") == "1"
    limit = total if trace_all else min(top, total)

    # Mostrar cada hit
    for i in range(limit):
        h = hits[i]
        
        # Anchor score calculado vs almacenado
        a_calculated = anchor_scores[i] if i < len(anchor_scores) else 0
        a_stored = h.get("_anchor_score", "?")
        
        # Metadata
        fd = h.get("file_date") or ""
        pg = h.get("page_num")
        ct = h.get("chunk_type") or ""
        gb = h.get("_glossary_bonus", "?")
        cid = _short_id(h.get("chunk_id"))
        
        path = h.get("path") or ""
        tail = path.split("/")[-1] if path else ""

        print(
            f"  {i+1:02d}) a_calc={a_calculated} | a_stored={a_stored} | "
            f"g_bonus={gb} | {fd} | p={pg} | {ct} | {tail} | cid={cid}"
        )
