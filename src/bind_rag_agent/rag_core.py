import mlflow.deployments
import re
import unicodedata
from typing import Dict, Any, List, Optional

from bind_rag_agent.config import (
    TOP_K_CANDIDATES, 
    TOP_K_FINAL, 
    MAX_CONTEXT_CHARS,
    RERANK_INPUT_MULTIPLIER,
    RERANK_INPUT_OFFSET,
    RERANK_TIE_BREAK_BLOCK_SIZE,
)
from bind_rag_agent.text_utils import drop_segment_topics_if_query_general
from bind_rag_agent.vector_search.retriever import retrieve_candidates, decompose_multi_segment_query
from bind_rag_agent.vector_search.rerank import (
    tie_break_by_date_in_blocks,
    enforce_anchor_priority,
    rerank_with_llm,
    trace_stage,
    sort_by_source_and_date,
)
from bind_rag_agent.vector_search.glossary_helper import prepare_rerank_candidates_glossary_aware
from bind_rag_agent.vector_search.evidence_handling import build_context, extract_evidence, answer_from_evidence
from bind_rag_agent.sql_search.sql_evidence import answer_sql, get_sql_evidence, is_evidence_usable, build_sql_response
from bind_rag_agent.sql_search.smart_routing import validate_and_route, should_try_sql


# -------------------------
# Orchestrator (end-to-end RAG)
# -------------------------
def answer_with_rag(query: str) -> Dict[str, Any]:

    # =========================================================================
    # FLUJO SQL
    # =========================================================================

    # Pre-validación: ¿vale la pena intentar SQL?
    try_sql, skip_reason = should_try_sql(query)
    
    sql_evidence = None
    
    if try_sql:
        sql_evidence = get_sql_evidence(query)
        
        # Post-validación semántica
        if sql_evidence:
            routing = validate_and_route(
                question=query,
                sql_query=sql_evidence.get('query'),        # ← Cambiar a .get()
                sql_result=sql_evidence.get('raw_data'),    # ← raw_data tiene los datos
                sql_error=sql_evidence.get('error_message') # ← error_message tiene el error
            )
            
            if not routing["use_sql"]:
                sql_evidence = None
    else:
        print(f"ℹ️ Saltando SQL: {skip_reason}")
    
    if sql_evidence and is_evidence_usable(sql_evidence):
        print(answer_sql(query)) # Solo DEBUG
        return build_sql_response(query, sql_evidence)
    
    # =========================================================================
    # FLUJO RAG NORMAL (cuando SQL no tiene respuesta)
    # =========================================================================
    
    # =====================================================================
    # NUEVO: Detectar si es query multi-segmento ANTES del retrieval
    # Esto se usa para:
    #   1. El retrieval ya lo detecta internamente (decompose_multi_segment_query)
    #   2. El sort por fuente+fecha (paso 3.5)
    #   3. Las funciones de evidencia/respuesta (prompts multi-evidencia)
    # =====================================================================
    multi_segment_info = decompose_multi_segment_query(query)
    
    # Posibles candidatos para la respuesta, se filtran por lexical_fallback (importante) y chunk_type:
    candidates = retrieve_candidates(query, k=TOP_K_CANDIDATES) or [] 
    trace_stage("1) retrieve_candidates", query, candidates)

    # Si no se menciona ningún segmento entonces descarta toda evidencia relacionada a cualquier segmento:
    candidates = drop_segment_topics_if_query_general(query, candidates)
    trace_stage("1.1) drop_segment_topics_if_query_general", query, candidates)

    # Soft ordering #1: anchors (gating suave por intención)
    candidates_anchor_sorted = enforce_anchor_priority(query, candidates)
    trace_stage("2) enforce_anchor_priority", query, candidates_anchor_sorted)

    # Soft ordering #2: glossary-aware (estricto por frases)
    TOP_K_RERANK_INPUT = max(TOP_K_FINAL * RERANK_INPUT_MULTIPLIER, TOP_K_FINAL + RERANK_INPUT_OFFSET) 
    hits_for_rerank = prepare_rerank_candidates_glossary_aware(query, candidates, max_input=TOP_K_RERANK_INPUT) 
    trace_stage(f"3) prepare_rerank_candidates_glossary_aware(max_input={TOP_K_RERANK_INPUT})", query, hits_for_rerank)

    # =====================================================================
    # NUEVO (paso 3.5): Sort por prioridad de fuente + fecha
    # Para queries multi-segmento donde todos los scores están empatados,
    # esto agrupa las páginas del Directorio más reciente primero.
    # Para queries normales, solo actúa como tiebreaker (no rompe nada).
    # =====================================================================
    if multi_segment_info:
        hits_for_rerank = sort_by_source_and_date(hits_for_rerank)
        trace_stage("3.5) sort_by_source_and_date (multi-segment)", query, hits_for_rerank)

    # Tie-break SOLO para empates (por file_date) — al final del pre-rerank
    hits_for_rerank_tiebroken = tie_break_by_date_in_blocks(hits_for_rerank, block_size=RERANK_TIE_BREAK_BLOCK_SIZE)
    trace_stage(f"4) tie_break_by_date_in_blocks(block_size={RERANK_TIE_BREAK_BLOCK_SIZE})", query, hits_for_rerank_tiebroken)

    # Reranking en base a las reglas definidas:
    # Para multi-segmento, pedimos más hits al reranker porque necesitamos
    # cubrir N segmentos × ~2 páginas cada uno (ej: 6 segmentos → ~12 hits)
    rerank_k = TOP_K_FINAL + len(multi_segment_info.get("segments", [])) if multi_segment_info else TOP_K_FINAL
    top_hits = rerank_with_llm(query, hits_for_rerank_tiebroken, top_k=rerank_k) or candidates[:rerank_k] 
    trace_stage(f"5) rerank_with_llm(top_k={rerank_k})", query, top_hits)

    # =====================================================================
    # NUEVO (paso 5.5): Re-sort POST-reranker para multi-segmento
    # Usa group_by_document=True para agrupar TODAS las páginas del mismo
    # documento juntas (ej: p24,p25,p26,p27,p28 del Directorio Nov18),
    # independientemente de diferencias menores de score.
    # Esto resuelve el problema de que p26 (score=9) caía al final
    # separada de sus páginas hermanas (score=11).
    # =====================================================================
    if multi_segment_info:
        top_hits = sort_by_source_and_date(top_hits, group_by_document=True)
        trace_stage("5.5) sort_by_source_and_date (post-rerank, grouped)", query, top_hits)

    # Se construye la evidencia:
    # NUEVO: Pasa multi_segment_info para ajustar prompts cuando hay multi-segmento
    evidence = extract_evidence(query, top_hits, multi_segment_info=multi_segment_info)
    print("Se armó la evidencia")
    
    # Se arma la respuesta final:
    # NUEVO: Pasa multi_segment_info para forzar modo multi-evidencia
    answer = answer_from_evidence(query, top_hits, evidence, multi_segment_info=multi_segment_info)
    print("Se armó la respuesta")
    
    # Se construye el contexto:
    _, citations = build_context(top_hits, max_chars=MAX_CONTEXT_CHARS) 
    print("Se armó el contexto")

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence,
        "citations": citations or [],
        "retrieved_candidates": candidates,
        "reranked_hits": top_hits,
        "response_source": "vector_rag",  # Indicador de origen
        "multi_segment": bool(multi_segment_info),  # NUEVO: flag para debug
    }