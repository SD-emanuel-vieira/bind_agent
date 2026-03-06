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
    DEDUP_ENABLED,
    DEDUP_QUOTA_OLD_FILE,
)
from bind_rag_agent.text_utils import drop_segment_topics_if_query_general
from bind_rag_agent.vector_search.retriever import retrieve_candidates, decompose_multi_segment_query
from bind_rag_agent.vector_search.rerank import (
    tie_break_by_date_in_blocks,
    enforce_anchor_priority,
    rerank_with_llm,
    trace_stage,
    sort_by_source_and_date,
    collapse_cross_file_duplicates,
)
from bind_rag_agent.vector_search.glossary_helper import prepare_rerank_candidates_glossary_aware
from bind_rag_agent.vector_search.evidence_handling import build_context, extract_evidence, answer_from_evidence
from bind_rag_agent.sql_search.sql_evidence import answer_sql, get_sql_evidence, is_evidence_usable, build_sql_response
from bind_rag_agent.sql_search.smart_routing import validate_and_route, should_try_sql
from bind_rag_agent.sql_search.sql_trace import trace_sql
from bind_rag_agent.token_counter import token_counter
from bind_rag_agent.delta_logger import log_rag_to_delta

# -------------------------
# Orchestrator (end-to-end RAG)
# -------------------------
def answer_with_rag(query: str) -> Dict[str, Any]:

    # Reset token counter para esta ejecución
    token_counter.reset()

    # =========================================================================
    # FLUJO SQL
    # =========================================================================

    # Pre-validación: ¿vale la pena intentar SQL?
    try_sql, skip_reason = should_try_sql(query)
    
    trace_sql(
        "1) should_try_sql",
        query=query,
        extra={"try_sql": try_sql, "skip_reason": skip_reason},
    )
    
    sql_evidence = None
    
    if try_sql:
        sql_evidence = get_sql_evidence(query)
        
        trace_sql(
            "2) get_sql_evidence",
            query=query,
            sql=sql_evidence.get('query'),
            sql_evidence=sql_evidence,
        )
        
        # Post-validación semántica
        if sql_evidence:
            routing = validate_and_route(
                question=query,
                sql_query=sql_evidence.get('query'),
                sql_result=sql_evidence.get('raw_data'),
                sql_error=sql_evidence.get('error_message')
            )
            
            trace_sql(
                "3) validate_and_route",
                query=query,
                routing=routing,
            )
            
            if not routing["use_sql"]:
                sql_evidence = None
    else:
        trace_sql(
            "1) should_try_sql → SKIP",
            query=query,
            skip_reason=skip_reason,
        )
    
    if sql_evidence and is_evidence_usable(sql_evidence):
        trace_sql(
            "4) SQL RESULT → Usando respuesta SQL",
            query=query,
            extra={"answer_preview": (sql_evidence.get("answer") or "")[:200]},
        )
        result = build_sql_response(query, sql_evidence)
        result.update(token_counter.get_totals())
        log_rag_to_delta(result) # se guarda el resultado en la tabla de logs del RAG
        return result
    
    trace_sql(
        "4) SQL RESULT → Fallback a vectorial",
        query=query,
        extra={"reason": "SQL no produjo evidencia usable" if try_sql else skip_reason},
    )

    # =========================================================================
    # Si el flujo SQL se intentó pero falló → NO caer al vectorial
    # =========================================================================
    if try_sql:
        error_detail = ""
        if sql_evidence:
            error_detail = sql_evidence.get("error_message") or "sin detalle"
        else:
            error_detail = "No se obtuvo evidencia SQL"

        trace_sql(
            "4) SQL FAILED → Respuesta de error (sin fallback vectorial)",
            query=query,
            extra={"error_detail": error_detail},
        )

        no_data_answer = (
            "No se encontró información en las matrices de resultados "
            "de cliente, producto y oficiales. "
            "Es posible que los datos solicitados no estén disponibles "
            "o que haya un problema temporal de acceso. "
            "Por favor, intentá reformular la consulta o contactá al administrador."
        )

        result = {
            "query": query,
            "answer": no_data_answer,
            "evidence": {
                "answerable": False,
                "source": "sql_table",
                "source_type": "structured_sql",
                "raw_data": None,
                "key_points": [],
                "missing": [error_detail],
            },
            "citations": [],
            "retrieved_candidates": [],
            "reranked_hits": [],
            "response_source": "sql_table_error",
            **token_counter.get_totals(),
        }
        log_rag_to_delta(result)
        return result

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

    # Deduplicar chunks estructuralmente idénticos entre archivos mensuales
    # (misma tabla con distintos valores numéricos → mantener solo la versión más reciente)
    if DEDUP_ENABLED:
        candidates = collapse_cross_file_duplicates(candidates, quota_old_file=DEDUP_QUOTA_OLD_FILE)
        trace_stage("1.5) collapse_cross_file_duplicates", query, candidates)

    # Soft ordering #1: anchors (gating suave por intención)
    candidates_anchor_sorted = enforce_anchor_priority(query, candidates)
    trace_stage("2) enforce_anchor_priority", query, candidates_anchor_sorted)

    # Soft ordering #2: glossary-aware (estricto por frases)
    TOP_K_RERANK_INPUT = max(TOP_K_FINAL * RERANK_INPUT_MULTIPLIER, TOP_K_FINAL + RERANK_INPUT_OFFSET) 
    hits_for_rerank = prepare_rerank_candidates_glossary_aware(query, candidates, max_input=TOP_K_RERANK_INPUT) 
    trace_stage(f"3) prepare_rerank_candidates_glossary_aware(max_input={TOP_K_RERANK_INPUT})", query, hits_for_rerank)

    # Reranking en base a las reglas definidas:
    # Para multi-segmento, pedimos más hits al reranker porque necesitamos
    # cubrir N segmentos × ~2 páginas cada uno (ej: 6 segmentos → ~12 hits)
    rerank_k = TOP_K_FINAL + len(multi_segment_info.get("segments", [])) if multi_segment_info else TOP_K_FINAL
    top_hits_rerank = rerank_with_llm(query, hits_for_rerank, top_k=rerank_k) or candidates[:rerank_k] 
    trace_stage(f"4) rerank_with_llm(top_k={rerank_k})", query, top_hits_rerank)

    # # Tie-break SOLO para empates (por file_date) — al final del pre-rerank
    top_hits = sort_by_source_and_date(top_hits_rerank, group_by_document=True)
    trace_stage("4.5) sort_by_source_and_date (post-rerank, recency)", query, top_hits)

    # =====================================================================

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

    result = {
        "query": query,
        "answer": answer,
        "evidence": evidence,
        "citations": citations or [],
        "retrieved_candidates": candidates,
        "reranked_hits": top_hits,
        "response_source": "vector_rag",  # Indicador de origen
        "multi_segment": bool(multi_segment_info),  # NUEVO: flag para debug
        **token_counter.get_totals(),
    }

    log_rag_to_delta(result) # se guarda el resultado en la tabla de logs del RAG

    return result