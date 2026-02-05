import mlflow.deployments

from rag_lib.secret_functions import *
from rag_lib.config import *
from rag_lib.text_utils import *

from rag_lib.retriever import retrieve_candidates
from rag_lib.rerank import tie_break_by_date_in_blocks,enforce_anchor_priority,rerank_with_llm,trace_stage
from rag_lib.glossary_helper import prepare_rerank_candidates_glossary_aware
from rag_lib.evidence_handling import build_context, extract_evidence, answer_from_evidence
from rag_lib.extract_query import answer_sql, get_sql_evidence, is_evidence_usable, build_sql_response

print("Config OK")
print("VS endpoint:", VS_ENDPOINT)
print("VS index:", VS_INDEX_FULL_NAME)
print("Embedding endpoint:", EMBED_ENDPOINT)
print("LLM endpoint:", LLM_ENDPOINT)

# -------------------------
# Orchestrator (end-to-end RAG)
# -------------------------
def answer_with_rag(query: str) -> Dict[str, Any]:
    
    # =========================================================================
    # PASO 0: Intentar responder con SQL primero (early exit si es exitoso)
    # =========================================================================
    sql_evidence = get_sql_evidence(query)
    
    if is_evidence_usable(sql_evidence):
        # Early exit: tenemos respuesta válida desde la tabla SQL
        # print(build_sql_response(query, sql_evidence))
        return build_sql_response(query, sql_evidence)
    
    # Si SQL no fue exitoso, continuar con el flujo RAG normal
    # (sql_evidence.success=False, has_data=False, answer=None, o error)
    
    # =========================================================================
    # FLUJO RAG NORMAL (cuando SQL no tiene respuesta)
    # =========================================================================
    
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
    TOP_K_RERANK_INPUT = max(TOP_K_FINAL * 3, TOP_K_FINAL + 12) 
    hits_for_rerank = prepare_rerank_candidates_glossary_aware(query, candidates, max_input=TOP_K_RERANK_INPUT) 
    trace_stage(f"3) prepare_rerank_candidates_glossary_aware(max_input={TOP_K_RERANK_INPUT})", query, hits_for_rerank)

    # Tie-break SOLO para empates (por file_date) — al final del pre-rerank
    hits_for_rerank_tiebroken = tie_break_by_date_in_blocks(hits_for_rerank, block_size=2)
    trace_stage("4) tie_break_by_date_in_blocks(block_size=2)", query, hits_for_rerank_tiebroken)

    # Reranking en base a las reglas definidas:
    top_hits = rerank_with_llm(query, hits_for_rerank_tiebroken, top_k=TOP_K_FINAL) or candidates[:TOP_K_FINAL] 
    trace_stage(f"5) rerank_with_llm(top_k={TOP_K_FINAL})", query, top_hits)

    # Se construye la evidencia:
    evidence = extract_evidence(query, top_hits)
    
    # Se arma la respuesta final:
    answer = answer_from_evidence(query, top_hits, evidence)
    
    # Se construye el contexto:
    _, citations = build_context(top_hits, max_chars=MAX_CONTEXT_CHARS) 

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence,
        "citations": citations or [],
        "retrieved_candidates": candidates,
        "reranked_hits": top_hits,
        "response_source": "vector_rag",  # Indicador de origen
    }