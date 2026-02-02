import mlflow.deployments

from rag_lib.secret_functions import *
from rag_lib.config import *
from rag_lib.text_utils import *

from rag_lib.retriever import retrieve_candidates
from rag_lib.rerank import tie_break_by_date_in_blocks,enforce_anchor_priority,rerank_with_llm,trace_stage
from rag_lib.glossary_helper import prepare_rerank_candidates_glossary_aware
from rag_lib.evidence_handling import build_context, extract_evidence, answer_from_evidence

print("Config OK")
print("VS endpoint:", VS_ENDPOINT)
print("VS index:", VS_INDEX_FULL_NAME)
print("Embedding endpoint:", EMBED_ENDPOINT)
print("LLM endpoint:", LLM_ENDPOINT)

# -------------------------
# Orchestrator (end-to-end RAG)
# -------------------------
def answer_with_rag(query: str) -> Dict[str, Any]:
    # Posibles candidatos para la respuesta, se filtran por lexical_fallback (importtante) y chunk_type:
    candidates = retrieve_candidates(query, k=TOP_K_CANDIDATES) or [] 
    trace_stage("1) retrieve_candidates", query, candidates)

    #Si no se menciona ningun segmento entonces descarta toda evidencia relacionada a cualquier segmentpo:
    candidates = drop_segment_topics_if_query_general(query, candidates)
    trace_stage("1.1) drop_segment_topics_if_query_general", query, candidates)

    # Soft ordering #1: anchors (gating suave por intención)
    # candidates_anchor_sorted = prefer_anchor_hits(query, hits_for_rerank)
    candidates_anchor_sorted = enforce_anchor_priority(query, candidates)
    trace_stage("2) enforce_anchor_priority", query, candidates_anchor_sorted)

    # Soft ordering #2: glossary-aware (estricto por frases)
    # Preprocesamiento de candidatos glossary-aware:
    TOP_K_RERANK_INPUT = max(TOP_K_FINAL * 3, TOP_K_FINAL + 12) 
    hits_for_rerank = prepare_rerank_candidates_glossary_aware(query, candidates, max_input=TOP_K_RERANK_INPUT) 
    trace_stage(f"3) prepare_rerank_candidates_glossary_aware(max_input={TOP_K_RERANK_INPUT})", query, hits_for_rerank)

    # Tie-break SOLO para empates (por file_date) — al final del pre-rerank
    hits_for_rerank_tiebroken = tie_break_by_date_in_blocks(hits_for_rerank, block_size=2)
    trace_stage("4) tie_break_by_date_in_blocks(block_size=2)", query, hits_for_rerank_tiebroken)

    # # Reranking en base a las reglas definidas:
    # top_hits = hits_for_rerank_tiebroken
    top_hits = rerank_with_llm(query, hits_for_rerank_tiebroken, top_k=TOP_K_FINAL) or candidates[:TOP_K_FINAL] 
    trace_stage(f"5) rerank_with_llm(top_k={TOP_K_FINAL})", query, top_hits)

    # ##-----------------DEBUGGING
    # print(f"Number of hits: {len(top_hits)}")
    # print(f"First hit keys: {top_hits[0].keys()}")
    # print(f"First hit chunk_text_clean length: {len(top_hits[0].get('chunk_text_clean', ''))}")

    # # Ver el primer hit (el que tiene Banco Santander)
    # hit = top_hits[0]
    # print(f"chunk_type: {hit.get('chunk_type')}")
    # print(f"topic_heuristic: {hit.get('topic_heuristic')}")
    # print(f"topic_llm: {hit.get('topic_llm')}")
    # print(f"topic: {hit.get('topic')}")

    # # Después de tener top_hits
    # context, cites = build_context(top_hits, max_chars=12000)

    # print("\n" + "="*60)
    # print("CONTEXT SENT TO LLM (primeros 3000 chars):")
    # print("="*60)
    # print(context[:3000])
    # print("="*60 + "\n")

    # evidence = extract_evidence(query, top_hits)

    # # Imprimir el resultado completo
    # import json
    # print("\n" + "="*60)
    # print("EXTRACT_EVIDENCE RESULT:")
    # print("="*60)
    # print(json.dumps(evidence, indent=2, ensure_ascii=False))
    # print("="*60 + "\n")


    ##-----------------------------

    # Se contruye la evidencia:
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
    }