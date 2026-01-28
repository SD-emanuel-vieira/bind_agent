import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
import mlflow.deployments
from mlflow.utils.databricks_utils import get_databricks_host_creds
from databricks.vector_search.client import VectorSearchClient
from rag_lib.secret_functions import *
from rag_lib.config import *
from rag_lib.text_utils import *
from rag_lib.retriever import _get_dbx_auth, _vs_full_text_query, lexical_fallback
from rag_lib.business_glossary import BUSINESS_GLOSSARY_V2
from rag_lib.glossary_helper import glossary_snippet, glossary_expand_terms, prepare_rerank_candidates_glossary_aware
from rag_lib.llm import call_chat, expand_query_for_retrieval
from rag_lib.embeddings import embed_query
from rag_lib.rerank import query_anchors,hit_anchor_score,tie_break_by_date_in_blocks,enforce_anchor_priority,rerank_with_llm,trace_stage
from rag_lib.evidence_handling import build_context, extract_evidence, answer_from_evidence

# Clients
vsc = VectorSearchClient()
index = vsc.get_index(VS_ENDPOINT, VS_INDEX_FULL_NAME)
client = mlflow.deployments.get_deploy_client("databricks")

print("Config OK")
print("VS endpoint:", VS_ENDPOINT)
print("VS index:", VS_INDEX_FULL_NAME)
print("Embedding endpoint:", EMBED_ENDPOINT)
print("LLM endpoint:", LLM_ENDPOINT)

# -------------------------
# CELL 5: RETRIEVER (Vector Search + expansion + lexical fallback)
# - Guarantees lexical fallback is merged even if vector search fails
# -------------------------
def retrieve_candidates(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    # ✅ default (por si falla el try)
    q_fulltext = query

    # 1) Try vector search (best effort)
    try:
        gl = glossary_expand_terms(query)
        terms = gl.get("terms", []) or []
        acronyms = gl.get("acronyms", []) or []

        q_llm = expand_query_for_retrieval(query) or query

        # Query para embeddings/vector: más rica (mejor semántica)
        q_embed = q_llm
        if terms:
            q_embed = q_embed + "\n\nGLOSSARY TERMS: " + " | ".join(terms)

        # Helper de quoting para FULL_TEXT
        def _qt(t: str) -> str:
            t = (t or "").strip()
            return f'"{t}"' if " " in t else t

        # ✅ FULL_TEXT "glossary-focused" si hay acrónimos relevantes (evita dilución)
        # (Ignoramos acrónimos de 1 letra tipo R/E)
        focus_acronyms = [a.strip() for a in acronyms if isinstance(a, str) and len(a.strip()) >= 2]

        if focus_acronyms:
            # FULL_TEXT más "afilado": buscar directo por ROA/ROE/etc
            q_fulltext = " ".join(_qt(a) for a in focus_acronyms)
        else:
            # FULL_TEXT estándar: query + términos (como estaba antes)
            q_fulltext = query
            if terms:
                q_fulltext = q_fulltext + " " + " ".join(_qt(t) for t in terms)

        qvec = embed_query(q_embed)

        if qvec:
            res = index.similarity_search(
                query_vector=qvec,  # direct access index requires query_vector
                columns=VS_COLUMNS,
                num_results=k
            )
            hits = parse_vs_similarity_response(res)

            for h in hits:
                raw = (h.get("chunk_text") or "").strip()
                h["chunk_text_clean"] = strip_chunk_prefix(raw)

    except Exception as e:
        print("Vector retrieval failed, fallback lexical only. Error:", repr(e))
        hits = []
        q_fulltext = query  # ✅ aseguramos valor válido

    # 2) Always add lexical fallback (FULL_TEXT)
    hits = lexical_fallback(q_fulltext, hits, limit=LEX_FALLBACK_LIMIT)

    # 2.1) Exclude evidence that would be chart analysis
    hits = filter_hits_by_query_gates(query, hits, CHUNK_TYPE_QUERY_GATES)

    # 3) Merge + dedupe by chunk_id
    merged = []
    seen = set()
    for h in hits:
        cid = h.get("chunk_id")
        if cid and cid not in seen:
            merged.append(h)
            seen.add(cid)

    return merged

# -------------------------
# CELL 10: Orchestrator (end-to-end RAG)
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

    # Reranking en base a las reglas definidas:
    top_hits = rerank_with_llm(query, hits_for_rerank_tiebroken, top_k=TOP_K_FINAL) or candidates[:TOP_K_FINAL] 
    trace_stage(f"5) rerank_with_llm(top_k={TOP_K_FINAL})", query, top_hits)

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