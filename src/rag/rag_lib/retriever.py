import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from mlflow.utils.databricks_utils import get_databricks_host_creds
from rag_lib.config import *
from rag_lib.text_utils import *

# -------------------------
# CELL 3: Lexical fallback via Vector Search FULL_TEXT (serving-friendly)
# - No Spark required (works in Model Serving)
# - Queries the same Vector Search index using query_type=FULL_TEXT
# - Merges + dedupes with existing hits
# -------------------------
def _get_dbx_auth() -> Tuple[str, str]:
    """Get Databricks host + token.

    Priority:
      1) Env vars: DATABRICKS_HOST/WORKSPACE_URL + DATABRICKS_TOKEN/TOKEN
      2) MLflow Databricks creds (works in Databricks notebooks)
      3) Notebook context token (best-effort)
    """
    host = (os.getenv("DATABRICKS_HOST") or os.getenv("WORKSPACE_URL") or "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN") or os.getenv("TOKEN") or ""

    # 2) MLflow helper (usually works in Databricks notebooks)
    if not host or not token:
        try:
            # from mlflow.utils.databricks_utils import get_databricks_host_creds
            creds = get_databricks_host_creds()
            host = host or (getattr(creds, "host", "") or "").rstrip("/")
            token = token or (getattr(creds, "token", "") or "")
        except Exception:
            pass

    # 3) Notebook context fallback (best-effort)
    if not host or not token:
        try:
            # Only import pyspark in notebook clusters; in serving this may not exist
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            if not host:
                host = ("https://" + spark.conf.get("spark.databricks.workspaceUrl")).rstrip("/")
            if not token:
                # dbutils available in notebooks
                token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        except Exception:
            pass

    if not host or not token:
        raise RuntimeError(
            "Missing Databricks auth. Set DATABRICKS_HOST + DATABRICKS_TOKEN (or run inside a Databricks notebook)."
        )

    return host, token

def _vs_full_text_query(
    index_name: str,
    query_text: str,
    columns: List[str],
    num_results: int,
    filters: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Call Vector Search REST API for FULL_TEXT queries and return rows as list[dict]."""
    host, token = _get_dbx_auth()
    url = f"{host}/api/2.0/vector-search/indexes/{index_name}/query"

    payload: Dict[str, Any] = {
        "query_text": query_text,
        "query_type": "FULL_TEXT",
        "columns": columns,
        "num_results": min(int(num_results), 200),
    }
    if filters is not None:
        payload["filters"] = filters

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result") or data
    colnames = result.get("column_names") or columns
    rows = result.get("data_array") or result.get("data") or []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = {colnames[i]: row[i] for i in range(min(len(colnames), len(row)))}
        out.append(d)
    return out

def lexical_fallback(query: str, hits: List[Dict[str, Any]], limit: int = LEX_FALLBACK_LIMIT) -> List[Dict[str, Any]]:
    """Best-effort: agrega matches lexicográficos desde Vector Search index (FULL_TEXT)."""
    q = (query or "").strip()
    if not q:
        return hits

    try:
        rows = _vs_full_text_query(
            index_name=VS_INDEX_FULL_NAME,
            query_text=q,
            columns=VS_COLUMNS,
            num_results=limit,
            filters=None,
        )
        print(f"Full-text search retornó: {len(rows)} resultados")
    
        print(f"Total resultados: {len(rows)}\n")

        # for i, r in enumerate(rows[:LEX_FALLBACK_LIMIT], 1):
        #     topic = r.get("topic_heuristic", "")
        #     chunk = r.get("chunk_text", "")[:200]
            
        #     has_gasto_in_topic = "gasto" in topic.lower()
        #     has_gasto_in_chunk = "gasto" in chunk.lower()
            
        #     print(f"--- Resultado {i} ---")
        #     print(f"  topic_heuristic: {topic[:80]}")
        #     print(f"  'gasto' en topic: {has_gasto_in_topic}")
        #     print(f"  'gasto' en chunk_text: {has_gasto_in_chunk}")
        #     print(f"  chunk_text (preview): {chunk[:100]}...")
        #     print()

    except Exception as e:
        # If serving env doesn't have auth configured, do not break main flow.
        print("Lexical FULL_TEXT fallback failed. Error:", repr(e))
        return hits

    lex_hits: List[Dict[str, Any]] = []
    for d in rows:
        raw = (d.get("chunk_text") or "").strip()
        d["chunk_text_clean"] = strip_chunk_prefix(raw)
        lex_hits.append(d)

    # merge + dedupe por chunk_id (primero mantiene orden de hits existentes)
    merged: List[Dict[str, Any]] = []
    seen = set()
    for h in hits + lex_hits:
        cid = h.get("chunk_id")
        key = cid if cid is not None else (h.get("path"), h.get("page_id"), h.get("page_num"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)

    # print(f"lexical_fallback: {len(merged)} hits")

    return merged

# # -------------------------
# # CELL 5: RETRIEVER (Vector Search + expansion + lexical fallback)
# # - Guarantees lexical fallback is merged even if vector search fails
# # -------------------------
# def retrieve_candidates(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
#     hits: List[Dict[str, Any]] = []

#     # ✅ default (por si falla el try)
#     q_fulltext = query

#     # 1) Try vector search (best effort)
#     try:
#         gl = glossary_expand_terms(query)
#         terms = gl.get("terms", []) or []
#         acronyms = gl.get("acronyms", []) or []

#         q_llm = expand_query_for_retrieval(query) or query

#         # Query para embeddings/vector: más rica (mejor semántica)
#         q_embed = q_llm
#         if terms:
#             q_embed = q_embed + "\n\nGLOSSARY TERMS: " + " | ".join(terms)

#         # Helper de quoting para FULL_TEXT
#         def _qt(t: str) -> str:
#             t = (t or "").strip()
#             return f'"{t}"' if " " in t else t

#         # ✅ FULL_TEXT "glossary-focused" si hay acrónimos relevantes (evita dilución)
#         # (Ignoramos acrónimos de 1 letra tipo R/E)
#         focus_acronyms = [a.strip() for a in acronyms if isinstance(a, str) and len(a.strip()) >= 2]

#         if focus_acronyms:
#             # FULL_TEXT más "afilado": buscar directo por ROA/ROE/etc
#             q_fulltext = " ".join(_qt(a) for a in focus_acronyms)
#         else:
#             # FULL_TEXT estándar: query + términos (como estaba antes)
#             q_fulltext = query
#             if terms:
#                 q_fulltext = q_fulltext + " " + " ".join(_qt(t) for t in terms)

#         qvec = embed_query(q_embed)

#         if qvec:
#             res = index.similarity_search(
#                 query_vector=qvec,  # direct access index requires query_vector
#                 columns=VS_COLUMNS,
#                 num_results=k
#             )
#             hits = parse_vs_similarity_response(res)

#             for h in hits:
#                 raw = (h.get("chunk_text") or "").strip()
#                 h["chunk_text_clean"] = strip_chunk_prefix(raw)

#     except Exception as e:
#         print("Vector retrieval failed, fallback lexical only. Error:", repr(e))
#         hits = []
#         q_fulltext = query  # ✅ aseguramos valor válido

#     # # 2) Always add lexical fallback (FULL_TEXT)
#     # hits = lexical_fallback(q_fulltext, hits, limit=LEX_FALLBACK_LIMIT)

#     # 2.1) Exclude evidence that would be chart analysis
#     hits = filter_hits_by_query_gates(query, hits, CHUNK_TYPE_QUERY_GATES)

#     # 3) Merge + dedupe by chunk_id
#     merged = []
#     seen = set()
#     for h in hits:
#         cid = h.get("chunk_id")
#         if cid and cid not in seen:
#             merged.append(h)
#             seen.add(cid)

#     return merged