import re
import json
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from mlflow.utils.databricks_utils import get_databricks_host_creds
from databricks.vector_search.client import VectorSearchClient

from bind_rag_agent.config import (
    VS_ENDPOINT, 
    VS_INDEX_FULL_NAME, 
    VS_COLUMNS, 
    TOP_K_CANDIDATES, 
    LEX_FALLBACK_LIMIT,
    CHUNK_TYPE_QUERY_GATES
)
from bind_rag_agent.text_utils import (
    parse_vs_similarity_response, 
    strip_chunk_prefix, 
    filter_hits_by_query_gates
)
from bind_rag_agent.vector_search.glossary_helper import glossary_expand_terms, extract_query_anchors
from bind_rag_agent.vector_search.llm import expand_query_for_retrieval
from bind_rag_agent.vector_search.embeddings import embed_query

import logging

logger = logging.getLogger(__name__)
METADATA_HYDRATION_TABLE = (os.getenv("RAG_METADATA_HYDRATION_TABLE") or "").strip()


# ============================================================
# NORMALIZACIÓN DE QUERIES AMBIGUAS
# ============================================================

_QUERY_NORMALIZATION_RULES = [
    {
        "pattern": r"\bresultado\s+neto\b",
        "exclude_patterns": [
            r"\bresultado\s+neto\s+contable\b",
            r"\bresultado\s+neto\s+comercial\b",
            r"\bresultado\s+neto\s+ajustado\b",
            r"\bresultado\s+neto\s+operativo\b",
        ],
        "replacement": "resultado de gestión neto",
        "note": "En BIND, 'resultado neto' a secas = Resultado Gestión Neto AxI",
    },
]


def normalize_query_for_retrieval(query: str) -> str:
    q_normalized = query
    q_lower = query.lower()
    
    for rule in _QUERY_NORMALIZATION_RULES:
        if not re.search(rule["pattern"], q_lower):
            continue
        excluded = any(
            re.search(exc, q_lower) 
            for exc in rule.get("exclude_patterns", [])
        )
        if excluded:
            continue
        q_normalized = re.sub(
            rule["pattern"], 
            rule["replacement"], 
            q_normalized, 
            flags=re.IGNORECASE
        )
        logger.info(f"[retriever] Query normalizada: '{query}' → '{q_normalized}' ({rule['note']})")
    
    return q_normalized


# ============================================================
# EXTRACCIÓN DE FECHA DE LA QUERY
# ============================================================

_MONTHS_MAP = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def extract_date_from_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Extrae mes y año de la query del usuario.
    
    Returns:
        {"month": 10, "year": 2025, "date_from": "2025-10-01"} o None
    
    Lógica: si el usuario pregunta por "octubre 2025", el archivo que contiene
    esos datos fue creado en octubre o noviembre 2025. Filtramos file_date
    desde el primer día de ese mes en adelante.
    """
    q_lower = (query or "").lower()
    
    # Patrón 1: nombre de mes + año ("octubre 2025", "octubre de 2025")
    pattern = r"\b(" + "|".join(_MONTHS_MAP.keys()) + r")\s+(?:de\s+)?(\d{4})\b"
    match = re.search(pattern, q_lower)
    if match:
        month_name, year_str = match.group(1), match.group(2)
        month = _MONTHS_MAP[month_name]
        year = int(year_str)
        date_from = f"{year}-{month:02d}-01"
        logger.info(f"[retriever] Fecha detectada: {month_name} {year} → file_date >= {date_from}")
        return {"month": month, "year": year, "date_from": date_from}
    
    # Patrón 2: MM/YYYY o MM-YYYY
    match = re.search(r"\b(\d{1,2})[/\-](\d{4})\b", q_lower)
    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        if 1 <= month <= 12 and 2020 <= year <= 2030:
            date_from = f"{year}-{month:02d}-01"
            logger.info(f"[retriever] Fecha detectada: {month}/{year} → file_date >= {date_from}")
            return {"month": month, "year": year, "date_from": date_from}
    
    # Patrón 3: solo nombre de mes (sin año) → asumir año actual
    for month_name, month_num in _MONTHS_MAP.items():
        if re.search(rf"\b{month_name}\b", q_lower):
            year = datetime.now().year
            date_from = f"{year}-{month_num:02d}-01"
            logger.info(f"[retriever] Fecha detectada (sin año): {month_name} → asumiendo {year}, file_date >= {date_from}")
            return {"month": month_num, "year": year, "date_from": date_from}
    
    return None


def build_date_filter(date_info: Dict[str, Any]) -> Dict[str, str]:
    """
    Construye filtro de fecha para Databricks Vector Search.
    file_date >= primer día del mes preguntado, sin límite superior.
    """
    return {"file_date >=": date_info["date_from"]}


# ============================================================
# INICIALIZACIÓN DEL CLIENTE DE VECTOR SEARCH
# ============================================================

import threading

_vsc: Optional[VectorSearchClient] = None
_index = None
_lock = threading.Lock()


def _has_non_empty_metadata_enrich(hit: Dict[str, Any]) -> bool:
    """True si metadata_enrich parece venir poblado en el hit."""
    raw = hit.get("metadata_enrich")
    if raw is None:
        return False
    if isinstance(raw, str):
        s = raw.strip().lower()
        return s not in {"", "null", "none", "{}", "[]"}
    if isinstance(raw, (list, tuple, set, dict)):
        return len(raw) > 0
    if isinstance(raw, (bytes, bytearray)):
        return len(raw) > 0
    return True


def _merge_hit_fields(preferred: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona dos hits manteniendo prioridad de orden en `preferred`,
    pero completando campos faltantes con `candidate`.
    """
    out = dict(preferred)
    
    # Si candidate trae metadata_enrich útil y preferred no, preservar candidate
    if not _has_non_empty_metadata_enrich(out) and _has_non_empty_metadata_enrich(candidate):
        out["metadata_enrich"] = candidate.get("metadata_enrich")
    
    # Completar campos de contexto si faltan
    fill_keys = [
        "chunk_text_clean",
        "chunk_text",
        "topic_heuristic",
        "topic_llm",
        "topic_content",
        "context_text",
        "page_segment",
        "path",
        "file_date",
        "chunk_type",
    ]
    for key in fill_keys:
        if not out.get(key) and candidate.get(key):
            out[key] = candidate.get(key)
    
    return out


def _hydrate_metadata_enrich_from_table(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback opcional: completa metadata_enrich desde una tabla Delta
    usando chunk_id cuando Vector Search devuelve metadata_enrich vacío.
    
    Activación:
      export RAG_METADATA_HYDRATION_TABLE="<catalog.schema.table>"
    """
    if not hits or not METADATA_HYDRATION_TABLE:
        return hits
    
    missing_ids: List[str] = []
    for h in hits:
        cid = (h.get("chunk_id") or "").strip()
        if not cid:
            continue
        if _has_non_empty_metadata_enrich(h):
            continue
        missing_ids.append(cid)
    
    if not missing_ids:
        return hits
    
    # Deduplicar preservando orden
    seen_ids = set()
    missing_ids = [cid for cid in missing_ids if not (cid in seen_ids or seen_ids.add(cid))]
    
    try:
        from pyspark.sql import SparkSession
    except Exception:
        return hits
    
    try:
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:
        return hits
    
    try:
        # Limitar tamaño del IN para evitar queries enormes
        max_ids = 500
        ids_slice = missing_ids[:max_ids]
        quoted_ids = ",".join("'" + cid.replace("'", "''") + "'" for cid in ids_slice)
        
        query = (
            f"SELECT chunk_id, metadata_enrich "
            f"FROM {METADATA_HYDRATION_TABLE} "
            f"WHERE chunk_id IN ({quoted_ids}) "
            f"AND metadata_enrich IS NOT NULL "
            f"AND TRIM(CAST(metadata_enrich AS STRING)) NOT IN ('', 'null', 'None', '{{}}', '[]')"
        )
        
        rows = spark.sql(query).collect()
        by_cid: Dict[str, Any] = {}
        for row in rows:
            # Row puede exponer acceso por atributo o dict-like
            cid = str(getattr(row, "chunk_id", None) or row["chunk_id"])
            md = getattr(row, "metadata_enrich", None) if hasattr(row, "metadata_enrich") else row["metadata_enrich"]
            if cid and md is not None:
                by_cid[cid] = md
        
        if not by_cid:
            return hits
        
        hydrated = 0
        for h in hits:
            cid = (h.get("chunk_id") or "").strip()
            if not cid or _has_non_empty_metadata_enrich(h):
                continue
            md = by_cid.get(cid)
            if md is not None:
                h["metadata_enrich"] = md
                hydrated += 1
        
    except Exception:
        pass
    
    return hits


def _get_index():
    """
    Obtiene el índice de Vector Search, inicializándolo si es necesario.
    Usa patrón singleton para evitar múltiples conexiones.
    """
    global _vsc, _index
    
    if _index is None:
        _vsc = VectorSearchClient()
        _index = _vsc.get_index(VS_ENDPOINT, VS_INDEX_FULL_NAME)
        logger.info(f"[retriever] Index inicializado: {VS_INDEX_FULL_NAME}")
    
    return _index


# ============================================================
# AUTENTICACIÓN DATABRICKS (para FULL_TEXT queries)
# ============================================================
def _get_dbx_auth() -> Tuple[str, str]:
    """Get Databricks host + token."""
    host = (os.getenv("DATABRICKS_HOST") or os.getenv("WORKSPACE_URL") or "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN") or os.getenv("TOKEN") or ""

    if not host or not token:
        try:
            creds = get_databricks_host_creds()
            host = host or (getattr(creds, "host", "") or "").rstrip("/")
            token = token or (getattr(creds, "token", "") or "")
        except Exception:
            pass

    if not host or not token:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            if not host:
                host = ("https://" + spark.conf.get("spark.databricks.workspaceUrl")).rstrip("/")
            if not token:
                token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        except Exception:
            pass

    if not host or not token:
        raise RuntimeError(
            "Missing Databricks auth. Set DATABRICKS_HOST + DATABRICKS_TOKEN (or run inside a Databricks notebook)."
        )

    return host, token


# ============================================================
# FULL_TEXT QUERY (Lexical Search)
# ============================================================
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


# ============================================================
# LEXICAL FALLBACK
# ============================================================
def lexical_fallback(
    query: str, 
    hits: List[Dict[str, Any]], 
    limit: int = LEX_FALLBACK_LIMIT,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Best-effort: agrega matches lexicográficos desde Vector Search index (FULL_TEXT).
    Merge + dedupe con hits existentes.
    """
    q = (query or "").strip()
    if not q:
        return hits

    try:
        rows = _vs_full_text_query(
            index_name=VS_INDEX_FULL_NAME,
            query_text=q,
            columns=VS_COLUMNS,
            num_results=limit,
            filters=filters,
        )
    except Exception as e:
        logger.warning(f"[retriever] Lexical FULL_TEXT fallback failed: {repr(e)}")
        return hits

    lex_hits: List[Dict[str, Any]] = []
    for d in rows:
        raw = (d.get("chunk_text") or "").strip()
        d["chunk_text_clean"] = strip_chunk_prefix(raw)
        lex_hits.append(d)
    
    merged: List[Dict[str, Any]] = []
    seen_index: Dict[Any, int] = {}
    for h in hits + lex_hits:
        cid = h.get("chunk_id")
        key = cid if cid is not None else (h.get("path"), h.get("page_id"), h.get("page_num"))
        idx = seen_index.get(key)
        if idx is None:
            seen_index[key] = len(merged)
            merged.append(h)
            continue
        merged[idx] = _merge_hit_fields(merged[idx], h)

    return merged


# ============================================================
# MERGE + DEDUP HELPER
# ============================================================
def _merge_and_dedup(
    primary: List[Dict[str, Any]], 
    secondary: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge dos listas de hits priorizando primary. Dedup por chunk_id.
    """
    merged: List[Dict[str, Any]] = []
    seen_index: Dict[str, int] = {}
    for h in primary + secondary:
        cid = h.get("chunk_id")
        if not cid:
            continue
        idx = seen_index.get(cid)
        if idx is None:
            seen_index[cid] = len(merged)
            merged.append(h)
            continue
        merged[idx] = _merge_hit_fields(merged[idx], h)
    return merged


# ============================================================
# FUNCIÓN PRINCIPAL: retrieve_candidates
# ============================================================
def retrieve_candidates(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    """
    Recupera candidatos usando retrieval en dos pasadas:
    
    Pasada 1 (CON filtro temporal): Si la query menciona una fecha,
    filtra vector search y lexical fallback por file_date >= mes preguntado.
    Esto prioriza documentos del período correcto.
    
    Pasada 2 (SIN filtro): Retrieval normal sin restricción de fecha.
    Captura evidencia que el filtro temporal pueda haber excluido.
    
    Los resultados se mergean priorizando la pasada filtrada.
    """
    # 0) Normalizar query ambigua
    query_normalized = normalize_query_for_retrieval(query)
    
    # 1) Detectar fecha en la query → construir filtro
    date_info = extract_date_from_query(query)
    date_filter = build_date_filter(date_info) if date_info else None
    
    if date_filter:
        logger.info(f"[retriever] Filtro temporal: {date_filter}")
    
    # 2) Preparar query expandida (se usa en ambas pasadas)
    q_fulltext = query_normalized
    qvec = None
    
    try:
        gl = glossary_expand_terms(query_normalized)
        terms = gl.get("terms", []) or []
        acronyms = gl.get("acronyms", []) or []

        q_llm = expand_query_for_retrieval(query_normalized) or query_normalized

        q_embed = q_llm
        if terms:
            q_embed = q_embed + "\n\nGLOSSARY TERMS: " + " | ".join(terms)

        def _qt(t: str) -> str:
            t = (t or "").strip()
            return f'"{t}"' if " " in t else t

        focus_acronyms = [a.strip() for a in acronyms if isinstance(a, str) and len(a.strip()) >= 2]

        if focus_acronyms:
            # IMPORTANTE: no perder los términos originales de la query.
            # Los acrónimos se AGREGAN, no reemplazan.
            q_fulltext = query_normalized + " " + " ".join(_qt(a) for a in focus_acronyms)
        else:
            q_fulltext = query_normalized
            if terms:
                q_fulltext = q_fulltext + " " + " ".join(_qt(t) for t in terms)

        qvec = embed_query(q_embed)

    except Exception as e:
        logger.warning(
            "Query expansion failed",
            extra={"error": repr(e), "query": query[:100]}
        )
    
    # ================================================================
    # PASADA 1: CON filtro temporal (solo si hay fecha detectada)
    # ================================================================
    hits_filtered: List[Dict[str, Any]] = []
    
    if date_filter and qvec:
        try:
            index = _get_index()
            res = index.similarity_search(
                query_vector=qvec,
                columns=VS_COLUMNS,
                num_results=k,
                filters=date_filter,
            )
            hits_filtered = parse_vs_similarity_response(res)

            for h in hits_filtered:
                raw = (h.get("chunk_text") or "").strip()
                h["chunk_text_clean"] = strip_chunk_prefix(raw)
            
            if hits_filtered:
                with_meta = sum(1 for h in hits_filtered if _has_non_empty_metadata_enrich(h))
                logger.info(
                    f"[retriever] Pasada 1 metadata_enrich poblado: {with_meta}/{len(hits_filtered)}"
                )
            
            logger.info(f"[retriever] Pasada 1 (filtrada): {len(hits_filtered)} hits")

        except Exception as e:
            logger.warning(f"[retriever] Pasada filtrada falló: {repr(e)}")
            hits_filtered = []
        
        # Lexical fallback CON filtro
        hits_filtered = lexical_fallback(
            q_fulltext, hits_filtered, limit=LEX_FALLBACK_LIMIT, filters=date_filter
        )

    # ================================================================
    # PASADA 2: SIN filtro (siempre se ejecuta)
    # ================================================================
    hits_unfiltered: List[Dict[str, Any]] = []
    
    if qvec:
        try:
            index = _get_index()
            res = index.similarity_search(
                query_vector=qvec,
                columns=VS_COLUMNS,
                num_results=k,
            )
            hits_unfiltered = parse_vs_similarity_response(res)

            for h in hits_unfiltered:
                raw = (h.get("chunk_text") or "").strip()
                h["chunk_text_clean"] = strip_chunk_prefix(raw)
            
            if hits_unfiltered:
                with_meta = sum(1 for h in hits_unfiltered if _has_non_empty_metadata_enrich(h))
                logger.info(
                    f"[retriever] Pasada 2 metadata_enrich poblado: {with_meta}/{len(hits_unfiltered)}"
                )

        except Exception as e:
            logger.warning(
                "Vector retrieval failed, using lexical fallback",
                extra={"error": repr(e), "query": query[:100]}
            )
            hits_unfiltered = []
    
    # Lexical fallback SIN filtro
    hits_unfiltered = lexical_fallback(q_fulltext, hits_unfiltered, limit=LEX_FALLBACK_LIMIT)

    # ================================================================
    # PASADA 3: Búsqueda léxica focalizada por ANCHORS
    # Red de seguridad: busca con solo los términos clave de la query,
    # sin ruido de stopwords ni expansiones. Esto captura chunks donde
    # el término literal (ej: "previsiones") aparece en el texto pero
    # no tiene suficiente similitud semántica para estar en el top-K.
    # ================================================================
    hits_anchor: List[Dict[str, Any]] = []
    
    try:
        anchors = extract_query_anchors(query_normalized)
        if anchors:
            q_anchors = " ".join(anchors)
            logger.info(f"[retriever] Anchor search: '{q_anchors}'")
            
            # Con filtro temporal si existe
            if date_filter:
                hits_anchor = lexical_fallback(
                    q_anchors, hits_anchor, limit=LEX_FALLBACK_LIMIT, filters=date_filter
                )
            
            # Sin filtro (complementario)
            hits_anchor = lexical_fallback(
                q_anchors, hits_anchor, limit=LEX_FALLBACK_LIMIT
            )
    except Exception as e:
        logger.warning(f"[retriever] Anchor search failed: {repr(e)}")

    # ================================================================
    # MERGE: filtrada (prioridad) + sin filtro + anchors (complemento)
    # ================================================================
    if hits_filtered:
        hits = _merge_and_dedup(hits_filtered, hits_unfiltered)
        hits = _merge_and_dedup(hits, hits_anchor)
        logger.info(
            f"[retriever] Merge: {len(hits_filtered)} filtrados + "
            f"{len(hits_unfiltered)} sin filtro + "
            f"{len(hits_anchor)} anchors → {len(hits)} total"
        )
    else:
        hits = _merge_and_dedup(hits_unfiltered, hits_anchor)
    
    # Fallback opcional: hidratar metadata_enrich desde tabla Delta por chunk_id
    hits = _hydrate_metadata_enrich_from_table(hits)

    # Filtrado por query gates
    hits = filter_hits_by_query_gates(query, hits, CHUNK_TYPE_QUERY_GATES)

    # Dedup final
    merged = []
    seen = set()
    for h in hits:
        cid = h.get("chunk_id")
        if cid and cid not in seen:
            merged.append(h)
            seen.add(cid)

    return merged