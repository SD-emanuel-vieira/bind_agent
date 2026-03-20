import re
import json
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set, Tuple
import os
from databricks.vector_search.client import VectorSearchClient
from concurrent.futures import ThreadPoolExecutor, as_completed

from bind_rag_agent.config import (
    VS_ENDPOINT, 
    VS_INDEX_FULL_NAME, 
    VS_COLUMNS, 
    TOP_K_CANDIDATES, 
    LEX_FALLBACK_LIMIT,
    CHUNK_TYPE_QUERY_GATES,
    SEGMENT_ALIASES,
    CANONICAL_SEGMENTS,
    HYDRATION_MAX_IDS,
    VS_FULL_TEXT_MAX_RESULTS,
    VS_REQUEST_TIMEOUT_SECS,
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
        max_ids = HYDRATION_MAX_IDS
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


def _get_index(force_reinit: bool = False):
    """
    Obtiene el índice de Vector Search, inicializándolo si es necesario.
    
    Thread-safe: usa double-checked locking para que solo un thread
    inicialice el cliente, incluso en flujos multi-segmento paralelos.
    
    Recuperación: si force_reinit=True, descarta la conexión existente
    y crea una nueva. Útil cuando el token del SP expira o hay un
    error de conectividad transitorio.
    
    Autenticación:
      - Si hay DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET + DATABRICKS_HOST
        (típico en serving endpoints con Service Principal), se pasan explícitamente
        al VectorSearchClient.
      - Si no, se usa auto-detect (típico en notebooks).
    """
    global _vsc, _index
    
    if _index is not None and not force_reinit:
        return _index
    
    with _lock:
        # Double-check: otro thread pudo inicializar mientras esperábamos el lock
        if _index is not None and not force_reinit:
            return _index
        
        sp_client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        sp_client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        workspace_url = os.environ.get("DATABRICKS_HOST")

        if sp_client_id and sp_client_secret and workspace_url:
            _vsc = VectorSearchClient(
                workspace_url=workspace_url,
                service_principal_client_id=sp_client_id,
                service_principal_client_secret=sp_client_secret,
            )
            logger.info("[retriever] VectorSearchClient autenticado con Service Principal")
        else:
            _vsc = VectorSearchClient()
            logger.info("[retriever] VectorSearchClient autenticado con contexto de notebook")

        _index = _vsc.get_index(VS_ENDPOINT, VS_INDEX_FULL_NAME)
        logger.info(f"[retriever] Index inicializado: {VS_INDEX_FULL_NAME}")
    
    return _index


# Palabras clave en errores que indican que reconectar puede resolver el problema
_RETRIABLE_ERROR_KEYWORDS = {"auth", "token", "expired", "unauthenticated", "permission denied", "unauthorized"}


def _similarity_search_with_retry(**kwargs) -> Any:
    """
    Wrapper sobre index.similarity_search con retry automático.
    
    Si el primer intento falla con un error de autenticación o token
    expirado, fuerza reconexión del VectorSearchClient y reintenta
    una vez. Esto cubre el caso donde el serving endpoint lleva horas
    corriendo y el token del SP expiró.
    
    Acepta los mismos kwargs que index.similarity_search().
    """
    index = _get_index()
    try:
        return index.similarity_search(**kwargs)
    except Exception as e:
        err_lower = str(e).lower()
        if any(kw in err_lower for kw in _RETRIABLE_ERROR_KEYWORDS):
            logger.warning(f"[retriever] Auth error detectado, reconectando: {repr(e)}")
            index = _get_index(force_reinit=True)
            return index.similarity_search(**kwargs)
        raise


# ============================================================
# LEXICAL FALLBACK (Hybrid Search via SDK)
# ============================================================
def lexical_fallback(
    query: str, 
    hits: List[Dict[str, Any]], 
    limit: int = LEX_FALLBACK_LIMIT,
    filters: Optional[Dict[str, str]] = None,
    query_vector: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Best-effort: agrega matches léxicos desde Vector Search index (hybrid).
    Merge + dedupe con hits existentes.
    
    Usa query_type="hybrid" (GA) que combina keyword + semántico.
    Si el index tiene embeddings self-managed (sin modelo asociado),
    query_vector es necesario para que hybrid funcione.
    """
    q = (query or "").strip()
    if not q:
        return hits

    try:
        kwargs = {
            "query_text": q,
            "columns": VS_COLUMNS,
            "num_results": min(int(limit), VS_FULL_TEXT_MAX_RESULTS),
            "query_type": "hybrid",
        }
        if filters is not None:
            kwargs["filters"] = filters
        if query_vector is not None:
            kwargs["query_vector"] = query_vector

        res = _similarity_search_with_retry(**kwargs)
        rows = parse_vs_similarity_response(res)
    except Exception as e:
        logger.warning(f"[retriever] Lexical fallback failed: {repr(e)}")
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
# DESCOMPOSICIÓN DE QUERY MULTI-SEGMENTO
# ============================================================

# Conectores que quedan huérfanos al remover nombres de segmentos
_SEGMENT_CONNECTORS = {"para", "de", "y", "e", "entre", "por"}


def _clean_base_query(text: str) -> str:
    """
    Limpia secuencias de conectores huérfanos que quedan después 
    de remover los nombres de segmentos de la query.
    
    Regla: un conector solo entre dos palabras de contenido es legítimo
    ("ingresos de octubre"). Dos o más conectores contiguos son residuo
    de la remoción ("ingresos de y en octubre" → "de y" son residuos).
    
    Los conectores al final de la string siempre se descartan.
    """
    words = text.split()
    cleaned: List[str] = []
    buffer: List[str] = []  # acumula conectores contiguos
    
    for w in words:
        if w in _SEGMENT_CONNECTORS:
            buffer.append(w)
        else:
            if len(buffer) == 1:
                # Un solo conector es legítimo: "ingresos de octubre"
                cleaned.append(buffer[0])
            # Si buffer tiene 2+ conectores, son residuos → descartar
            buffer = []
            cleaned.append(w)
    
    # No agregar buffer final (conectores al final = residuos)
    return " ".join(cleaned)


# Patrones que indican "todos los segmentos" sin nombrarlos explícitamente
_ALL_SEGMENTS_PATTERNS = [
    r"cada\s+segmento",              # "cada segmento"
    r"cada\s+banca",                 # "cada banca"
    r"todos?\s+los\s+segmentos?",    # "todos los segmentos"
    r"todas?\s+las\s+bancas?",       # "todas las bancas"
    r"por\s+segmento",              # "desglose por segmento"
    r"por\s+banca",                 # "resultado por banca"
    r"de\s+cada\s+segmento",        # "resultado de cada segmento"
    r"de\s+cada\s+banca",           # "resultado de cada banca"
    r"desglose\s+(?:por\s+)?(?:segmentos?|bancas?)",  # "desglose por segmentos"
    r"breakdown\s+(?:por\s+)?(?:segmentos?|bancas?)",  # "breakdown por segmento"
]


def _detect_all_segments_intent(query: str) -> bool:
    """
    Detecta si la query pide info de TODOS los segmentos sin nombrarlos
    explícitamente.
    
    Ejemplos:
        "resultado operativo de cada segmento para octubre 2025" → True
        "resultado operativo por banca octubre 2025" → True
        "resultado operativo de octubre 2025" → False
        "resultado operativo de corporate octubre 2025" → False
    """
    q_lower = query.lower().strip()
    for pattern in _ALL_SEGMENTS_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _remove_all_segments_phrase(text: str) -> str:
    """
    Remueve las frases de 'todos los segmentos' del texto base
    para construir sub-queries limpias.
    
    "resultado operativo de cada segmento para octubre 2025"
    → "resultado operativo para octubre 2025"
    """
    for pattern in _ALL_SEGMENTS_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def decompose_multi_segment_query(
    query: str,
) -> Optional[Dict[str, Any]]:
    """
    Detecta si la query pide info de múltiples segmentos y la descompone
    en sub-queries independientes.
    
    Detecta DOS tipos de queries multi-segmento:
    
    A) Segmentos explícitos: "resultado operativo para Empresas, corporate e institucional"
       → Descompone solo en los segmentos nombrados.
       
    B) Todos los segmentos implícitos: "resultado operativo de cada segmento"
       → Descompone en TODOS los segmentos canónicos.
    
    Returns None si la query es single-segment o general.
    
    Returns dict con:
        base_query:       parte sin segmentos ("resultado operativo de octubre 2025")
        segments:         lista de segmentos canónicos
        sub_queries:      lista de sub-queries, una por segmento
        all_segments:     True si se expandió a todos los segmentos (tipo B)
    """
    q_lower = query.lower().strip().strip("¿?").strip()
    
    # ================================================================
    # TIPO B: "cada segmento" / "por banca" → expandir a TODOS
    # ================================================================
    if _detect_all_segments_intent(query):
        segments = list(CANONICAL_SEGMENTS)  # todos
        
        base = _remove_all_segments_phrase(q_lower)
        base = re.sub(r"[,;]", " ", base)
        base = re.sub(r"\s+", " ", base).strip()
        base = _clean_base_query(base)
        
        sub_queries = [f"{base} {seg}" for seg in segments]
        
        logger.info(
            f"[retriever] ALL-SEGMENTS decomposition: "
            f"segments={segments}, base='{base}', "
            f"sub_queries={sub_queries}"
        )
        
        return {
            "base_query": base,
            "segments": segments,
            "sub_queries": sub_queries,
            "all_segments": True,
        }
    
    # ================================================================
    # TIPO A: Segmentos explícitos nombrados ("empresas, corporate e institucional")
    # ================================================================
    detected: List[Tuple[str, int]] = []
    for alias, canonical in SEGMENT_ALIASES.items():
        match = re.search(rf"\b{re.escape(alias)}\b", q_lower)
        if match and not any(seg == canonical for seg, _ in detected):
            detected.append((canonical, match.start()))
    
    # Solo descomponer si hay 2+ segmentos distintos
    if len(detected) < 2:
        return None
    
    detected.sort(key=lambda x: x[1])
    segments = [seg for seg, _ in detected]
    
    # Construir base_query removiendo los segmentos
    base = q_lower
    for alias in sorted(SEGMENT_ALIASES.keys(), key=len, reverse=True):
        base = re.sub(rf"\b{re.escape(alias)}\b", " ", base)
    
    base = re.sub(r"[,;]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    base = _clean_base_query(base)
    
    sub_queries = [f"{base} {seg}" for seg in segments]
    
    logger.info(
        f"[retriever] Multi-segment decomposition: "
        f"segments={segments}, base='{base}', "
        f"sub_queries={sub_queries}"
    )
    
    return {
        "base_query": base,
        "segments": segments,
        "sub_queries": sub_queries,
        "all_segments": False,
    }


# ============================================================
# MERGE BALANCEADO ROUND-ROBIN (para multi-segmento)
# ============================================================

def _balanced_merge(
    by_segment: Dict[str, List[Dict[str, Any]]],
    k_total: int,
) -> List[Dict[str, Any]]:
    """
    Merge round-robin que garantiza representación de cada segmento.
    
    Orden de prioridad por ronda:
    1. Segmentos reales (en orden de aparición en la query)
    2. _general (tablas consolidadas, etc.)
    
    Dedup por chunk_id: si un chunk ya se agregó por otro segmento,
    se salta y se toma el siguiente.
    """
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    
    # Segmentos reales primero, _general al final
    real = [s for s in by_segment if s != "_general"]
    order = real + (["_general"] if "_general" in by_segment else [])
    
    indices = {seg: 0 for seg in order}
    
    safety_limit = k_total * 3
    iterations = 0
    
    while len(merged) < k_total and iterations < safety_limit:
        added_this_round = False
        
        for seg in order:
            hits = by_segment.get(seg, [])
            idx = indices[seg]
            
            # Buscar el siguiente hit no-duplicado
            while idx < len(hits):
                hit = hits[idx]
                cid = hit.get("chunk_id")
                idx += 1
                
                if cid and cid in seen:
                    continue
                
                # Tag de origen para tracing/debug
                hit["_retrieval_segment"] = seg
                merged.append(hit)
                if cid:
                    seen.add(cid)
                added_this_round = True
                break
            
            indices[seg] = idx
            
            if len(merged) >= k_total:
                break
        
        if not added_this_round:
            break
        
        iterations += 1
    
    return merged


# ============================================================
# RETRIEVAL POR SEGMENTO CON PARALELIZACIÓN
# ============================================================

# Max workers para sub-queries paralelas (ajustar según entorno)
_MAX_PARALLEL_WORKERS = int(os.getenv("RAG_MAX_PARALLEL_WORKERS", "4"))


def _retrieve_per_segment(
    decomposition: Dict[str, Any],
    k_total: int = TOP_K_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Ejecuta retrieval independiente para cada sub-query y merge
    los resultados con cobertura balanceada por segmento.
    
    Las sub-queries se ejecutan EN PARALELO para minimizar latencia.
    
    Args:
        decomposition: Output de decompose_multi_segment_query
        k_total:       Total máximo de candidatos a retornar
    
    Returns:
        Lista de candidatos con cobertura balanceada de todos los segmentos.
    """
    segments = decomposition["segments"]
    sub_queries = decomposition["sub_queries"]
    n_segments = len(segments)
    
    # k por segmento: suficientes para llenar el total con margen de dedup
    k_per = max(k_total // n_segments + 5, 15)
    
    # Preparar todas las tareas: sub-queries + query base
    tasks: List[Tuple[str, str]] = []  # (segment_label, query_text)
    for segment, sub_q in zip(segments, sub_queries):
        tasks.append((segment, sub_q))
    tasks.append(("_general", decomposition["base_query"]))
    
    # Ejecutar en paralelo
    by_segment: Dict[str, List[Dict[str, Any]]] = {}
    
    with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_WORKERS, len(tasks))) as pool:
        future_to_segment = {
            pool.submit(_retrieve_single_query, task_query, k_per): task_seg
            for task_seg, task_query in tasks
        }
        
        for future in as_completed(future_to_segment):
            seg = future_to_segment[future]
            try:
                hits = future.result()
                by_segment[seg] = hits
                logger.info(f"[retriever] Sub-query '{seg}': {len(hits)} hits")
            except Exception as e:
                logger.warning(f"[retriever] Sub-query '{seg}' failed: {repr(e)}")
                by_segment[seg] = []
    
    # Merge round-robin balanceado
    merged = _balanced_merge(by_segment, k_total)
    
    logger.info(
        f"[retriever] Multi-segment merge: "
        f"{' + '.join(f'{s}={len(by_segment.get(s,[]))}' for s in segments + ['_general'])} "
        f"→ {len(merged)} total"
    )
    
    return merged


# ============================================================
# FUNCIÓN DE RETRIEVAL SINGLE-QUERY (lógica original)
# ============================================================

def _retrieve_single_query(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    """
    Recupera candidatos para UNA sola query usando retrieval en dos pasadas.
    
    Esta es la lógica original de retrieve_candidates, renombrada para
    poder reutilizarla como building block del retrieval multi-segmento.
    
    Pasada 1 (CON filtro temporal): Si la query menciona una fecha,
    filtra vector search y lexical fallback por file_date >= mes preguntado.
    
    Pasada 2 (SIN filtro): Retrieval normal sin restricción de fecha.
    
    Pasada 3 (ANCHORS): Búsqueda léxica focalizada por términos clave.
    
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
    
    # --- Fallback: si la expansión falló, embedear la query cruda ---
    if qvec is None:
        try:
            qvec = embed_query(query_normalized)
            logger.info("[retriever] Embedding generado desde query cruda (fallback)")
        except Exception as e:
            logger.warning(f"[retriever] Embedding fallback también falló: {repr(e)}")
    
    # ================================================================
    # PASADA 1: CON filtro temporal (solo si hay fecha detectada)
    # ================================================================
    hits_filtered: List[Dict[str, Any]] = []
    
    if date_filter and qvec:
        try:
            res = _similarity_search_with_retry(
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
            q_fulltext, hits_filtered, limit=LEX_FALLBACK_LIMIT, 
            filters=date_filter, query_vector=qvec
        )

    # ================================================================
    # PASADA 2: SIN filtro (siempre se ejecuta)
    # ================================================================
    hits_unfiltered: List[Dict[str, Any]] = []
    
    if qvec:
        try:
            res = _similarity_search_with_retry(
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
    hits_unfiltered = lexical_fallback(
        q_fulltext, hits_unfiltered, limit=LEX_FALLBACK_LIMIT,
        query_vector=qvec
    )

    # ================================================================
    # PASADA 3: Búsqueda léxica focalizada por ANCHORS
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
                    q_anchors, hits_anchor, limit=LEX_FALLBACK_LIMIT, 
                    filters=date_filter, query_vector=qvec
                )
            
            # Sin filtro (complementario)
            hits_anchor = lexical_fallback(
                q_anchors, hits_anchor, limit=LEX_FALLBACK_LIMIT,
                query_vector=qvec
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


# ============================================================
# FUNCIÓN PRINCIPAL: retrieve_candidates (PUNTO DE ENTRADA)
# ============================================================

def retrieve_candidates(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    """
    Recupera candidatos usando retrieval inteligente.
    
    NUEVO: Detecta automáticamente queries que piden información sobre
    múltiples segmentos (ej: "empresas, corporate e institucional") y
    ejecuta retrieval independiente por cada segmento con merge balanceado.
    
    Para queries de un solo segmento o generales, usa el flujo existente
    sin ningún cambio.
    
    Flujo:
        1. Detectar si es multi-segmento
           ├── SI:  descomponer → retrieval paralelo por sub-query → merge round-robin
           └── NO:  retrieval normal (pasada 1 + pasada 2 + anchors)
        
        2. El resultado se pasa al pipeline downstream (rerank, evidence, answer)
           exactamente igual que antes.
    """
    # ================================================================
    # Intentar descomposición multi-segmento
    # ================================================================
    decomposition = decompose_multi_segment_query(query)
    
    if decomposition:
        logger.info(
            f"[retriever] MULTI-SEGMENT query detected: "
            f"segments={decomposition['segments']}, "
            f"sub_queries={decomposition['sub_queries']}"
        )
        return _retrieve_per_segment(
            decomposition=decomposition,
            k_total=k,
        )
    
    # ================================================================
    # Query normal (single segment o general) → flujo existente
    # ================================================================
    return _retrieve_single_query(query, k)