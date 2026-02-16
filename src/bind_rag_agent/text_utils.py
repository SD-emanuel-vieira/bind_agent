import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import unicodedata
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import defaultdict
from bind_rag_agent.config import SEGMENTS, SEGMENT_ALIASES
import logging

# Configurar logger
logger = logging.getLogger(__name__)

# ==============================================================================
# SECCIÓN 1: UTILIDADES DE TEXTO PURO
# ==============================================================================
def strip_chunk_prefix(text: str) -> str:
    """Remove [SOURCE]/[TOPIC] prefix (if your gold chunks included it)."""
    if not text:
        return ""
    return re.sub(r"(?s)^\[SOURCE:[^\]]*\]\s*\n\[TOPIC:[^\]]*\]\s*\n\s*", "", text).strip()

def shorten(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else (t[:n].rstrip() + "…")

def safe_json_load(s: str, log_errors: bool = True) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    
    i = s.find("{")
    j = s.rfind("}")
    
    if i >= 0 and j > i:
        s2 = s[i:j+1]
        try:
            return json.loads(s2)
        except json.JSONDecodeError as e:
            if log_errors:
                logger.warning(f"JSON parse failed: {e}. Input: {s2[:200]}...")
            return {}
    
    if log_errors:
        logger.warning(f"No JSON found in: {s[:100]}...")
    return {}

def extract_chat_content(resp: Any) -> str:
    """Best-effort extraction of chat content from serving response."""
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            msg = resp["choices"][0].get("message", {})
            return msg.get("content", "") or ""
        if "predictions" in resp and resp["predictions"]:
            p0 = resp["predictions"][0]
            if isinstance(p0, dict) and "content" in p0:
                return p0["content"]
            if isinstance(p0, str):
                return p0
    return str(resp)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def safe_get_str(d: dict, key: str, default: str = "") -> str:
    """
    Obtiene un valor de un diccionario como string normalizado.
    
    Maneja de forma segura:
    - Keys que no existen
    - Valores None
    - Valores de tipos no-string
    
    Args:
        d: Diccionario fuente
        key: Key a buscar
        default: Valor por defecto si no existe
        
    Returns:
        Valor como string, o default si no existe/es None
        
    Example:
        >>> safe_get_str({"name": "Test"}, "name")
        'Test'
        >>> safe_get_str({"count": 42}, "count")
        '42'
        >>> safe_get_str({}, "missing")
        ''
    """
    val = d.get(key)
    if val is None:
        return default
    return str(val)


# ==============================================================================
# SECCIÓN 2: PARSERS DE RESPUESTAS API
# ==============================================================================

def parse_vs_similarity_response(res: Any) -> List[Dict[str, Any]]:
    """Normalize Vector Search similarity_search response to list[dict]."""
    if isinstance(res, dict):
        # Intentar obtener columns de diferentes ubicaciones
        cols = None
        data = None
        
        # Formato nuevo: manifest.columns + result.data_array
        if "manifest" in res:
            manifest_cols = res.get("manifest", {}).get("columns", [])
            if manifest_cols:
                # Extraer nombres de columnas si vienen como [{'name': 'col1'}, ...]
                if isinstance(manifest_cols[0], dict):
                    cols = [c.get("name") for c in manifest_cols]
                else:
                    cols = manifest_cols
            data = res.get("result", {}).get("data_array", [])
        
        # Formato antiguo: result.columns + result.data_array
        if not cols:
            r = res.get("result") or res
            cols = r.get("columns")
            data = r.get("data_array") or r.get("data") or []
        
        if cols and data:
            out = []
            for row in data:
                n = min(len(cols), len(row))
                d = {}
                for i in range(n):
                    col_raw = cols[i]
                    d[col_raw] = row[i]
                    
                    # Alias canónico: strip + lowercase (si difiere del original)
                    if isinstance(col_raw, str):
                        col_norm = col_raw.strip()
                        if col_norm and col_norm != col_raw and col_norm not in d:
                            d[col_norm] = row[i]
                        col_lower = col_norm.lower()
                        if col_lower and col_lower not in d:
                            d[col_lower] = row[i]
                out.append(d)
            return out
    
    if isinstance(res, list):
        return res
    return []

# ==============================================================================
# SECCIÓN 3: LÓGICA DE FILTRADO DE SEGMENTOS
# ==============================================================================
def normalize_text(s: str) -> str:
    """
    Normalización canónica para matching de texto.
    Usado por: rerank, glossary_helper, business_rules, smart_routing
    """
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_for_search(s: Optional[str]) -> str:
    """
    Normalización para búsqueda full-text.
    
    Similar a normalize_text pero con strip final adicional
    para garantizar limpieza en búsquedas.
    
    Args:
        s: Texto a normalizar
        
    Returns:
        Texto normalizado y limpio para búsqueda
    """
    return normalize_text(s).strip()
    
# ============================================================
# ALIASES PARA COMPATIBILIDAD HACIA ATRÁS
# ============================================================
# Estos aliases permiten que el código existente siga funcionando
# sin necesidad de modificar los imports inmediatamente.

_norm = normalize_text
_norm_for_match = normalize_for_search

def filter_hits_by_query_gates(query: str, hits: list[dict], gates: dict[str, list[str]]) -> list[dict]:
    """
    Si hit.chunk_type está en `gates`, solo se mantiene si la query contiene alguna keyword permitida.
    """
    qn = _norm_q(query)
    if not qn or not hits:
        return hits

    out = []
    for h in hits:
        ct = (h.get("chunk_type") or "").strip().lower()
        if ct in gates:
            allowed = gates.get(ct) or []
            allowed_norm = [_norm_q(k) for k in allowed if k]
            if not any(k and k in qn for k in allowed_norm):
                continue
        out.append(h)

    # opcional: no vaciar todo si el filtro fue demasiado agresivo
    return out or hits

"""
Función mejorada: drop_segment_topics_if_query_general

ACTUALIZADO:
- Comportamiento para 3 casos:
  1. Query SIN segmento → descartar hits CON segmentos
  2. Query CON segmento específico → mantener SOLO hits con ese segmento
  3. Query COMPARATIVA entre segmentos → mantener hits de TODOS los segmentos (balanceado)
"""

# Patrones que indican query comparativa entre segmentos
COMPARATIVE_PATTERNS = [
    r"qu[eé]\s+segmento",           # "qué segmento ha generado más"
    r"cu[aá]l\s+segmento",          # "cuál segmento tiene mayor"
    r"qu[eé]\s+banca",              # "qué banca ha generado más"
    r"cu[aá]l\s+banca",             # "cuál banca tiene mayor"
    r"comparar?\s+(?:los\s+)?segmentos?",  # "comparar segmentos"
    r"comparar?\s+(?:las\s+)?bancas?",     # "comparar bancas"
    r"entre\s+(?:los\s+)?segmentos?",      # "entre segmentos"
    r"ranking\s+(?:de\s+)?segmentos?",     # "ranking de segmentos"
    r"ranking\s+(?:de\s+)?bancas?",        # "ranking de bancas"
    r"(?:m[aá]s|mayor|mejor)\s+.*\s+por\s+segmento",  # "más ingresos por segmento"
    r"(?:m[aá]s|mayor|mejor)\s+.*\s+por\s+banca",     # "mayor resultado por banca"
    r"todos\s+los\s+segmentos?",    # "todos los segmentos"
    r"todas\s+las\s+bancas?",       # "todas las bancas"
    r"cada\s+segmento",             # "cada segmento"
    r"cada\s+banca",                # "cada banca"
    r"por\s+segmento",              # "desglose por segmento"
    r"por\s+banca",                 # "desglose por banca"
    r"desglose",                    # "desglose"
    r"breakdown",                   # "breakdown"
]


def _norm_q(s: str) -> str:
    """Normaliza texto para matching."""
    s = (s or "").lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_comparative_query(query: str) -> bool:
    """
    Detecta si la query es comparativa entre segmentos.
    
    Ejemplos de queries comparativas:
    - "¿Qué segmento ha generado más ingresos?"
    - "Comparar resultados por banca"
    - "Ranking de segmentos por MF préstamos"
    - "¿Cuál banca tiene mayor resultado operativo?"
    """
    qn = _norm_q(query)
    
    for pattern in COMPARATIVE_PATTERNS:
        if re.search(pattern, qn):
            return True
    
    return False


def _detect_segment_in_query(query: str) -> Set[str]:
    """
    Detecta qué segmentos específicos se mencionan en la query.
    
    Returns:
        Set de segmentos canónicos detectados
    """
    qn = _norm_q(query)
    detected = set()
    
    for alias, canonical in SEGMENT_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, qn):
            detected.add(canonical)
    
    return detected

def _safe_get_str(d: dict, key: str) -> str:
    """Obtiene valor como string, manejando None y tipos no-string."""
    val = d.get(key)
    if val is None:
        return ""
    return str(val)

def _hit_has_segment(hit: Dict[str, Any], segment: str) -> bool:
    topic_parts = [
        _safe_get_str(hit, "page_segment"),
        _safe_get_str(hit, "topic_heuristic"),
        _safe_get_str(hit, "topic_llm"),
        _safe_get_str(hit, "topic_content"),
    ]
    
    topic_combined = _norm_q(" ".join(topic_parts))
    
    for alias, canonical in SEGMENT_ALIASES.items():
        if canonical == segment:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, topic_combined):
                return True
    
    return False


def _hit_get_segment(hit: Dict[str, Any]) -> str:
    """
    Obtiene el segmento de un hit, o "general" si no tiene.
    """
    topic_parts = [
        hit.get("page_segment") or "",
        hit.get("topic_heuristic") or "",
        hit.get("topic_llm") or "",
        hit.get("topic_content") or "",
    ]
    topic_combined = _norm_q(" ".join(topic_parts))
    
    for alias, canonical in SEGMENT_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, topic_combined):
            return canonical
    
    return "general"


def _hit_has_any_segment(hit: Dict[str, Any]) -> bool:
    """
    Verifica si un hit menciona ALGÚN segmento en sus topics.
    """
    return _hit_get_segment(hit) != "general"


def _is_likely_consolidated_table(hit: Dict[str, Any]) -> bool:
    """
    Detecta si un hit general probablemente contiene información consolidada
    de múltiples segmentos (típicamente una tabla P&L general).
    
    Indicadores:
    - Es de tipo 'table'
    - El texto contiene múltiples menciones de segmentos
    - O contiene palabras clave de consolidación
    """
    # Solo considerar tablas
    if hit.get("chunk_type") != "table":
        return False
    
    # Obtener texto del chunk
    chunk_text = _norm_q(hit.get("chunk_text") or "")
    
    # Contar menciones de diferentes segmentos en el texto
    segments_found = set()
    for alias, canonical in SEGMENT_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, chunk_text):
            segments_found.add(canonical)
    
    # Si tiene 3+ segmentos diferentes, probablemente es consolidada
    if len(segments_found) >= 3:
        return True
    
    # Buscar palabras clave de consolidación
    consolidation_patterns = [
        r"\btotal\s+(?:segmentos?|bancas?)\b",
        r"\bconsolidado\b",
        r"\bagregado\b",
        r"\btodos?\s+(?:los?\s+)?(?:segmentos?|bancas?)\b",
        r"\bresumen\s+(?:por\s+)?(?:segmento|banca)\b",
        r"\bdesglose\s+(?:por\s+)?(?:segmento|banca)\b",
    ]
    
    for pattern in consolidation_patterns:
        if re.search(pattern, chunk_text):
            return True
    
    return False


def _balance_segments(hits: List[Dict[str, Any]], max_per_segment: int = 3) -> List[Dict[str, Any]]:
    """
    Balancea los hits para tener representación de múltiples segmentos.
    
    Estrategia: round-robin entre segmentos para asegurar diversidad.
    """
    # Agrupar por segmento
    by_segment = defaultdict(list)
    for h in hits:
        seg = _hit_get_segment(h)
        by_segment[seg].append(h)
    
    # Round-robin para balancear
    balanced = []
    segment_keys = list(by_segment.keys())
    indices = {seg: 0 for seg in segment_keys}
    
    # Primero, un hit de cada segmento
    for seg in segment_keys:
        if by_segment[seg]:
            balanced.append(by_segment[seg][0])
            indices[seg] = 1
    
    # Luego, llenar hasta max_per_segment por segmento
    for round_num in range(1, max_per_segment):
        for seg in segment_keys:
            if indices[seg] < len(by_segment[seg]) and indices[seg] < max_per_segment:
                balanced.append(by_segment[seg][indices[seg]])
                indices[seg] += 1
    
    return balanced


def drop_segment_topics_if_query_general(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filtrado inteligente de segmentos con 3 comportamientos:
    
    1. Query COMPARATIVA entre segmentos:
       → PRIORIZAR hits generales (tablas consolidadas)
       → COMPLEMENTAR con hits de segmentos específicos balanceados
       → Ejemplo: "¿Qué segmento ha generado más ingresos?"
       
    2. Query CON segmento específico:
       → Mantener SOLO hits con ESE segmento
       → Ejemplo: "Ingresos de institucional octubre 2025"
       
    3. Query SIN segmento (general):
       → Descartar hits CON segmentos
       → Ejemplo: "Resultado operativo octubre 2025"
    """
    if not hits:
        return hits

    qn = _norm_q(query)
    
    # ============================================================
    # CASO 1: Query comparativa entre segmentos
    # → PRIORIZAR tablas generales consolidadas
    # → COMPLEMENTAR con hits específicos si es necesario
    # ============================================================
    if _is_comparative_query(query):
        # Separar hits generales vs específicos
        general_hits = [h for h in hits if not _hit_has_any_segment(h)]
        segment_hits = [h for h in hits if _hit_has_any_segment(h)]
        
        result = []
        
        # PRIORIDAD 1A: Tablas generales que probablemente son consolidadas
        # (tienen múltiples segmentos en el contenido)
        consolidated_tables = [h for h in general_hits if _is_likely_consolidated_table(h)]
        if consolidated_tables:
            # Tomar las 3-5 mejor rankeadas
            result.extend(consolidated_tables[:5])
        
        # PRIORIDAD 1B: Otras tablas/texto generales si no hay consolidadas
        if len(result) < 2:
            other_general = [h for h in general_hits if not _is_likely_consolidated_table(h)]
            result.extend(other_general[:3 - len(result)])
        
        # PRIORIDAD 2: Complementar con hits de segmentos específicos balanceados
        # Solo si no tenemos suficiente información general
        if len(result) < 3 and segment_hits:
            balanced = _balance_segments(segment_hits, max_per_segment=2)
            # Agregar hasta completar ~8 hits totales
            result.extend(balanced[:8 - len(result)])
        
        return result if result else hits
    
    # ============================================================
    # CASO 2: Query menciona segmento(s) específico(s)
    # → Mantener SOLO hits que mencionen alguno de esos segmentos
    # ============================================================
    query_segments = _detect_segment_in_query(query)
    
    if query_segments:
        out = []
        for h in hits:
            hit_matches_query_segment = any(
                _hit_has_segment(h, seg) for seg in query_segments
            )
            
            if hit_matches_query_segment:
                out.append(h)
        
        return out or hits  # fallback
    
    # ============================================================
    # CASO 3: Query NO menciona segmentos (general)
    # → Descartar hits que mencionen segmentos
    # ============================================================
    out = []
    for h in hits:
        if not _hit_has_any_segment(h):
            out.append(h)

    return out or hits  # fallback