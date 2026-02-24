"""
glossary_helper.py - Orquestador de búsqueda con glosario y entidades.

Este módulo combina la funcionalidad de:
- text_normalization.py: Normalización de texto
- glossary_lookup.py: Búsqueda en glosario de negocio
- entity_detection.py: Detección de entidades nombradas

Función principal: prepare_rerank_candidates_glossary_aware()

NOTA: Este archivo mantiene compatibilidad hacia atrás re-exportando
las funciones públicas. El código existente puede seguir importando:
    from bind_rag_agent.vector_search.glossary_helper import glossary_snippet
"""

import ast
import json
import re
from typing import Any, Dict, List, Set, Optional

# ============================================================
# IMPORTS DE MÓDULOS SEPARADOS
# ============================================================

from bind_rag_agent.text_utils import (
    normalize_text as _norm,
    normalize_for_search as _norm_for_match,
)

from bind_rag_agent.vector_search.glossary_lookup import (
    glossary_snippet,
    glossary_expand_terms,
    glossary_bonus,
    detect_glossary_phrases,
    detect_glossary_tokens,
)

from bind_rag_agent.vector_search.entity_detection import (
    detect_named_entities,
    STOPWORDS_ES as _STOP,
    MONTHS_ES as _MONTHS,
)


# ============================================================
# API PÚBLICA - Re-exportar para compatibilidad
# ============================================================

__all__ = [
    # Funciones principales
    'glossary_snippet',
    'glossary_expand_terms',
    'glossary_bonus',
    'prepare_rerank_candidates_glossary_aware',
    'compute_metadata_bonus',
    
    # Funciones de detección (para uso interno y testing)
    'extract_query_anchors',
    'compute_anchor_score',
    
    # Aliases internos (para compatibilidad con código legacy)
    '_extract_query_anchors',
    '_compute_anchor_score',
]


# ============================================================
# EXTRACCIÓN DE ANCHORS
# ============================================================

def extract_query_anchors(query: str) -> List[str]:
    """
    Extrae anchors (términos clave) de la query para matching.
    
    Los anchors se extraen respetando el siguiente orden de prioridad:
    1. Entidades nombradas completas (ej: "Banco Santander" como frase)
    2. Frases del glosario (ej: "retorno sobre activos" → frase + acrónimo "roa")
    3. Tokens del glosario (ej: "YTD" → acrónimo)
    4. Palabras restantes (excluyendo stopwords y meses)
    
    Args:
        query: Pregunta del usuario
        
    Returns:
        Lista de anchors en orden de extracción
        
    Example:
        >>> extract_query_anchors("¿Cuál fue el ROA del Banco Santander en octubre?")
        ['banco santander', 'roa', 'octubre']
        
        # Nota: "Banco Santander" es UNA frase, no dos palabras separadas
        # Esto evita que "banco" matchee con "BIND Banco" incorrectamente
    """
    qn = _norm(query)
    
    anchors: List[str] = []
    consumed_words: Set[str] = set()
    
    # ============================================================
    # Paso 1: Detectar entidades nombradas PRIMERO
    # Esto asegura que "Banco Santander" se trate como unidad
    # ============================================================
    named_entities = detect_named_entities(query)
    for entity, entity_words in named_entities:
        anchors.append(entity)
        consumed_words.update(entity_words)
    
    # ============================================================
    # Paso 2: Detectar frases del glosario
    # Solo las que no tengan palabras ya consumidas
    # ============================================================
    phrases, phrase_words = detect_glossary_phrases(query)
    for phrase, acronym in phrases:
        phrase_word_set = set(phrase.split())
        
        if not phrase_word_set.intersection(consumed_words):
            anchors.append(phrase)
            anchors.append(acronym)
            consumed_words.update(phrase_word_set)
    
    # ============================================================
    # Paso 3: Detectar tokens/acrónimos del glosario
    # ============================================================
    tokens = detect_glossary_tokens(query, consumed_words)
    for token, acronym in tokens:
        if token not in consumed_words:
            anchors.append(acronym)
            consumed_words.add(token)
    
    # ============================================================
    # Paso 4: Extraer palabras restantes
    # Excluir stopwords, meses, y palabras muy cortas
    # ============================================================
    words = re.findall(r"[a-z0-9]+", qn)
    
    for word in words:
        # Saltar si ya fue consumida
        if word in consumed_words:
            continue
        
        # Saltar stopwords
        if word in _STOP:
            continue
        
        # Saltar meses (son contexto temporal, no keywords)
        if word in _MONTHS:
            continue
        
        # Saltar palabras muy cortas (excepto años de 4 dígitos)
        if len(word) < 3:
            if not (word.isdigit() and len(word) == 4):
                continue
        
        anchors.append(word)
        consumed_words.add(word)
    
    # ============================================================
    # Paso 5: Deduplicar preservando orden
    # ============================================================
    seen: Set[str] = set()
    unique: List[str] = []
    
    for anchor in anchors:
        if anchor not in seen:
            seen.add(anchor)
            unique.append(anchor)
    
    return unique


def compute_anchor_score(hit: Dict[str, Any], anchors: List[str]) -> int:
    """
    Calcula el anchor score para un hit (chunk).
    
    El score mide qué tan bien el hit matchea con los anchors de la query.
    Se usa para priorizar hits que contienen los términos exactos buscados.
    
    Scoring:
    - Entidad/frase completa (ej: "banco santander"): +5 puntos
    - Palabra individual: +1 punto
    
    Args:
        hit: Diccionario con chunk_text_clean y topic_heuristic
        anchors: Lista de anchors extraídos de la query
        
    Returns:
        Score total (entero >= 0)
        
    Example:
        >>> hit = {"chunk_text_clean": "El Banco Santander reportó..."}
        >>> compute_anchor_score(hit, ["banco santander", "roa"])
        5  # Matchea "banco santander" (+5), no matchea "roa" (+0)
    """
    txt = _norm(hit.get("chunk_text_clean") or hit.get("chunk_text") or "")
    topic = _norm(hit.get("topic_heuristic") or "")
    combined = txt + " " + topic
    
    score = 0
    
    for anchor in anchors:
        if " " in anchor:
            # Es una entidad/frase → bonus alto si matchea completa
            if anchor in combined:
                score += 5
        else:
            # Es una palabra/acrónimo → bonus estándar con word boundary
            pattern = rf"\b{re.escape(anchor)}\b"
            if re.search(pattern, combined):
                score += 1
    
    return score


# ============================================================
# METADATA ENRICH BONUS
# ============================================================

def _parse_metadata_enrich(hit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parsea metadata_enrich de un hit.
    
    Soporta formatos frecuentes de Vector Search:
    - dict nativo
    - JSON string
    - string con representación Python (single quotes)
    - bytes
    - objetos con asDict() (ej. Row)
    """
    def _is_non_empty(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            s = v.strip().lower()
            return s not in {"", "null", "none", "{}", "[]"}
        if isinstance(v, (list, tuple, set, dict)):
            return len(v) > 0
        if isinstance(v, (bytes, bytearray)):
            return len(v) > 0
        return True
    
    def _pick(keys: List[str]) -> Any:
        for key in keys:
            if key in hit:
                val = hit.get(key)
                if _is_non_empty(val):
                    return val
        return None
    
    def _metadata_from_flat_fields() -> Dict[str, Any]:
        extracted: Dict[str, Any] = {}
        
        field_candidates = {
            "table_metrics": [
                "table_metrics",
                "metadata_enrich.table_metrics",
                "metadata_enrich_table_metrics",
                "metadata_table_metrics",
                "metadata.table_metrics",
                "tableMetrics",
                "metrics",
            ],
            "keywords": [
                "keywords",
                "metadata_enrich.keywords",
                "metadata_enrich_keywords",
                "metadata_keywords",
                "metadata.keywords",
            ],
            "entities": [
                "entities",
                "metadata_enrich.entities",
                "metadata_enrich_entities",
                "metadata_entities",
                "metadata.entities",
            ],
            "data_period": [
                "data_period",
                "metadata_enrich.data_period",
                "metadata_enrich_data_period",
                "metadata_data_period",
                "period",
                "anio",
                "year",
            ],
            "content_category": [
                "content_category",
                "metadata_enrich.content_category",
                "metadata_enrich_content_category",
                "metadata_content_category",
                "category",
            ],
        }
        
        for target_key, candidates in field_candidates.items():
            val = _pick(candidates)
            if _is_non_empty(val):
                extracted[target_key] = val
        
        return extracted
    
    raw = None
    for raw_key in (
        "metadata_enrich",
        "metadata_enrich_json",
        "metadata_json",
        "metadata",
        "metadata_enriched",
    ):
        if raw_key in hit:
            candidate = hit.get(raw_key)
            if _is_non_empty(candidate):
                raw = candidate
                break
    
    if raw is None:
        raw = _metadata_from_flat_fields()
        if not raw:
            return {}
    
    if hasattr(raw, "asDict"):
        try:
            raw = raw.asDict(recursive=True)
        except Exception:
            pass
    
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return {}
    
    if isinstance(raw, dict):
        # Completar con posibles columnas flat si faltan campos
        flat_meta = _metadata_from_flat_fields()
        if flat_meta:
            merged = dict(raw)
            for k, v in flat_meta.items():
                if not _is_non_empty(merged.get(k)):
                    merged[k] = v
            return merged
        return raw
    
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        
        # Intentar JSON (incluye caso doble-serializado)
        for _ in range(2):
            try:
                parsed = json.loads(s)
            except (json.JSONDecodeError, TypeError, ValueError):
                break
            
            if isinstance(parsed, dict):
                flat_meta = _metadata_from_flat_fields()
                if flat_meta:
                    merged = dict(parsed)
                    for k, v in flat_meta.items():
                        if not _is_non_empty(merged.get(k)):
                            merged[k] = v
                    return merged
                return parsed
            if isinstance(parsed, str):
                s = parsed.strip()
                continue
            break
        
        # Fallback: representación Python de dict/lista
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return _metadata_from_flat_fields() or {}
        
        if hasattr(parsed, "asDict"):
            try:
                parsed = parsed.asDict(recursive=True)
            except Exception:
                pass
        
        if isinstance(parsed, dict):
            flat_meta = _metadata_from_flat_fields()
            if flat_meta:
                merged = dict(parsed)
                for k, v in flat_meta.items():
                    if not _is_non_empty(merged.get(k)):
                        merged[k] = v
                return merged
            return parsed
        return _metadata_from_flat_fields() or {}
    
    return _metadata_from_flat_fields() or {}


def _metadata_field_to_list(value: Any) -> List[str]:
    """Normaliza un campo metadata (list/string/dict) a lista de strings."""
    if value is None:
        return []
    
    if hasattr(value, "asDict"):
        try:
            value = value.asDict(recursive=True)
        except Exception:
            pass
    
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return []
    
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(s)
                if isinstance(candidate, (list, tuple, set, dict)):
                    parsed = candidate
                    break
            except Exception:
                continue
        
        if parsed is not None:
            value = parsed
        else:
            return [s]
    
    if isinstance(value, dict):
        value = list(value.values())
    
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if hasattr(item, "asDict"):
                try:
                    item = item.asDict(recursive=True)
                except Exception:
                    pass
            if isinstance(item, dict):
                if "name" in item and item["name"] is not None:
                    out.append(str(item["name"]))
                elif "value" in item and item["value"] is not None:
                    out.append(str(item["value"]))
                else:
                    out.extend(str(v) for v in item.values() if v is not None)
            else:
                out.append(str(item))
        return out
    
    return [str(value)]


def compute_metadata_bonus(hit: Dict[str, Any], anchors: List[str]) -> int:
    """
    Calcula un bonus basado en coincidencias entre los anchors de la query
    y los campos de metadata_enrich (table_metrics, keywords, entities).
    
    Scoring:
    - Anchor matchea exacto en table_metrics:  +3 (más específico)
    - Anchor es substring de table_metrics:     +2 (match parcial, ej: "previsiones" en "Previsiones & Otros")
    - Anchor matchea en keywords:               +1
    - Anchor matchea en entities:               +2
    - Anchor matchea en data_period:            +2
    - Anchor matchea en content_category:       +1
    
    Args:
        hit: Diccionario con campo metadata_enrich
        anchors: Lista de anchors extraídos de la query
        
    Returns:
        Score total (entero >= 0)
    """
    meta = _parse_metadata_enrich(hit)
    if not meta or not anchors:
        return 0
    
    table_metrics = [_norm(m) for m in _metadata_field_to_list(meta.get("table_metrics")) if m]
    keywords = [_norm(k) for k in _metadata_field_to_list(meta.get("keywords")) if k]
    entities = [_norm(e) for e in _metadata_field_to_list(meta.get("entities")) if e]
    data_period = _norm(str(meta.get("data_period") or ""))
    content_category = _norm(str(meta.get("content_category") or ""))
    
    score = 0
    
    for anchor in anchors:
        anchor_n = _norm(anchor)
        if not anchor_n or len(anchor_n) < 3:
            continue
        
        # table_metrics: match exacto (+3) o substring (+2)
        for metric in table_metrics:
            if anchor_n == metric:
                score += 3
                break
            elif anchor_n in metric or metric in anchor_n:
                score += 2
                break
        
        # keywords: match exacto o substring (+1)
        for kw in keywords:
            if anchor_n == kw or anchor_n in kw or kw in anchor_n:
                score += 1
                break
        
        # entities: match exacto o substring (+2)
        for ent in entities:
            if anchor_n == ent or anchor_n in ent or ent in anchor_n:
                score += 2
                break
        
        # data_period: match exacto o substring (+2)
        if data_period and (anchor_n == data_period or anchor_n in data_period or data_period in anchor_n):
            score += 2
        
        # content_category: match exacto o substring (+1)
        if content_category and (
            anchor_n == content_category or anchor_n in content_category or content_category in anchor_n
        ):
            score += 1
    
    return score


# Aliases para compatibilidad con código que usa nombres con underscore
_extract_query_anchors = extract_query_anchors
_compute_anchor_score = compute_anchor_score


# ============================================================
# FUNCIÓN PRINCIPAL: PREPARACIÓN DE CANDIDATOS
# ============================================================

def _file_date_as_int(hit: Dict[str, Any]) -> int:
    """Convierte file_date a int para comparación numérica. '2025-11-18' → 20251118."""
    fd = hit.get("file_date") or ""
    try:
        return int(fd.replace("-", ""))
    except (ValueError, AttributeError):
        return 0


def _relevance_sort_key(h: Dict[str, Any]) -> tuple:
    """Sort key estándar por relevancia (sin fecha)."""
    return (
        h.get("_anchor_score", 0) * 100,
        h.get("_metadata_bonus", 0) * 10,
        h.get("_glossary_bonus", 0),
        float(h.get("score") or h.get("similarity") or 0.0),
    )


def _select_with_recency_priority(
    hits: List[Dict[str, Any]],
    max_input: int,
    recency_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Selecciona hasta max_input hits garantizando que el archivo más reciente
    tenga representación prioritaria, sin sacrificar relevancia.

    Estrategia:
      1. Agrupar hits por file_date (cada fecha = un archivo mensual).
      2. Ordenar grupos de más reciente a más antiguo.
      3. Reservar recency_ratio × max_input slots para el archivo más reciente
         (solo llenados con hits que tengan score > 0, es decir, relevantes).
      4. Llenar el resto con los mejores hits de los demás archivos.
      5. Dentro de cada grupo, mantener orden por relevancia.

    Args:
        hits: Lista de hits ya con scores calculados (_anchor_score, etc.)
        max_input: Máximo total de candidatos a retornar.
        recency_ratio: Fracción de slots reservados para el archivo más reciente.
                       Default desde config RAG_RECENCY_RATIO (0.60).
                       Ej: con max_input=24 → 14 slots reservados.

    Returns:
        Lista de hasta max_input hits, con prioridad al archivo más reciente.
    """
    from bind_rag_agent.config import RECENCY_RATIO as _DEFAULT_RATIO

    if recency_ratio is None:
        recency_ratio = _DEFAULT_RATIO
    if not hits or max_input <= 0:
        return []

    # --- Agrupar por file_date ---
    from collections import defaultdict
    by_date: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for h in hits:
        fd_int = _file_date_as_int(h)
        by_date[fd_int].append(h)

    # Ordenar cada grupo internamente por relevancia
    for fd_int in by_date:
        by_date[fd_int].sort(key=_relevance_sort_key, reverse=True)

    # Ordenar fechas de más reciente a más antigua
    sorted_dates = sorted(by_date.keys(), reverse=True)

    if len(sorted_dates) <= 1:
        # Solo hay un archivo (o ninguno con fecha), no hace falta lógica especial
        all_sorted = sorted(hits, key=_relevance_sort_key, reverse=True)
        return all_sorted[:max_input]

    # --- Asignar slots ---
    newest_date = sorted_dates[0]
    newest_hits = by_date[newest_date]

    # Solo reservar slots para hits relevantes del archivo más reciente
    # (anchor_score > 0 indica que el chunk tiene match con la query)
    newest_relevant = [
        h for h in newest_hits if h.get("_anchor_score", 0) > 0
    ]

    reserved_slots = min(
        int(max_input * recency_ratio),
        len(newest_relevant),
    )
    # Garantizar al menos max_input // 3 para el más reciente si tiene hits relevantes
    if newest_relevant:
        reserved_slots = max(reserved_slots, min(max_input // 3, len(newest_relevant)))

    selected_newest = newest_relevant[:reserved_slots]
    selected_ids = {h.get("chunk_id") for h in selected_newest}

    # --- Llenar remaining_slots con los mejores hits de otros archivos ---
    remaining_slots = max_input - len(selected_newest)
    rest_hits = [h for h in hits if h.get("chunk_id") not in selected_ids]
    rest_hits.sort(key=_relevance_sort_key, reverse=True)
    selected_rest = rest_hits[:remaining_slots]

    # --- Combinar: newest primero (por relevancia), luego el resto ---
    result = selected_newest + selected_rest
    return result


def prepare_rerank_candidates_glossary_aware(
    query: str,
    hits: List[Dict[str, Any]],
    max_input: int,
) -> List[Dict[str, Any]]:
    """
    Prepara y ordena candidatos para reranking usando conocimiento del glosario.
    
    Esta función es el corazón del pre-procesamiento antes del reranking LLM.
    Ordena los hits por múltiples criterios para que los más relevantes
    lleguen primero al reranker.
    
    Criterios de ordenamiento (en orden de prioridad):
    1. Anchor score × 100 (match con entidades y keywords)
    2. Glossary bonus (match con términos del glosario)
    3. Base score (similarity score del vector search)
    
    RECENCY PRIORITY (NUEVO):
    Cuando hay múltiples archivos mensuales con contenido equivalente,
    se reserva ~60% de los slots para el archivo más reciente (por file_date).
    Esto garantiza que el reranker LLM reciba suficientes chunks del archivo
    más actual, evitando que se diluyan entre archivos históricos con
    scores idénticos.
    
    Side effects:
    - Agrega "_anchor_score" a cada hit
    - Agrega "_glossary_bonus" a cada hit
    - Agrega "_glossary_phrase_hits" a cada hit
    - Agrega "_glossary_token_hits" a cada hit
    
    Args:
        query: Pregunta del usuario
        hits: Lista de hits (chunks) del retrieval
        max_input: Máximo número de candidatos a retornar
        
    Returns:
        Lista ordenada de hits, truncada a max_input
    """
    if not hits:
        return []
    
    # ============================================================
    # Paso 1: Calcular anchor scores
    # ============================================================
    anchors = extract_query_anchors(query)
    
    for hit in hits:
        anchor_score = compute_anchor_score(hit, anchors) if anchors else 0
        hit["_anchor_score"] = anchor_score
        
        metadata_bonus = compute_metadata_bonus(hit, anchors) if anchors else 0
        hit["_metadata_bonus"] = metadata_bonus
    
    # ============================================================
    # Paso 2: Obtener términos del glosario
    # ============================================================
    gl = glossary_expand_terms(query)
    terms = gl.get("terms", []) or []
    
    # Si no hay términos del glosario, ordenar solo por anchor score + metadata
    if not terms:
        for hit in hits:
            hit["_glossary_bonus"] = 0
            hit["_glossary_phrase_hits"] = 0
            hit["_glossary_token_hits"] = 0
        
        return _select_with_recency_priority(hits, max_input)
    
    # ============================================================
    # Paso 3: Separar frases vs tokens del glosario
    # ============================================================
    phrases: List[str] = []
    tokens: List[str] = []
    
    for term in terms:
        term = (term or "").strip()
        if not term:
            continue
        
        if " " in term:
            phrases.append(term)
        else:
            # Solo tokens cortos o con caracteres especiales
            term_lower = term.lower()
            if len(term_lower) <= 5 or "%" in term or "." in term:
                tokens.append(term)
    
    # Deduplicar
    def _dedupe(items: List[str]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for item in items:
            item_norm = _norm(item)
            if item_norm and item_norm not in seen:
                seen.add(item_norm)
                out.append(item)
        return out
    
    phrases = _dedupe(phrases)
    tokens = _dedupe(tokens)
    
    # Si no hay frases ni tokens útiles, ordenar solo por anchor score + metadata
    if not phrases and not tokens:
        for hit in hits:
            hit["_glossary_bonus"] = 0
        
        return _select_with_recency_priority(hits, max_input)
    
    # ============================================================
    # Paso 4: Compilar patterns para matching eficiente
    # ============================================================
    def _phrase_pattern(phrase: str) -> Optional[re.Pattern]:
        """Crea pattern para frase multi-palabra."""
        phrase_norm = _norm(phrase)
        parts = [re.escape(w) for w in phrase_norm.split() if w]
        if not parts:
            return None
        regex = r"\b" + r"\b\s+\b".join(parts) + r"\b"
        return re.compile(regex, flags=re.IGNORECASE)
    
    phrase_patterns = [
        pat for pat in (_phrase_pattern(p) for p in phrases) 
        if pat is not None
    ]
    
    token_patterns: List[re.Pattern] = []
    for token in tokens:
        token_norm = _norm(token)
        if not token_norm:
            continue
        # Pattern para token exacto
        token_patterns.append(
            re.compile(rf"\b{re.escape(token_norm)}\b", flags=re.IGNORECASE)
        )
        # Pattern para token seguido de % (ej: "15%")
        token_patterns.append(
            re.compile(rf"\b{re.escape(token_norm)}\s*%", flags=re.IGNORECASE)
        )
    
    # ============================================================
    # Paso 5: Calcular glossary bonus para cada hit
    # ============================================================
    def _count_matches(hit: Dict[str, Any]) -> tuple:
        """Cuenta matches de frases y tokens en un hit."""
        txt = hit.get("chunk_text_clean") or hit.get("chunk_text") or ""
        topic = hit.get("topic_heuristic") or ""
        combined = _norm(txt + " " + topic)
        
        phrase_hits = sum(1 for pat in phrase_patterns if pat.search(combined))
        token_hits = sum(1 for pat in token_patterns if pat.search(combined))
        
        return phrase_hits, token_hits
    
    def _base_score(hit: Dict[str, Any]) -> float:
        """Obtiene el score base (similarity) del hit."""
        return float(hit.get("score") or hit.get("similarity") or 0.0)
    
    for hit in hits:
        phrase_hits, token_hits = _count_matches(hit)
        
        hit["_glossary_phrase_hits"] = phrase_hits
        hit["_glossary_token_hits"] = token_hits
        
        # Calcular bonus: frases valen más que tokens
        glossary_bonus_val = phrase_hits * 50 + token_hits * 5
        
        # Penalizar hits sin ningún match del glosario
        penalty = 30 if (phrase_hits == 0 and token_hits == 0) else 0
        
        hit["_glossary_bonus"] = glossary_bonus_val - penalty
    
    # ============================================================
    # Paso 6: Selección con prioridad de recencia
    # ============================================================
    return _select_with_recency_priority(hits, max_input)