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


# Aliases para compatibilidad con código que usa nombres con underscore
_extract_query_anchors = extract_query_anchors
_compute_anchor_score = compute_anchor_score


# ============================================================
# FUNCIÓN PRINCIPAL: PREPARACIÓN DE CANDIDATOS
# ============================================================

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
    
    # ============================================================
    # Paso 2: Obtener términos del glosario
    # ============================================================
    gl = glossary_expand_terms(query)
    terms = gl.get("terms", []) or []
    
    # Si no hay términos del glosario, ordenar solo por anchor score
    if not terms:
        for hit in hits:
            hit["_glossary_bonus"] = 0
            hit["_glossary_phrase_hits"] = 0
            hit["_glossary_token_hits"] = 0
        
        ranked = sorted(
            hits,
            key=lambda h: (
                h.get("_anchor_score", 0),
                float(h.get("score") or h.get("similarity") or 0.0)
            ),
            reverse=True
        )
        return ranked[:max_input]
    
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
    
    # Si no hay frases ni tokens útiles, ordenar solo por anchor score
    if not phrases and not tokens:
        for hit in hits:
            hit["_glossary_bonus"] = 0
        
        ranked = sorted(
            hits,
            key=lambda h: (
                h.get("_anchor_score", 0),
                float(h.get("score") or h.get("similarity") or 0.0)
            ),
            reverse=True
        )
        return ranked[:max_input]
    
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
    # Paso 6: Ordenar por criterios múltiples
    # ============================================================
    ranked = sorted(
        hits,
        key=lambda h: (
            h.get("_anchor_score", 0) * 100,  # Prioridad 1: anchor score
            h.get("_glossary_bonus", 0),       # Prioridad 2: glossary bonus
            _base_score(h)                      # Prioridad 3: similarity score
        ),
        reverse=True
    )
    
    return ranked[:max_input]
