"""
glossary_lookup.py - Búsqueda y expansión de términos del glosario de negocio.

Responsabilidades:
- Buscar términos en el glosario (exact match + fuzzy)
- Generar snippets de definiciones para prompts LLM
- Expandir queries con sinónimos y aliases
- Detectar frases y tokens del glosario

Dependencias:
- business_glossary.py (BUSINESS_GLOSSARY_V2)
- text_normalization.py (normalize_text)
"""

from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple, Any, Optional
from bind_rag_agent.vector_search.business_glossary import BUSINESS_GLOSSARY_V2
from bind_rag_agent.text_utils import normalize_text as _norm

# ============================================================
# BÚSQUEDA EN GLOSARIO
# ============================================================

def _find_glossary_matches(
    query: str, 
    fuzzy_threshold: float = 0.88
) -> List[str]:
    """
    Encuentra términos del glosario que coinciden con la query.
    
    Estrategias (en orden):
    1. Match exacto: si el alias normalizado está contenido en la query
    2. Fuzzy match: si la similitud supera el threshold
    
    Args:
        query: Query normalizada
        fuzzy_threshold: Umbral para fuzzy matching (0.0 - 1.0)
        
    Returns:
        Lista de acrónimos del glosario que coinciden
    """
    q = _norm(query)
    if not q:
        return []
    
    found = []
    
    # 1) Match exacto por "contains" sobre aliases normalizados
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = obj.get("aliases", [])
        candidates = [acronym] + list(aliases)
        
        for alias in candidates:
            if _norm(alias) in q:
                found.append(acronym)
                break
    
    # 2) Fuzzy fallback si no encontró nada
    if not found:
        tokens = q.split()
        
        for acronym, obj in BUSINESS_GLOSSARY_V2.items():
            best_score = 0.0
            
            for alias in [acronym] + list(obj.get("aliases", [])):
                alias_norm = _norm(alias)
                if not alias_norm:
                    continue
                
                # Comparar con query completa
                best_score = max(best_score, SequenceMatcher(None, alias_norm, q).ratio())
                
                # Comparar con tokens individuales (para palabras sueltas)
                if " " not in alias_norm:
                    for token in tokens:
                        best_score = max(best_score, SequenceMatcher(None, alias_norm, token).ratio())
            
            if best_score >= fuzzy_threshold:
                found.append(acronym)
    
    # Eliminar duplicados preservando orden
    return list(dict.fromkeys(found))


def glossary_snippet(
    query: str, 
    max_terms: int = 12, 
    fuzzy_threshold: float = 0.88
) -> str:
    """
    Genera un snippet con definiciones del glosario para incluir en prompts.
    
    El snippet incluye solo los términos relevantes para la query,
    formateado para que el LLM entienda que son definiciones internas.
    
    Args:
        query: Pregunta del usuario
        max_terms: Máximo de términos a incluir
        fuzzy_threshold: Umbral para fuzzy matching
        
    Returns:
        String con definiciones formateadas, o string vacío si no hay matches
        
    Example:
        >>> glossary_snippet("¿Cuál fue el ROA del banco?")
        'GLOSARIO (definiciones internas...):
        - ROA: Return on Assets / Retorno sobre Activos'
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return ""
    
    found = _find_glossary_matches(q_raw, fuzzy_threshold)[:max_terms]
    
    if not found:
        return ""
    
    lines = [
        "GLOSARIO (definiciones internas para interpretar términos; NO son evidencia del documento):"
    ]
    
    for acronym in found:
        desc = BUSINESS_GLOSSARY_V2[acronym].get('desc', '')
        lines.append(f"- {acronym}: {desc}")
    
    return "\n".join(lines)


def glossary_expand_terms(
    query: str, 
    max_terms: int = 12, 
    fuzzy_threshold: float = 0.88
) -> Dict[str, List[str]]:
    """
    Expande la query con términos del glosario para mejorar retrieval.
    
    Retorna los acrónimos encontrados y todos sus aliases/sinónimos,
    útil para enriquecer la query antes de embedding o búsqueda léxica.
    
    Args:
        query: Pregunta del usuario
        max_terms: Máximo de acrónimos a procesar
        fuzzy_threshold: Umbral para fuzzy matching
        
    Returns:
        Dict con:
        - "acronyms": Lista de acrónimos encontrados
        - "terms": Lista de todos los términos (acrónimos + aliases)
        
    Example:
        >>> glossary_expand_terms("¿Cuál fue el ROA?")
        {
            "acronyms": ["ROA"],
            "terms": ["ROA", "roa", "return on assets", "retorno sobre activos", ...]
        }
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return {"acronyms": [], "terms": []}
    
    found = _find_glossary_matches(q_raw, fuzzy_threshold)[:max_terms]
    
    if not found:
        return {"acronyms": [], "terms": []}
    
    # Recopilar todos los términos (acrónimo + aliases)
    terms = []
    for acronym in found:
        terms.append(acronym)
        terms.extend(BUSINESS_GLOSSARY_V2[acronym].get("aliases", []))
    
    # Deduplicar preservando orden
    unique_terms = []
    seen = set()
    for term in terms:
        term_norm = _norm(term)
        if term_norm and term_norm not in seen:
            seen.add(term_norm)
            unique_terms.append(term)
    
    return {"acronyms": found, "terms": unique_terms}


# ============================================================
# SCORING DE HITS
# ============================================================

def glossary_bonus(hit: Dict[str, Any], terms: List[str]) -> int:
    """
    Calcula un bonus score para un hit basado en términos del glosario.
    
    El bonus es mayor cuando:
    - El término aparece en el topic (más relevante)
    - El término es una frase multi-palabra (más específico)
    
    Args:
        hit: Diccionario con chunk_text y topic_heuristic
        terms: Lista de términos del glosario a buscar
        
    Returns:
        Score de bonus (entero >= 0)
        
    Scoring:
    - Frase en topic: +5
    - Palabra en topic: +2
    - Frase en texto: +3
    - Palabra en texto: +1
    """
    txt = hit.get("chunk_text_clean") or hit.get("chunk_text") or ""
    topic = hit.get("topic_heuristic") or ""
    
    combined = _norm(txt + " " + topic)
    topic_norm = _norm(topic)
    
    bonus = 0
    seen = set()
    
    for term in terms or []:
        term_norm = _norm(term)
        if not term_norm or term_norm in seen:
            continue
        
        seen.add(term_norm)
        
        if term_norm in combined:
            is_phrase = " " in term
            in_topic = term_norm in topic_norm
            
            if in_topic:
                bonus += 5 if is_phrase else 2
            else:
                bonus += 3 if is_phrase else 1
    
    return bonus


# ============================================================
# DETECCIÓN DE FRASES Y TOKENS
# ============================================================

def detect_glossary_phrases(query: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """
    Detecta frases multi-palabra del glosario en la query.
    
    Las frases se detectan en orden de longitud (más largas primero)
    para evitar que frases cortas "consuman" palabras de frases más largas.
    
    Args:
        query: Query del usuario
        
    Returns:
        Tuple de:
        - Lista de (frase_normalizada, acronimo) encontradas
        - Set de palabras ya consumidas
        
    Example:
        >>> detect_glossary_phrases("retorno sobre activos del banco")
        ([("retorno sobre activos", "roa")], {"retorno", "sobre", "activos"})
    """
    qn = _norm(query)
    
    found_phrases: List[Tuple[str, str]] = []
    consumed_words: Set[str] = set()
    
    # Recopilar todas las frases (aliases multi-palabra)
    all_phrases = []
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = [acronym.lower()] + [a.lower() for a in obj.get("aliases", [])]
        
        for alias in aliases:
            alias_norm = _norm(alias)
            # Solo frases (más de una palabra)
            if alias_norm and " " in alias_norm:
                all_phrases.append((alias_norm, acronym.lower(), len(alias_norm)))
    
    # Ordenar por longitud descendente (frases más largas primero)
    all_phrases.sort(key=lambda x: x[2], reverse=True)
    
    # Detectar frases sin superposición
    for alias_norm, acronym, _ in all_phrases:
        if alias_norm in qn:
            alias_words = set(alias_norm.split())
            
            # Solo agregar si no hay superposición con palabras ya consumidas
            if not alias_words.intersection(consumed_words):
                found_phrases.append((alias_norm, acronym))
                consumed_words.update(alias_words)
    
    return found_phrases, consumed_words


def detect_glossary_tokens(
    query: str, 
    consumed: Optional[Set[str]] = None
) -> List[Tuple[str, str]]:
    """
    Detecta acrónimos/tokens simples del glosario en la query.
    
    Solo detecta palabras individuales (no frases), excluyendo
    las que ya fueron consumidas por detect_glossary_phrases.
    
    Args:
        query: Query del usuario
        consumed: Set de palabras ya consumidas (opcional)
        
    Returns:
        Lista de (token_normalizado, acronimo) encontrados
        
    Example:
        >>> detect_glossary_tokens("¿Cuál fue el ROA?", set())
        [("roa", "roa")]
    """
    qn = _norm(query)
    words = set(qn.split())
    consumed = consumed or set()
    
    found_tokens: List[Tuple[str, str]] = []
    
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        acr_norm = _norm(acronym)
        
        # Verificar el acrónimo principal
        if acr_norm in words and acr_norm not in consumed:
            found_tokens.append((acr_norm, acronym.lower()))
            continue
        
        # Verificar aliases de una sola palabra
        for alias in obj.get("aliases", []):
            alias_norm = _norm(alias)
            
            # Solo tokens (una palabra) no consumidos
            if " " not in alias_norm and alias_norm in words and alias_norm not in consumed:
                found_tokens.append((alias_norm, acronym.lower()))
                break
    
    return found_tokens
