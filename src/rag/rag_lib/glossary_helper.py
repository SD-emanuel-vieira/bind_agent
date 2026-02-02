"""
glossary_helper.py - Utilidades de glosario y extracción de anchors

ACTUALIZADO:
- Detección de entidades nombradas (named entities) en anchor extraction
- "Banco Santander" se trata como frase completa, no como "banco" + "santander" separados
- Esto evita que chunks con "BIND Banco" obtengan puntos cuando se pregunta por Santander
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Set
from rag_lib.business_glossary import BUSINESS_GLOSSARY_V2


def _norm(s: str) -> str:
    """Normalize for matching: lowercase + remove accents + collapse whitespace."""
    s = (s or "").strip().lower()
    s = "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )
    s = re.sub(r"\s+", " ", s)
    return s


def glossary_snippet(query: str, max_terms: int = 12, fuzzy_threshold: float = 0.88) -> str:
    q_raw = (query or "").strip()
    if not q_raw:
        return ""
    q = _norm(q_raw)

    # 1) match por "contains" sobre aliases normalizados
    found = []
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = obj.get("aliases", [])
        candidates = [acronym] + list(aliases)
        for a in candidates:
            if _norm(a) in q:
                found.append(acronym)
                break

    # 2) fuzzy fallback si no encontró nada
    if not found:
        tokens = q.split()
        for acronym, obj in BUSINESS_GLOSSARY_V2.items():
            best = 0.0
            for a in [acronym] + list(obj.get("aliases", [])):
                an = _norm(a)
                if not an:
                    continue
                best = max(best, SequenceMatcher(None, an, q).ratio())
                if " " not in an:
                    for t in tokens:
                        best = max(best, SequenceMatcher(None, an, t).ratio())
            if best >= fuzzy_threshold:
                found.append(acronym)

    found = list(dict.fromkeys(found))[:max_terms]
    if not found:
        return ""

    lines = ["GLOSARIO (definiciones internas para interpretar términos; NO son evidencia del documento):"]
    for ac in found:
        lines.append(f"- {ac}: {BUSINESS_GLOSSARY_V2[ac]['desc']}")
    return "\n".join(lines)


def glossary_expand_terms(query: str, max_terms: int = 12, fuzzy_threshold: float = 0.88) -> dict:
    """
    Devuelve:
      {
        "acronyms": ["ROA", ...],
        "terms": ["ROA", "roa", "retorno sobre activos", ...]
      }
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return {"acronyms": [], "terms": []}

    q = _norm(q_raw)

    found = []
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = obj.get("aliases", [])
        candidates = [acronym] + list(aliases)
        for a in candidates:
            if _norm(a) in q:
                found.append(acronym)
                break

    if not found:
        tokens = q.split()
        for acronym, obj in BUSINESS_GLOSSARY_V2.items():
            best = 0.0
            for a in [acronym] + list(obj.get("aliases", [])):
                an = _norm(a)
                if not an:
                    continue
                best = max(best, SequenceMatcher(None, an, q).ratio())
                if " " not in an:
                    for t in tokens:
                        best = max(best, SequenceMatcher(None, an, t).ratio())
            if best >= fuzzy_threshold:
                found.append(acronym)

    found = list(dict.fromkeys(found))[:max_terms]
    if not found:
        return {"acronyms": [], "terms": []}

    terms = []
    for ac in found:
        terms.append(ac)
        terms.extend(BUSINESS_GLOSSARY_V2[ac].get("aliases", []))

    out = []
    seen = set()
    for t in terms:
        tn = _norm(t)
        if not tn or tn in seen:
            continue
        seen.add(tn)
        out.append(t)

    return {"acronyms": found, "terms": out}


def _norm_for_match(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def glossary_bonus(hit: dict, terms: list[str]) -> int:
    txt = hit.get("chunk_text_clean") or hit.get("chunk_text") or ""
    topic = hit.get("topic_heuristic") or ""
    
    combined = _norm_for_match(txt + " " + topic)
    
    bonus = 0
    seen = set()
    for term in terms or []:
        tn = _norm_for_match(term)
        if not tn or tn in seen:
            continue
        seen.add(tn)
        if tn in combined:
            if tn in _norm_for_match(topic):
                bonus += 5 if " " in term else 2
            else:
                bonus += 3 if " " in term else 1
    return bonus


# ============================================================
# STOPWORDS Y MESES
# ============================================================

_STOP = {
    "cual", "cuál", "cuales", "cuáles", "como", "cómo", 
    "que", "qué", "quien", "quién", "donde", "dónde", "cuando", "cuándo",
    "son", "es", "fue", "fueron", "ser", "estar", "sido", "siendo",
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "para", "por", "en", "con", "sin", "sobre", "entre", "hacia",
    "y", "o", "ni", "pero", "sino", "aunque",
    "se", "le", "lo", "les", "nos", "me", "te",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
    "al", "a", "ha", "han", "hay", "he", "has",
    "muy", "mas", "más", "menos", "tan", "tanto", "mucho", "poco",
    "si", "no", "ya", "aun", "todavia", "tambien", "solo", "sólo",
}

_MONTHS = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
}


# ============================================================
# DETECCIÓN DE FRASES DEL GLOSARIO
# ============================================================

def _detect_glossary_phrases(query: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """
    Detecta frases del glosario en la query.
    """
    qn = _norm(query)
    
    found_phrases = []
    consumed_words: Set[str] = set()
    
    all_aliases = []
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = [acronym.lower()] + [a.lower() for a in obj.get("aliases", [])]
        for alias in aliases:
            alias_norm = _norm(alias)
            if alias_norm and " " in alias_norm:
                all_aliases.append((alias_norm, acronym.lower(), len(alias_norm)))
    
    all_aliases.sort(key=lambda x: x[2], reverse=True)
    
    for alias_norm, acronym, _ in all_aliases:
        if alias_norm in qn:
            alias_words = set(alias_norm.split())
            if not alias_words.intersection(consumed_words):
                found_phrases.append((alias_norm, acronym))
                consumed_words.update(alias_words)
    
    return found_phrases, consumed_words


def _detect_glossary_tokens(query: str, consumed: Set[str]) -> List[Tuple[str, str]]:
    """
    Detecta acrónimos/tokens del glosario en la query.
    """
    qn = _norm(query)
    words = set(qn.split())
    
    found_tokens = []
    
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        acr_norm = _norm(acronym)
        
        if acr_norm in words and acr_norm not in consumed:
            found_tokens.append((acr_norm, acronym.lower()))
            continue
        
        for alias in obj.get("aliases", []):
            alias_norm = _norm(alias)
            if " " not in alias_norm and alias_norm in words and alias_norm not in consumed:
                found_tokens.append((alias_norm, acronym.lower()))
                break
    
    return found_tokens


# ============================================================
# DETECCIÓN DE ENTIDADES NOMBRADAS (NUEVO)
# ============================================================

# Prefijos que indican nombre propio
_ENTITY_PREFIXES = {
    "banco", "banque", "bank",
    "grupo", "group",
    "empresa", "compañía", "company",
    "fondo", "fund", "fci",
    "corporación", "corporation", "corp",
}

# Palabras que NO deben iniciar una entidad
_NON_ENTITY_STARTERS = {
    "comparar", "cual", "cuales", "como", "que", "donde", "cuando",
    "mostrar", "buscar", "encontrar", "ver", "dame", "dime",
    "cuanto", "cuanta", "cuantos", "cuantas",
}

# Indicadores de entidades financieras conocidas
_ENTITY_INDICATORS = {
    "santander", "galicia", "bbva", "macro", "hsbc", "icbc", "patagonia",
    "supervielle", "hipotecario", "nacion", "nación", "provincia", "ciudad",
    "comafi", "itau", "itaú", "credicoop", "citi", "citibank",
    "mercado", "mercadolibre", "mercadopago",
    "uala", "ualá", "prex", "naranja", "brubank",
    "bind",
}


def _is_capitalized_word(word: str) -> bool:
    """Verifica si una palabra está capitalizada."""
    return bool(word) and word[0].isupper() and len(word) > 1


def _detect_named_entities(query: str) -> List[Tuple[str, Set[str]]]:
    """
    Detecta entidades nombradas en la query de forma dinámica.
    
    Estrategias:
    1. Patrón "Prefijo + Nombre": "Banco Santander", "Grupo Galicia"
    2. Palabras capitalizadas consecutivas: "Mercado Pago"
    3. Indicadores conocidos: palabras que típicamente son entidades
    
    Returns:
        Lista de (entidad_normalizada, {palabras_que_consume})
    """
    entities = []
    consumed_words: Set[str] = set()
    
    # Limpiar puntuación
    query_clean = re.sub(r'[^\w\s]', '', query)
    
    words_original = query_clean.split()
    words_norm = [_norm(w) for w in words_original]
    
    i = 0
    while i < len(words_norm):
        word = words_norm[i]
        word_orig = words_original[i]
        
        # Estrategia 1: Prefijo + Nombre (ej: "Banco Santander")
        if word in _ENTITY_PREFIXES and i + 1 < len(words_norm):
            entity_parts = [word]
            j = i + 1
            
            while j < len(words_norm):
                next_word = words_norm[j]
                next_orig = words_original[j]
                
                if _is_capitalized_word(next_orig) or next_word in _ENTITY_INDICATORS:
                    entity_parts.append(next_word)
                    j += 1
                else:
                    break
            
            if len(entity_parts) > 1:
                entity = " ".join(entity_parts)
                consumed = set(entity_parts)
                entities.append((entity, consumed))
                consumed_words.update(consumed)
                i = j
                continue
        
        # Estrategia 2: Palabras capitalizadas consecutivas
        if _is_capitalized_word(word_orig) and word not in _ENTITY_PREFIXES:
            if word in _NON_ENTITY_STARTERS:
                i += 1
                continue
                
            entity_parts = [word]
            j = i + 1
            
            while j < len(words_norm):
                next_orig = words_original[j]
                next_word = words_norm[j]
                
                if _is_capitalized_word(next_orig):
                    entity_parts.append(next_word)
                    j += 1
                else:
                    break
            
            if len(entity_parts) >= 2 or word in _ENTITY_INDICATORS:
                entity = " ".join(entity_parts)
                consumed = set(entity_parts)
                
                if not consumed.intersection(consumed_words):
                    entities.append((entity, consumed))
                    consumed_words.update(consumed)
                    i = j
                    continue
        
        # Estrategia 3: Indicadores conocidos como palabra suelta
        if word in _ENTITY_INDICATORS and word not in consumed_words:
            entities.append((word, {word}))
            consumed_words.add(word)
        
        i += 1
    
    return entities


# ============================================================
# EXTRACCIÓN DE ANCHORS (MEJORADA)
# ============================================================

def _extract_query_anchors(query: str) -> List[str]:
    """
    Extrae anchors de la query, respetando:
    1. Entidades nombradas (Banco Santander → frase completa)
    2. Frases del glosario (retorno sobre activos → frase + acrónimo)
    3. Palabras sueltas (no stopwords)
    
    Ejemplo:
        Query: "Cuál fueron los ingresos YTD del Banco Santander?"
        Anchors: ['banco santander', 'ytd', 'ingresos']
        
        (NO: ['banco', 'santander', 'ytd', 'ingresos'] donde 'banco' matchea BIND)
    """
    qn = _norm(query)
    
    anchors = []
    consumed_words: Set[str] = set()
    
    # 1) NUEVO: Detectar entidades nombradas PRIMERO
    named_entities = _detect_named_entities(query)
    for entity, entity_words in named_entities:
        anchors.append(entity)
        consumed_words.update(entity_words)
    
    # 2) Detectar frases del glosario (que no hayan sido consumidas)
    phrases, phrase_words = _detect_glossary_phrases(query)
    for phrase, acronym in phrases:
        phrase_word_set = set(phrase.split())
        if not phrase_word_set.intersection(consumed_words):
            anchors.append(phrase)
            anchors.append(acronym)
            consumed_words.update(phrase_word_set)
    
    # 3) Detectar tokens/acrónimos del glosario
    tokens = _detect_glossary_tokens(query, consumed_words)
    for token, acronym in tokens:
        if token not in consumed_words:
            anchors.append(acronym)
            consumed_words.add(token)
    
    # 4) Extraer palabras restantes
    words = re.findall(r"[a-z0-9]+", qn)
    
    for w in words:
        if w in consumed_words:
            continue
        if w in _STOP:
            continue
        if w in _MONTHS:
            continue
        if len(w) < 3:
            if not (w.isdigit() and len(w) == 4):
                continue
        
        anchors.append(w)
        consumed_words.add(w)
    
    # Deduplicar
    seen: Set[str] = set()
    unique = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    
    return unique


def _compute_anchor_score(hit: dict, anchors: List[str]) -> int:
    """
    Calcula el anchor score para un hit.
    
    Puntajes:
    - Entidad/frase completa (ej: "banco santander"): 5 puntos
    - Palabra suelta: 1 punto
    
    Esto asegura que un chunk con "Banco Santander" tenga MUCHO más score
    que uno con solo "BIND Banco" cuando se pregunta por Santander.
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
            # Es una palabra/acrónimo → bonus estándar
            pattern = rf"\b{re.escape(anchor)}\b"
            if re.search(pattern, combined):
                score += 1
    
    return score


# ============================================================
# PREPARACIÓN DE CANDIDATOS PARA RERANKING
# ============================================================

def prepare_rerank_candidates_glossary_aware(
    query: str,
    hits: List[Dict[str, Any]],
    max_input: int,
) -> List[Dict[str, Any]]:
    """
    Prepara candidatos para reranking, ordenando por:
    1. Anchor score (entidades + frases del glosario + keywords)
    2. Glossary bonus
    3. Base score
    """
    if not hits:
        return []
    
    # Extraer anchors (ahora incluye entidades nombradas)
    anchors = _extract_query_anchors(query)
    
    for h in hits:
        anchor_score = _compute_anchor_score(h, anchors) if anchors else 0
        h["_anchor_score"] = anchor_score

    # Glossary expansion
    gl = glossary_expand_terms(query)
    terms = gl.get("terms", []) or []

    if not terms:
        for h in hits:
            h["_glossary_bonus"] = 0
            h["_glossary_phrase_hits"] = 0
            h["_glossary_token_hits"] = 0
        
        ranked = sorted(
            hits,
            key=lambda h: (
                h.get("_anchor_score", 0),
                float(h.get("score") or h.get("similarity") or 0.0)
            ),
            reverse=True
        )
        return ranked[:max_input]

    # Separar frases vs tokens
    phrases = []
    tokens = []

    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        if " " in t:
            phrases.append(t)
        else:
            tl = t.lower()
            if len(tl) <= 5 or "%" in t or "." in t:
                tokens.append(t)

    def _dedupe(xs):
        out, seen = [], set()
        for x in xs:
            xn = _norm(x)
            if not xn or xn in seen:
                continue
            seen.add(xn)
            out.append(x)
        return out

    phrases = _dedupe(phrases)
    tokens = _dedupe(tokens)

    if not phrases and not tokens:
        for h in hits:
            h["_glossary_bonus"] = 0
        
        ranked = sorted(
            hits,
            key=lambda h: (
                h.get("_anchor_score", 0),
                float(h.get("score") or h.get("similarity") or 0.0)
            ),
            reverse=True
        )
        return ranked[:max_input]

    # Compilar patterns
    def _phrase_pattern(p: str) -> re.Pattern:
        pn = _norm(p)
        parts = [re.escape(w) for w in pn.split() if w]
        if not parts:
            return None
        rx = r"\b" + r"\b\s+\b".join(parts) + r"\b"
        return re.compile(rx, flags=re.IGNORECASE)

    phrase_pats = [pp for pp in (_phrase_pattern(p) for p in phrases) if pp is not None]

    token_pats = []
    for t in tokens:
        tn = _norm(t)
        if not tn:
            continue
        token_pats.append(re.compile(rf"\b{re.escape(tn)}\b", flags=re.IGNORECASE))
        token_pats.append(re.compile(rf"\b{re.escape(tn)}\s*%", flags=re.IGNORECASE))

    def anchor_matches(h: Dict[str, Any]) -> tuple[int, int]:
        txt = h.get("chunk_text_clean") or h.get("chunk_text") or ""
        topic = h.get("topic_heuristic") or ""
        combined = txt + " " + topic
        tn = _norm(combined)

        phrase_hits = sum(1 for pat in phrase_pats if pat.search(tn))
        token_hits = sum(1 for pat in token_pats if pat.search(tn))
        return phrase_hits, token_hits

    def base_score(h: Dict[str, Any]) -> float:
        return float(h.get("score") or h.get("similarity") or 0.0)

    for h in hits:
        ph, tk = anchor_matches(h)
        h["_glossary_phrase_hits"] = ph
        h["_glossary_token_hits"] = tk
        glossary_bonus_val = ph * 50 + tk * 5
        penalty = 30 if (ph == 0 and tk == 0) else 0
        h["_glossary_bonus"] = glossary_bonus_val - penalty

    # Ordenar: anchor_score > glossary_bonus > base_score
    ranked = sorted(
        hits,
        key=lambda h: (
            h.get("_anchor_score", 0) * 100,
            h.get("_glossary_bonus", 0),
            base_score(h)
        ),
        reverse=True
    )
    return ranked[:max_input]