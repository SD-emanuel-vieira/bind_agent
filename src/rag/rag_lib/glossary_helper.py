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
        # siempre incluimos el propio acrónimo como alias
        candidates = [acronym] + list(aliases)
        for a in candidates:
            if _norm(a) in q:
                found.append(acronym)
                break

    # 2) fuzzy fallback si no encontró nada (o encontró muy poco)
    if not found:
        tokens = q.split()
        # compara cada alias contra ventanas del query
        for acronym, obj in BUSINESS_GLOSSARY_V2.items():
            best = 0.0
            for a in [acronym] + list(obj.get("aliases", [])):
                an = _norm(a)
                if not an:
                    continue
                # quick check: si el alias tiene muchas palabras, intenta match con el query entero
                best = max(best, SequenceMatcher(None, an, q).ratio())
                # y también contra tokens (para typos de 1 palabra)
                if " " not in an:
                    for t in tokens:
                        best = max(best, SequenceMatcher(None, an, t).ratio())
            if best >= fuzzy_threshold:
                found.append(acronym)

    # orden: mostrar primero los términos "más largos" (más específicos)
    found = list(dict.fromkeys(found))  # dedupe preservando orden
    found = found[:max_terms]
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
        "acronyms": ["ROA", ...],            # canónicos detectados
        "terms": ["ROA", "roa", "retorno sobre activos", ...]  # para expansión
      }
    Reusa la misma lógica de matching que glossary_snippet().
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

    # Terms de expansión: canónico + aliases
    terms = []
    for ac in found:
        terms.append(ac)
        terms.extend(BUSINESS_GLOSSARY_V2[ac].get("aliases", []))

    # dedupe (preserva orden) + filtra vacíos
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
# NUEVO: Extracción de anchors que respeta frases del glosario
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


def _detect_glossary_phrases(query: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """
    Detecta frases del glosario en la query.
    
    Returns:
        - Lista de (frase_normalizada, acrónimo) encontrados
        - Set de palabras "consumidas" por las frases
    """
    qn = _norm(query)
    
    found_phrases = []
    consumed_words: Set[str] = set()
    
    # Recopilar todas las frases (aliases con espacios)
    all_aliases = []
    for acronym, obj in BUSINESS_GLOSSARY_V2.items():
        aliases = [acronym.lower()] + [a.lower() for a in obj.get("aliases", [])]
        for alias in aliases:
            alias_norm = _norm(alias)
            if alias_norm and " " in alias_norm:
                all_aliases.append((alias_norm, acronym.lower(), len(alias_norm)))
    
    # Ordenar por longitud descendente (frases más largas primero)
    all_aliases.sort(key=lambda x: x[2], reverse=True)
    
    # Buscar frases en la query
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


def _extract_query_anchors(query: str) -> List[str]:
    """
    Extrae anchors de la query, respetando frases del glosario.
    
    Proceso:
    1. Detectar frases del glosario → usar frase completa + acrónimo
    2. Detectar tokens/acrónimos del glosario → usar acrónimo
    3. Extraer palabras restantes (no stopwords, no ya consumidas)
    
    Ejemplo:
        Query: "Cual es el retorno sobre activos para octubre de 2025?"
        Anchors: ['retorno sobre activos', 'roa', '2025']
    """
    qn = _norm(query)
    
    anchors = []
    consumed_words: Set[str] = set()
    
    # 1) Detectar frases del glosario
    phrases, phrase_words = _detect_glossary_phrases(query)
    for phrase, acronym in phrases:
        anchors.append(phrase)
        anchors.append(acronym)
        consumed_words.update(phrase.split())
    
    # 2) Detectar tokens/acrónimos del glosario
    tokens = _detect_glossary_tokens(query, consumed_words)
    for token, acronym in tokens:
        if token not in consumed_words:
            anchors.append(acronym)
            consumed_words.add(token)
    
    # 3) Extraer palabras restantes
    words = re.findall(r"[a-z0-9]+", qn)
    
    for w in words:
        if w in consumed_words:
            continue
        if w in _STOP:
            continue
        if w in _MONTHS:
            continue
        if len(w) < 4:
            if not (w.isdigit() and len(w) == 4):
                continue
        
        anchors.append(w)
        consumed_words.add(w)
    
    # Deduplicar preservando orden
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
    
    - Frases del glosario: bonus mayor (3 puntos)
    - Acrónimos/palabras: bonus estándar (1 punto)
    """
    txt = _norm(hit.get("chunk_text_clean") or hit.get("chunk_text") or "")
    topic = _norm(hit.get("topic_heuristic") or "")
    combined = txt + " " + topic
    
    score = 0
    
    for anchor in anchors:
        if " " in anchor:
            # Es una frase → bonus alto si matchea completa
            if anchor in combined:
                score += 3
        else:
            # Es una palabra/acrónimo → usar word boundary
            pattern = rf"\b{re.escape(anchor)}\b"
            if re.search(pattern, combined):
                score += 1
    
    return score


def prepare_rerank_candidates_glossary_aware(
    query: str,
    hits: List[Dict[str, Any]],
    max_input: int,
) -> List[Dict[str, Any]]:
    """
    Prepara candidatos para reranking, ordenando por:
    1. Anchor score (frases/términos del glosario + keywords)
    2. Glossary bonus
    3. Base score (similitud del vector search)
    """
    if not hits:
        return []
    
    # Extraer anchors (ahora respeta frases del glosario)
    anchors = _extract_query_anchors(query)
    
    for h in hits:
        anchor_score = _compute_anchor_score(h, anchors) if anchors else 0
        h["_anchor_score"] = anchor_score

    # Glossary expansion
    gl = glossary_expand_terms(query)
    terms = gl.get("terms", []) or []

    # Si no hay términos del glosario, ordenar solo por anchor_score + base_score
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