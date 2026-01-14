import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
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

    # orden: mostrar primero los términos “más largos” (más específicos)
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
    t = _norm_for_match(txt)
    bonus = 0
    seen = set()
    for term in terms or []:
        tn = _norm_for_match(term)
        if not tn or tn in seen:
            continue
        seen.add(tn)
        if tn in t:
            # frase > palabra suelta
            bonus += 3 if " " in term else 1
    return bonus

# def prepare_rerank_candidates_glossary_aware(
#     query: str,
#     hits: List[Dict[str, Any]],
#     max_input: int,
# ) -> List[Dict[str, Any]]:
#     # from rag_lib.business_glossary import glossary_expand_terms, _norm

#     gl = glossary_expand_terms(query)
#     terms = gl.get("terms", []) or []
#     if not terms or not hits:
#         return hits[:max_input]

#     def bonus(h: Dict[str, Any]) -> int:
#         txt = h.get("chunk_text_clean") or h.get("chunk_text") or ""
#         tn = _norm(txt)
#         b = 0
#         for t in terms:
#             tt = _norm(t)
#             if not tt:
#                 continue
#             if tt in tn:
#                 b += 3 if " " in t else 1
#         return b

#     def base_score(h: Dict[str, Any]) -> float:
#         # ajustá según tu schema real
#         return float(h.get("score") or h.get("similarity") or 0.0)

#     # anotamos bonus para debug si querés
#     for h in hits:
#         h["_glossary_bonus"] = bonus(h)

#     ranked = sorted(hits, key=lambda h: (h["_glossary_bonus"], base_score(h)), reverse=True)
#     return ranked[:max_input]

def prepare_rerank_candidates_glossary_aware(
    query: str,
    hits: List[Dict[str, Any]],
    max_input: int,
) -> List[Dict[str, Any]]:

    gl = glossary_expand_terms(query)
    terms = gl.get("terms", []) or []
    if not terms or not hits:
        return hits[:max_input]

    # --- 1) separar anchors: frases vs tokens (acrónimos)
    phrases = []
    tokens = []

    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        if " " in t:
            phrases.append(t)
        else:
            # solo tokens "fuertes": acrónimos cortos o con %/puntos
            tl = t.lower()
            if len(tl) <= 5 or "%" in t or "." in t:
                tokens.append(t)

    # dedupe por normalización
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
    tokens  = _dedupe(tokens)

    # Si no hay anchors fuertes, no hacemos nada
    if not phrases and not tokens:
        return hits[:max_input]

    # --- 2) compilar regex estrictos
    def _phrase_pattern(p: str) -> re.Pattern:
        # normalizamos y escapamos, y exigimos límites de palabra
        # "retorno sobre patrimonio" -> r"\bretorno\b\s+\bsobre\b\s+\bpatrimonio\b"
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
        # \broe\b match exacto
        token_pats.append(re.compile(rf"\b{re.escape(tn)}\b", flags=re.IGNORECASE))
        # opcional: permitir "roe%" si no lo pusiste explícito como alias
        token_pats.append(re.compile(rf"\b{re.escape(tn)}\s*%", flags=re.IGNORECASE))

    # --- 3) scoring: frases pesan mucho más que tokens
    def anchor_matches(h: Dict[str, Any]) -> tuple[int, int]:
        txt = h.get("chunk_text_clean") or h.get("chunk_text") or ""
        tn = _norm(txt)

        phrase_hits = sum(1 for pat in phrase_pats if pat.search(tn))
        token_hits  = sum(1 for pat in token_pats  if pat.search(tn))
        return phrase_hits, token_hits

    def base_score(h: Dict[str, Any]) -> float:
        return float(h.get("score") or h.get("similarity") or 0.0)

    for h in hits:
        ph, tk = anchor_matches(h)
        h["_glossary_phrase_hits"] = ph
        h["_glossary_token_hits"] = tk

        # frases mandan; tokens ayudan pero no ganan solos
        bonus = ph * 50 + tk * 5

        # penalización fuerte si no hay frases NI tokens (evita "patrimonio neto")
        penalty = 30 if (ph == 0 and tk == 0) else 0

        h["_glossary_bonus"] = bonus - penalty

    ranked = sorted(
        hits,
        key=lambda h: (h["_glossary_bonus"], base_score(h)),
        reverse=True
    )
    return ranked[:max_input]
