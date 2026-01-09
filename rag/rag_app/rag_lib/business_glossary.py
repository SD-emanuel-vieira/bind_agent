import re
import unicodedata
from difflib import SequenceMatcher

# -----------------------------
# Business glossary (stable terms)
# -----------------------------
BUSINESS_GLOSSARY_V2 = {
    "CdG":  {"desc": "Control de Gestión", "aliases": []},
    "TC":   {"desc": "Tarjeta de Crédito", "aliases": []},
    "TC CCL": {"desc": "Tarjeta de Crédito Contado con Liquidación", "aliases": []},
    "i.a.": {"desc": "Interanual (año contra año).", "aliases": []},
    "BADLAR": {"desc": "Tasa BADLAR: tasa de referencia en Argentina para depósitos a plazo fijo mayoristas (típicamente > ARS 1M) a ~30–35 días.", "aliases": []},
    "Bdg": {"desc": "Budget / Presupuesto / Presupuestado", "aliases": []},
    "FYF": {"desc": "Full Year Forecast / Pronóstico de Año Completo", "aliases": []},
    "TEM": {"desc": "Tasa Efectiva Mensual", "aliases": []},
    "R":   {"desc": "Real", "aliases": []},
    "E":   {"desc": "Estimado", "aliases": []},
    "IPC": {"desc": "Índice de Precios al Consumidor", "aliases": []},
    "Lefi 1d": {"desc": "Letras de Liquidez (LEFI) con liquidación diaria.", "aliases": []},
    "ROA": {"desc": "Return on Assets / Retorno sobre Activos", "aliases": []},
    "ROE": {"desc": "Return on Equity / Retorno sobre Patrimonio", "aliases": []},
    "Com": {"desc": "Comisiones", "aliases": []},
    "ARS": {"desc": "Pesos Argentinos", "aliases": []},
    "MM":  {"desc": "Millones", "aliases": []},
    "P&L": {"desc": "Profit and Loss / Ganancias y Pérdidas", "aliases": []},
    "YTD": {"desc": "Year to Date / Año hasta la fecha", "aliases": []},
    "Xsell": {"desc": "Cross Sell", "aliases": []},
    "Bi":  {"desc": "Billones", "aliases": []},
    "MF":  {"desc": "Margen Financiero", "aliases": ["márgen financiero", "margen financiero"]},
    "FX":  {"desc": "Forex / Foreign Exchange", "aliases": []},
    "AxI": {"desc": "Ajustado por Inflación", "aliases": []},
    "TNA": {"desc": "Tasa Nominal Anual", "aliases": []},
    "KPI": {"desc": "Key Performance Indicator", "aliases": []},
    "IIBB": {"desc": "Ingresos Brutos", "aliases": []},
    "HC":  {"desc": "Head Count", "aliases": []},
    "PEA": {"desc": "Personas equivalentes a tiempo completo (FTE).", "aliases": []},
    "MoM": {"desc": "Month over Month / Mes a Mes", "aliases": []},
    "SGR": {"desc": "Sociedad de Garantía Recíproca", "aliases": []},
    "RECPAM": {"desc": "Resultado por Exposición a los Cambios en el Poder Adquisitivo de la Moneda", "aliases": []},
    "NDF": {"desc": "Non-Deliverable Forward: derivado para fijar un tipo de cambio futuro entre dos monedas (sin entrega física).", "aliases": []},
    "CC":  {"desc": "Casa Central", "aliases": []},
    "MAV": {"desc": "Mercado Argentino de Valores", "aliases": []},
    "Tx":  {"desc": "Transacciones", "aliases": []},
    "Comex": {"desc": "Comercio Exterior", "aliases": []},
    "NPL": {"desc": "Non-Performing Loans / Préstamos morosos", "aliases": []},
    "CaR": {"desc": "Ratio de Adecuación de Capital", "aliases": []},
    "RAW": {"desc": "Activos ponderados por riesgo", "aliases": []},
    "BaaS": {"desc": "Banking as a Service: BIND ofrece infraestructura bancaria para integrar servicios en apps de terceros no bancarios.", "aliases": []},
}

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
