import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from rag_lib.config import *
from rag_lib.text_utils import *
from rag_lib.llm import call_chat
from rag_lib.glossary_helper import glossary_snippet

# -------------------------
# CELL 7: Build context (with [S#] citations)
# -------------------------
def build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict[str, str]]]:
    blocks = []
    cites = []
    total = 0

    for i, h in enumerate(hits, start=1):
        sid = f"S{i}"
        path = h.get("path")
        page_num = h.get("page_num")
        topic = h.get("topic")
        text = (h.get("chunk_text_clean") or "").strip()

        header = f"[{sid}] path={path} | page_num={page_num} | topic={topic}\n"
        block = header + text + "\n"

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        cites.append({"sid": sid, "path": str(path), "page_num": str(page_num), "topic": str(topic)})
        total += len(block)

    return "\n---\n".join(blocks), cites

# -------------------------
# CELL 8: Extracción de evidencia (GENÉRICA)
# -------------------------
def extract_evidence(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extrae evidencia que soporte responder la pregunta (no está atada a 'previsiones').
    Devuelve JSON con:
      - answerable: si el contexto permite responder
      - key_points: lista de puntos con citas (por sid) que se pueden afirmar
      - evidence: lista de quotes literales (cortos) con sid/page/topic
      - missing: qué faltaría si no es respondible
    """
    context, _ = build_context(hits, max_chars=12000)

    system = (
        "Eres un extractor de evidencia para un sistema RAG. "
        "Tu tarea es identificar en el CONTEXTO los extractos que permiten responder la pregunta. "
        "Devuelve SOLO JSON válido, sin texto adicional."
    )

    user = (
        f"Pregunta: {query}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        "Devuelve JSON EXACTO con este esquema:\n"
        "{\n"
        '  "answerable": true/false,\n'
        '  "missing": ["si answerable=false, qué información falta"],\n'
        '  "key_points": [\n'
        '    {"claim": "hecho conciso que responde parte de la pregunta", "sids": ["S1","S3"]}\n'
        "  ],\n"
        '  "evidence": [\n'
        '    {"sid": "S2", "page_num": 7, "topic": "…", "quote": "extracto literal corto"}\n'
        "  ]\n"
        "}\n\n"
        "REGLAS:\n"
        "- NO inventes información. Usa SOLO lo presente en el CONTEXTO.\n"
        "- Cada quote debe ser literal y de máximo 25 palabras.\n"
        "- key_points deben ser verificables por las citas (sids).\n"
        "- Si la pregunta no contiene 'Empresas', 'Corporate' o 'Institucional','Baas' o 'Minorista', y aparece explícito, answerable=false.\n"
        "- Si la pregunta pide un segmento específico ('Empresas', 'Institucional', 'Corporate','Baas', 'Minorista')  y no aparece explícito, answerable=false.\n"
        "- Incluye máximo 12 key_points y máximo 12 evidence.\n"
    )

    raw = call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.0,
        max_tokens=800
    )
    parsed = safe_json_load(raw) or {"answerable": False, "missing": ["No se pudo parsear evidencia"], "key_points": [], "evidence": []}

    # --- Post-procesado defensivo: dedup + defaults ---
    parsed.setdefault("answerable", False)
    parsed.setdefault("missing", [])
    parsed.setdefault("key_points", [])
    parsed.setdefault("evidence", [])

    # Dedup evidence by (sid, quote)
    seen_ev = set()
    uniq_ev = []
    for e in parsed.get("evidence", []) or []:
        sid = (e.get("sid") or "").strip()
        quote = (e.get("quote") or "").strip()
        key = (sid, quote)
        if sid and quote and key not in seen_ev:
            uniq_ev.append(e)
            seen_ev.add(key)
    parsed["evidence"] = uniq_ev[:6]

    # Dedup key_points by claim text
    seen_kp = set()
    uniq_kp = []
    for kp in parsed.get("key_points", []) or []:
        claim = (kp.get("claim") or "").strip()
        if claim and claim not in seen_kp:
            uniq_kp.append(kp)
            seen_kp.add(claim)
    parsed["key_points"] = uniq_kp[:6]

    return parsed 

# -------------------------
# CELL 9: Respuesta con evidencia
# -------------------------
def answer_from_evidence(query: str, hits: List[Dict[str, Any]], evidence: Dict[str, Any]) -> str:
    """
    Respuesta directa y factual.
    - Modo 'comparativo/generico': resumen por elemento + desglose + lectura rápida.
    - Modo 'focalizado': 1 línea + bullets factuales.
    Sin recomendaciones ni próximos pasos.
    """
    context, _ = build_context(hits, max_chars=12000)

    ql = (query or "").lower()
    is_comparative = any(k in ql for k in [" vs ", " versus", "compar", "octubre", "septiembre", "moM", "mom"])
    # Heurística: si la pregunta menciona explícitamente un segmento, tratar como focalizada
    focused_segment = None
    for seg in ["empresas", "corporate", "institucional", "minorista", "banca", "bind"]:
        if seg in ql:
            focused_segment = seg
            break

    mode = "comparative" if (is_comparative and focused_segment is None) else "focused"

    system = (
        "Eres una base de conocimiento sobre reportes corporativos (RAG). \n"
        "Usa el glosario solo para interpretar términos, pero no lo cites como evidencia. Las afirmaciones sobre hechos deben salir del CONTEXTO/EVIDENCE.\n"
        "Responde de forma directa, detallada y 100% basada en evidencia. \n"
        "NO des recomendaciones, NO incluyas próximos pasos, NO inventes datos. \n"
        "Usa SOLO CONTEXTO y EVIDENCE.\n"
        "- Segmento y banca son sinónimos.\n"
        "- Si en la consulta no especifican tiempo, que la respuesta traiga el dato del último mes y el acumulado del año.\n"
    )

    # Plantilla para modo comparativo (más parecido a tu ejemplo)
    comparative_format = (
        "FORMATO OBLIGATORIO (MODO COMPARATIVO):\n"
        "1) ONE-LINER (1 frase): responde directo qué muestra el documento sobre la comparación solicitada.\n"
        "2) 'Resumen por elementos' (si el documento lo trae):\n"
        "   - 2 a 6 líneas compactas, una por segmento, con: Segmento: valor (unidad) + variación vs periodo [S#]\n"
        "3) 'Componentes principales por segmento' (solo si hay drivers en el texto):\n"
        "   - Para cada segmento relevante:\n"
        "     Segmento:\n"
        "       - Driver 1: dato literal/variación [S#]\n"
        "       - Driver 2: ... [S#]\n"
        "4) 'Lectura rápida' (1 frase): síntesis comparativa basada únicamente en las variaciones reportadas (sin opinión).\n"
        "REGLAS:\n"
        "- No asumas segmentos: SOLO incluye segmentos que aparecen explícitamente en el CONTEXTO.\n"
        "- No infieras drivers: solo los que estén escritos.\n"
        "- Cada línea o bullet debe terminar con al menos una cita [S#].\n"
        "- Si faltan números para un segmento, omite ese segmento (no inventes).\n"
    )

    # Plantilla para modo focalizado
    focused_format = (
        "FORMATO OBLIGATORIO (MODO FOCALIZADO):\n"
        "RESPUESTA DIRECTA (1 línea): contesta exactamente lo preguntado.\n"
        "DETALLE (2 a 8 bullets):\n"
        "- Solo hechos verificables (valores, variaciones, periodos, definiciones).\n"
        "- Cada bullet termina con citas [S#].\n"
        "REGLAS:\n"
        "- Si se pide un segmento específico y no aparece explícito, responde:\n"
        "  'No encuentro datos específicos de <segmento> en los extractos recuperados.'\n"
        "- No extrapoles entre segmentos.\n"
    )

    gloss = glossary_snippet(query)  # <-- 1) calcular texto del glosario (string)

    user = (
        f"Modo: {mode}\n"
        f"Pregunta: {query}\n\n"
        f"{gloss}\n"
        f"EVIDENCE (JSON):\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"{comparative_format if mode == 'comparative' else focused_format}"
        "\nREGLAS CRÍTICAS GENERALES:\n"
        "- Si evidence.answerable es false: responde exactamente 'No encuentro esa información en el documento.'\n"
        "  y luego 1-3 bullets con lo que falta (evidence.missing) y citas si aplican.\n"
        "- Si evidence.answerable es true: prioriza detallar usando el texto (tablas OCR, valores, variaciones).\n"
        "- NO uses lenguaje de recomendación ('deberías', 'conviene', 'haría').\n"
        "- NO inventes cifras: si la cifra no está, no la pongas.\n"
    )

    return call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.05,
        max_tokens=650
    )