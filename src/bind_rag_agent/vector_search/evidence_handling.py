import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from bind_rag_agent.config import *
from bind_rag_agent.text_utils import *
from bind_rag_agent.vector_search.llm import call_chat
from bind_rag_agent.vector_search.glossary_helper import glossary_snippet
from bind_rag_agent.vector_search.business_rules import get_rules_snippet


# -------------------------
# Clasificación de fuente por path
# -------------------------
_SOURCE_LABELS = [
    (r"directorio", "Directorio"),
    (r"cdg",        "CdG"),
]

def _classify_source(path: str) -> str:
    """Clasifica el tipo de fuente por el path del archivo."""
    p = (path or "").lower()
    for pattern, label in _SOURCE_LABELS:
        if re.search(pattern, p):
            return label
    return "Otro"


# -------------------------
# Build context (with [S#] citations) - MEJORADO
# -------------------------
def build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict[str, str]]]:
    """
    Construye el contexto para el LLM.
    
    MEJORADO: 
    - El topic se presenta de forma prominente para tablas sin headers.
    - Cada hit se etiqueta con su fuente (Directorio, CdG, etc.) y si es
      la fuente primaria (primer hit de ese tipo de fuente).
    """
    blocks = []
    cites = []
    total = 0
    
    # Determinar la fuente primaria (la del hit #1, que ya es el mejor rankeado)
    primary_source = _classify_source(hits[0].get("path", "")) if hits else ""

    for i, h in enumerate(hits, start=1):
        sid = f"S{i}"
        path = h.get("path")
        page_num = h.get("page_num")
        source = _classify_source(path)
        is_primary = (source == primary_source and i == 1) or (source == primary_source)
        source_tag = f"FUENTE: {source}" + (" (PRIMARIA)" if source == primary_source else " (SECUNDARIA)")
        
        # Obtener el mejor topic disponible
        topic = (
            h.get("topic_llm") or 
            h.get("topic_content") or 
            h.get("topic_heuristic") or 
            h.get("topic") or 
            ""
        )
        
        text = (h.get("chunk_text_clean") or "").strip()
        chunk_type = h.get("chunk_type") or "text"
        
        if chunk_type == "table":
            header = (
                f"[{sid}] TABLA | Contenido: {topic}\n"
                f"Archivo: {path} | Página: {page_num} | {source_tag}\n"
                f"NOTA: Los valores en esta tabla corresponden a '{topic}'. "
                f"Usa el título/topic para interpretar qué representan los números.\n"
            )
        else:
            header = f"[{sid}] path={path} | page_num={page_num} | topic={topic} | {source_tag}\n"
        
        block = header + text + "\n"

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        cites.append({
            "sid": sid, 
            "path": str(path), 
            "page_num": str(page_num), 
            "topic": str(topic),
            "source": source,
        })
        total += len(block)

    return "\n---\n".join(blocks), cites


# -------------------------
# Extracción de evidencia - MEJORADO
# -------------------------
def extract_evidence(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    if not hits:
        return {"answerable": False, "missing": ["No se encontraron documentos"]}
    
    context, _ = build_context(hits, max_chars=12000)

    system = (
        "Eres un extractor de evidencia para un sistema RAG financiero. "
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
        "REGLAS CRÍTICAS:\n"
        "1. TABLAS SIN HEADERS: Si un chunk es una TABLA, usa el 'topic' o 'Contenido:' para entender QUÉ representan los valores.\n"
        "   - Ejemplo: Si topic='Resultados Integrales (YTD julio 2025)' y ves 'Banco Santander | 1,234', \n"
        "     entonces 1,234 es el Resultado Integral YTD julio 2025 de Banco Santander.\n"
        "2. ASOCIAR ENTIDAD CON VALOR: Si la pregunta es sobre 'Banco Santander' y el chunk tiene una fila con 'Banco Santander | X',\n"
        "   ese valor X corresponde a Banco Santander para la métrica indicada en el topic.\n"
        "3. NO inventes información. Usa SOLO lo presente en el CONTEXTO.\n"
        "4. Cada quote debe ser literal y de máximo 25 palabras.\n"
        "5. Si la pregunta menciona una entidad específica, verifica que ESA entidad aparezca en el chunk.\n"
        "6. Incluye máximo 12 key_points y máximo 12 evidence.\n"
        "7. DEDUP POR FUENTE: Si dos chunks de fuentes distintas (Directorio vs CdG) reportan la misma métrica "
        "con valores diferentes, usa SOLO el dato de la FUENTE PRIMARIA e ignora la SECUNDARIA. "
        "No reportes el mismo dato dos veces con valores distintos.\n"
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
# Respuesta con evidencia - MEJORADO
# -------------------------
def answer_from_evidence(query: str, hits: List[Dict[str, Any]], evidence: Dict[str, Any]) -> str:
    """
    Respuesta directa y factual.
    
    MEJORADO: Instrucciones para interpretar tablas usando el topic.
    """
    context, _ = build_context(hits, max_chars=12000)

    ql = (query or "").lower()
    is_comparative = any(k in ql for k in [" vs ", " versus", "compar", "octubre", "septiembre", "moM", "mom"])
    
    focused_segment = None
    for seg in ["empresas", "corporate", "institucional", "minorista", "banca", "bind"]:
        if seg in ql:
            focused_segment = seg
            break

    mode = "comparative" if (is_comparative and focused_segment is None) else "focused"

    # Obtener contextos adicionales
    evidence_text = " ".join(
        (h.get("chunk_text_clean") or h.get("chunk_text") or "") 
        for h in (hits or [])
    )
    business_rules = get_rules_snippet(query, evidence_text)

    system = (
        "Eres una base de conocimiento sobre reportes corporativos (RAG). \n"
        "Responde de forma directa, detallada y 100% basada en evidencia. \n"
        "NO des recomendaciones, NO incluyas próximos pasos, NO inventes datos. \n"
        "Usa SOLO CONTEXTO y EVIDENCE.\n\n"
        "REGLA CRÍTICA PARA TABLAS:\n"
        "- Muchas tablas NO tienen headers de columna explícitos.\n"
        "- El 'topic' o 'Contenido:' del chunk indica QUÉ métrica representan los valores.\n"
        "- Si topic='Resultados Integrales (YTD julio 2025)' y ves 'Banco Santander | 500',\n"
        "  entonces 500 es el Resultado Integral YTD julio 2025 de Banco Santander.\n"
        "- SIEMPRE interpreta los valores usando el topic como contexto semántico.\n\n"
        "REGLA CRÍTICA DE DESAMBIGUACIÓN DE MÉTRICAS:\n"
        "- 'Resultado neto' SIN calificador adicional = 'Resultado de Gestión Neto AxI' (NO el Resultado Contable).\n"
        "- Solo responder con 'Resultado Contable' si el usuario pregunta explícitamente por 'resultado contable' o 'resultado neto contable'.\n"
        "- Consulta las REGLAS DE NEGOCIO inyectadas para ver la jerarquía de resultados.\n"
    )

    comparative_format = (
        "FORMATO OBLIGATORIO (MODO COMPARATIVO):\n"
        "1) ONE-LINER (1 frase): responde directo qué muestra el documento.\n"
        "2) 'Resumen por elementos':\n"
        "   - Entidad: valor (interpretado según el topic del chunk) [S#]\n"
        "3) 'Lectura rápida' (1 frase): síntesis.\n"
    )

    focused_format = (
        "FORMATO OBLIGATORIO (MODO FOCALIZADO):\n"
        "RESPUESTA DIRECTA (1 línea): contesta exactamente lo preguntado.\n"
        "- Usa el topic del chunk para entender qué representa el valor.\n"
        "DETALLE (2 a 8 bullets):\n"
        "- Cada bullet termina con citas [S#].\n"
    )

    gloss = glossary_snippet(query)

    user = (
        f"Modo: {mode}\n"
        f"Pregunta: {query}\n\n"
        f"{gloss}\n\n"
        f"{business_rules}\n\n"
        f"EVIDENCE (JSON):\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"{comparative_format if mode == 'comparative' else focused_format}"
        "\nREGLAS CRÍTICAS:\n"
        "- Si evidence.answerable es false: responde 'No encuentro esa información.'\n"
        "- Si evidence.answerable es true: extrae el valor de la entidad preguntada.\n"
        "- IMPORTANTE: El topic del chunk te dice qué métrica es. Úsalo para interpretar.\n"
        "- NO inventes cifras: si no está, no la pongas.\n"
        "- DEDUP POR FUENTE: Cada chunk está etiquetado como FUENTE PRIMARIA o SECUNDARIA. "
        "Si dos fuentes distintas reportan la misma métrica con valores diferentes, "
        "usa SOLO el valor de la FUENTE PRIMARIA. No menciones el dato duplicado de la fuente secundaria.\n"
    )

    return call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.05,
        max_tokens=700
    )