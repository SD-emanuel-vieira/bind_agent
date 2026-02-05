"""
evidence_handling.py - Construcción de contexto, extracción de evidencia y generación de respuestas

ACTUALIZADO v2:
- NUEVO: Validación de entidades críticas antes de generar respuesta
- Si la query pregunta por una entidad (ej: "cliente Grimoldi") que NO aparece
  en ningún chunk, el sistema responde "No encuentro información sobre X"
- Esto previene asociaciones espurias donde el LLM encuentra una métrica
  y la asocia incorrectamente a una entidad que no existe en las evidencias

NOTA SOBRE PUNTO DE INTEGRACIÓN:
- Este archivo incluye validación en extract_evidence() como fallback
- PERO la mejor práctica es validar TEMPRANO en rag_core.py sobre `candidates`
  (ANTES del rerank con LLM) para hacer early exit y ahorrar llamadas al LLM
- Ver SOLUCION_ENTIDADES.md para la integración recomendada en rag_core.py
"""

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
from rag_lib.business_rules import get_rules_snippet
from rag_lib.entity_validation import (
    validate_entities_in_chunks,
    get_entity_validation_prompt_snippet,
    extract_critical_entities
)


# -------------------------
# Build context (with [S#] citations) - MEJORADO
# -------------------------
def build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict[str, str]]]:
    """
    Construye el contexto para el LLM.
    
    MEJORADO: El topic se presenta de forma más prominente para que el LLM
    entienda qué tipo de datos contiene cada chunk (especialmente tablas sin headers).
    """
    blocks = []
    cites = []
    total = 0

    for i, h in enumerate(hits, start=1):
        sid = f"S{i}"
        path = h.get("path")
        page_num = h.get("page_num")
        
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
        
        # MEJORADO: Para tablas, el topic es CRÍTICO para entender qué son los datos
        if chunk_type == "table":
            header = (
                f"[{sid}] TABLA | Contenido: {topic}\n"
                f"Archivo: {path} | Página: {page_num}\n"
                f"NOTA: Los valores en esta tabla corresponden a '{topic}'. "
                f"Usa el título/topic para interpretar qué representan los números.\n"
            )
        else:
            header = f"[{sid}] path={path} | page_num={page_num} | topic={topic}\n"
        
        block = header + text + "\n"

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        cites.append({
            "sid": sid, 
            "path": str(path), 
            "page_num": str(page_num), 
            "topic": str(topic)
        })
        total += len(block)

    return "\n---\n".join(blocks), cites


# -------------------------
# Extracción de evidencia - MEJORADO con validación de entidades
# -------------------------
def extract_evidence(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extrae evidencia que soporte responder la pregunta.
    
    MEJORADO v2:
    - Valida que las entidades críticas de la query aparezcan en los chunks
    - Si no aparecen, marca answerable=false inmediatamente
    - Instrucciones mejoradas para el LLM
    """
    # ============================================================
    # PASO 0: Validar entidades críticas ANTES de llamar al LLM
    # ============================================================
    entity_validation = validate_entities_in_chunks(query, hits)
    
    # Si hay entidades críticas que no aparecen en ningún chunk,
    # retornar directamente sin llamar al LLM
    if not entity_validation["is_valid"]:
        missing_str = ", ".join(entity_validation["missing_entities"])
        return {
            "answerable": False,
            "missing": [f"No se encontró información sobre '{missing_str}' en las evidencias disponibles"],
            "key_points": [],
            "evidence": [],
            "_entity_validation": entity_validation,
            "_early_exit": True  # Flag para debugging
        }
    
    # ============================================================
    # PASO 1: Construir contexto y llamar al LLM
    # ============================================================
    context, _ = build_context(hits, max_chars=12000)
    
    # Generar snippet de validación de entidades para el prompt
    entity_prompt_snippet = get_entity_validation_prompt_snippet(entity_validation)

    system = (
        "Eres un extractor de evidencia para un sistema RAG financiero. "
        "Tu tarea es identificar en el CONTEXTO los extractos que permiten responder la pregunta. "
        "Devuelve SOLO JSON válido, sin texto adicional."
    )

    user = (
        f"Pregunta: {query}\n\n"
        f"{entity_prompt_snippet}\n"
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
        "5. ⚠️ REGLA DE ENTIDAD OBLIGATORIA: Si la pregunta menciona una entidad específica (ej: un cliente, empresa, banco),\n"
        "   verifica que ESA ENTIDAD EXACTA aparezca en el chunk. Si NO aparece, answerable=false.\n"
        "   - Ejemplo: Si preguntan por 'Grimoldi' pero ningún chunk menciona 'Grimoldi', answerable=false.\n"
        "   - NO asumas que un valor genérico corresponde a una entidad que NO aparece en el texto.\n"
        "6. Incluye máximo 12 key_points y máximo 12 evidence.\n"
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

    # ============================================================
    # PASO 2: Post-validación de entidades en claims
    # ============================================================
    # Verificar que los key_points no mencionen entidades que no están en las evidencias
    if entity_validation["critical_entities"]:
        parsed = _validate_claims_against_entities(parsed, entity_validation, hits)

    # Agregar metadata de validación
    parsed["_entity_validation"] = entity_validation

    return parsed 


def _validate_claims_against_entities(
    parsed: Dict[str, Any], 
    entity_validation: Dict[str, Any],
    hits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Valida que los claims no asocien valores a entidades que no están en las evidencias.
    
    Si un claim menciona una entidad que no está en ningún chunk, lo elimina.
    """
    critical_entities = entity_validation.get("critical_entities", [])
    entity_coverage = entity_validation.get("entity_coverage", {})
    
    # Filtrar key_points que mencionen entidades sin cobertura
    valid_key_points = []
    for kp in parsed.get("key_points", []):
        claim = (kp.get("claim") or "").lower()
        
        # Verificar si el claim menciona alguna entidad sin cobertura
        mentions_missing_entity = False
        for entity in critical_entities:
            if entity in claim and entity_coverage.get(entity, 0) == 0:
                mentions_missing_entity = True
                break
        
        if not mentions_missing_entity:
            valid_key_points.append(kp)
    
    # Si se filtraron TODOS los key_points, marcar como no respondible
    if parsed.get("key_points") and not valid_key_points:
        parsed["answerable"] = False
        missing_entities = [e for e in critical_entities if entity_coverage.get(e, 0) == 0]
        if missing_entities:
            parsed["missing"] = [f"No se encontró información sobre '{', '.join(missing_entities)}' en las evidencias"]
    
    parsed["key_points"] = valid_key_points
    
    return parsed


# -------------------------
# Respuesta con evidencia - MEJORADO
# -------------------------
def answer_from_evidence(query: str, hits: List[Dict[str, Any]], evidence: Dict[str, Any]) -> str:
    """
    Respuesta directa y factual.
    
    MEJORADO v2:
    - Si evidence.answerable=false por entidad faltante, da respuesta clara
    - Incluye validación de entidades en el prompt
    """
    # ============================================================
    # CASO ESPECIAL: Entidad no encontrada
    # ============================================================
    entity_validation = evidence.get("_entity_validation", {})
    missing_entities = entity_validation.get("missing_entities", [])
    
    if not evidence.get("answerable", True) and missing_entities:
        missing_str = ", ".join(missing_entities)
        return (
            f"No encuentro información sobre '{missing_str}' en las evidencias disponibles.\n\n"
            f"Las evidencias recuperadas contienen información general del banco BIND "
            f"(segmentos como Empresas, Corporate, Institucional), pero no mencionan "
            f"específicamente a '{missing_str}'.\n\n"
            f"Si '{missing_str}' es un cliente específico, es posible que:\n"
            f"- No exista información sobre este cliente en los documentos indexados\n"
            f"- El nombre esté escrito de forma diferente en los documentos\n"
            f"- La información sea confidencial y no esté en los reportes corporativos"
        )

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

    # Generar snippet de validación de entidades
    entity_prompt_snippet = get_entity_validation_prompt_snippet(entity_validation)

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
        "REGLA CRÍTICA DE ENTIDADES:\n"
        "- Si la pregunta menciona una entidad específica (cliente, empresa, banco),\n"
        "  VERIFICA que esa entidad aparezca EXPLÍCITAMENTE en el contexto.\n"
        "- NO asocies valores a entidades que NO aparecen en las evidencias.\n"
        "- Si la entidad no aparece, responde: 'No encuentro información sobre [entidad]'.\n"
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
        f"{entity_prompt_snippet}\n"
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
        "- ⚠️ Si la pregunta es sobre una entidad específica que NO aparece en el contexto,\n"
        "  responde: 'No encuentro información sobre [entidad] en las evidencias disponibles.'\n"
    )

    return call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.05,
        max_tokens=700
    )
