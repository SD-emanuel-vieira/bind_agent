import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
from bind_rag_agent.config import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_MULTI_SEGMENT,
    MAX_CONTEXT_CHARS_NORMAL,
    MIN_CHARS_PER_CHUNK,
    MAX_TOKENS_EVIDENCE_MULTI,
    MAX_TOKENS_EVIDENCE_NORMAL,
    MAX_TOKENS_ANSWER_MULTI,
    MAX_TOKENS_ANSWER_NORMAL,
    TEMPERATURE_EVIDENCE,
    TEMPERATURE_ANSWER_GENERATION,
    EVIDENCE_LIMIT_MULTI,
    EVIDENCE_LIMIT_NORMAL,
    KEY_POINTS_LIMIT_MULTI,
    KEY_POINTS_LIMIT_NORMAL,
    LLM_ENDPOINT,
    SEGMENT_ALIASES,
    SEGMENTS,
)
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
def build_context(
    hits: List[Dict[str, Any]], 
    max_chars: int = MAX_CONTEXT_CHARS,
    max_chars_per_chunk: Optional[int] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Construye el contexto para el LLM.
    
    MEJORADO: 
    - El topic se presenta de forma prominente para tablas sin headers.
    - Cada hit se etiqueta con su fuente (Directorio, CdG, etc.).
    
    NUEVO parámetro max_chars_per_chunk:
    - Cuando es None: comportamiento original (chunks completos, trunca si se pasa el total).
    - Cuando tiene valor: cada chunk individual se trunca a ese límite antes de agregarse.
      Esto garantiza que TODOS los chunks (o la mayoría) quepan en el contexto,
      sacrificando detalle de cada tabla en vez de perder segmentos enteros.
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
        context_text = (h.get("context_text") or "").strip()
        
        if chunk_type == "table":
            header = (
                f"[{sid}] TABLA | Contenido: {topic}\n"
                f"Archivo: {path} | Página: {page_num} | {source_tag}\n"
            )
            if context_text:
                header += (
                    f"CONTEXTO DE LA MÉTRICA: {context_text}\n"
                    f"NOTA: Los valores en esta tabla corresponden ESPECÍFICAMENTE a "
                    f"'{context_text}'. NO confundir con otras métricas (ej: resultado operativo ≠ resultado de gestión).\n"
                )
            else:
                header += (
                    f"NOTA: Los valores en esta tabla corresponden a '{topic}'. "
                    f"Usa el título/topic para interpretar qué representan los números.\n"
                )
        else:
            header = f"[{sid}] path={path} | page_num={page_num} | topic={topic} | {source_tag}\n"
            if context_text:
                header += f"CONTEXTO: {context_text}\n"
        
        # Truncar texto del chunk si hay límite por chunk
        if max_chars_per_chunk and len(text) > max_chars_per_chunk:
            # Reservar espacio para el header y un aviso de truncado
            text_budget = max_chars_per_chunk - len(header) - 50
            if text_budget > 200:
                text = text[:text_budget] + "\n[... tabla truncada por espacio ...]"
        
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
# Extracción de evidencia - MEJORADO con soporte multi-segmento
# -------------------------
def extract_evidence(
    query: str, 
    hits: List[Dict[str, Any]], 
    multi_segment_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extrae evidencia estructurada de los hits.
    
    NUEVO: Cuando multi_segment_info está presente, ajusta el prompt para
    requerir cobertura de TODOS los segmentos y aumenta los límites.
    
    Para queries normales (multi_segment_info=None), el comportamiento
    es idéntico al anterior.
    """
    if not hits:
        return {"answerable": False, "missing": ["No se encontraron documentos"]}
    
    # Para multi-segmento, dar más espacio y limitar cada chunk individualmente
    # para garantizar que TODOS los segmentos quepan en el contexto
    if multi_segment_info:
        n_segments = len(multi_segment_info.get("segments", []))
        max_ctx = MAX_CONTEXT_CHARS_MULTI_SEGMENT
        # Presupuesto por chunk: garantiza que al menos n_segments + 3 chunks quepan
        min_chunks_needed = n_segments + 3
        chars_per_chunk = max(max_ctx // min_chunks_needed, MIN_CHARS_PER_CHUNK)
    else:
        max_ctx = MAX_CONTEXT_CHARS_NORMAL
        chars_per_chunk = None  # sin límite por chunk → comportamiento original
    
    context, _ = build_context(hits, max_chars=max_ctx, max_chars_per_chunk=chars_per_chunk)

    system = (
        "Eres un extractor de evidencia para un sistema RAG financiero. "
        "Tu tarea es identificar en el CONTEXTO los extractos que permiten responder la pregunta. "
        "Devuelve SOLO JSON válido, sin texto adicional."
    )

    # ================================================================
    # Instrucción adicional para multi-segmento
    # ================================================================
    multi_segment_instruction = ""
    if multi_segment_info:
        segments = multi_segment_info.get("segments", [])
        segment_list = ", ".join(segments)
        
        # Construir mapa de aliases para que el LLM sepa cómo buscar cada segmento
        alias_map = {}
        for alias, canonical in SEGMENT_ALIASES.items():
            if canonical in segments:
                alias_map.setdefault(canonical, []).append(alias)
        
        alias_hints = []
        for seg in segments:
            aliases = alias_map.get(seg, [seg])
            unique_aliases = sorted(set(a for a in aliases if a != seg))
            if unique_aliases:
                alias_hints.append(f"'{seg}' (también puede aparecer como: {', '.join(unique_aliases)})")
            else:
                alias_hints.append(f"'{seg}'")
        
        alias_text = "\n".join(f"     - {h}" for h in alias_hints)
        
        multi_segment_instruction = (
            f"\n9. COBERTURA MULTI-SEGMENTO OBLIGATORIA:\n"
            f"   La pregunta pide información de ESTOS segmentos: [{segment_list}].\n\n"
            f"   MAPA DE SEGMENTOS (busca estas variantes en CADA chunk):\n"
            f"{alias_text}\n\n"
            f"   CHECKLIST OBLIGATORIO — para cada segmento debes:\n"
            f"   a) Buscar en TODOS los chunks [S1] a [S{len(hits)}] si contiene datos de ese segmento.\n"
            f"   b) Los datos de un segmento pueden estar en CUALQUIER chunk, no solo en los primeros.\n"
            f"   c) Si encuentras el dato → agrégalo como key_point con el sid correspondiente.\n"
            f"   d) Si NO lo encuentras en ningún chunk → agrégalo en 'missing'.\n"
            f"   e) answerable=true solo si tienes datos de AL MENOS la mayoría de los segmentos.\n"
            f"   f) IMPORTANTE: Cada página del Directorio suele tener datos de UN segmento distinto.\n"
            f"      Revisa TODAS las páginas del Directorio antes de declarar que falta un segmento.\n"
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
        "   - Si el chunk tiene 'CONTEXTO DE LA MÉTRICA', ese campo te dice EXACTAMENTE qué métrica es.\n"
        "   - Ejemplo: Si CONTEXTO='Resultado de gestión neto AxI' y ves 'Corporate | 969', \n"
        "     entonces 969 es el Resultado de gestión neto AxI de Corporate, NO el resultado operativo.\n"
        "2. ASOCIAR ENTIDAD CON VALOR: Si la pregunta es sobre 'Banco Santander' y el chunk tiene una fila con 'Banco Santander | X',\n"
        "   ese valor X corresponde a Banco Santander para la métrica indicada en el topic o CONTEXTO.\n"
        "3. NO inventes información. Usa SOLO lo presente en el CONTEXTO.\n"
        "4. Cada quote debe ser literal y de máximo 25 palabras.\n"
        "5. Si la pregunta menciona una entidad específica, verifica que ESA entidad aparezca en el chunk.\n"
        "6. Incluye máximo 12 key_points y máximo 12 evidence.\n"
        "7. DEDUP POR FUENTE: Si dos chunks de fuentes distintas (Directorio vs CdG) reportan la misma métrica "
        "con valores diferentes, usa SOLO el dato de la FUENTE PRIMARIA e ignora la SECUNDARIA. "
        "No reportes el mismo dato dos veces con valores distintos.\n"
        "8. DISTINCIÓN DE MÉTRICAS: Si el CONTEXTO DE LA MÉTRICA indica una métrica distinta a la preguntada,\n"
        "   marca answerable=false o indica en missing qué métrica tiene vs cuál se pidió.\n"
        f"{multi_segment_instruction}"
    )

    # Para multi-segmento, más tokens para cubrir todos los segmentos
    max_tok = MAX_TOKENS_EVIDENCE_MULTI if multi_segment_info else MAX_TOKENS_EVIDENCE_NORMAL

    raw = call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=TEMPERATURE_EVIDENCE,
        max_tokens=max_tok
    )
    parsed = safe_json_load(raw) or {"answerable": False, "missing": ["No se pudo parsear evidencia"], "key_points": [], "evidence": []}

    # --- Post-procesado defensivo: dedup + defaults ---
    parsed.setdefault("answerable", False)
    parsed.setdefault("missing", [])
    parsed.setdefault("key_points", [])
    parsed.setdefault("evidence", [])

    # Límites ajustados para multi-segmento
    ev_limit = EVIDENCE_LIMIT_MULTI if multi_segment_info else EVIDENCE_LIMIT_NORMAL
    kp_limit = KEY_POINTS_LIMIT_MULTI if multi_segment_info else KEY_POINTS_LIMIT_NORMAL

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
    parsed["evidence"] = uniq_ev[:ev_limit]

    # Dedup key_points by claim text
    seen_kp = set()
    uniq_kp = []
    for kp in parsed.get("key_points", []) or []:
        claim = (kp.get("claim") or "").strip()
        if claim and claim not in seen_kp:
            uniq_kp.append(kp)
            seen_kp.add(claim)
    parsed["key_points"] = uniq_kp[:kp_limit]

    return parsed 


# -------------------------
# Respuesta con evidencia - MEJORADO con soporte multi-segmento
# -------------------------
def answer_from_evidence(
    query: str, 
    hits: List[Dict[str, Any]], 
    evidence: Dict[str, Any],
    multi_segment_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Respuesta directa y factual.
    
    NUEVO: Cuando multi_segment_info está presente, fuerza modo 
    "multi_segment" que requiere cobertura de todos los segmentos.
    
    Para queries normales (multi_segment_info=None), el comportamiento
    es idéntico al anterior.
    """
    # Para multi-segmento, más contexto con límite por chunk
    if multi_segment_info:
        n_segments = len(multi_segment_info.get("segments", []))
        max_ctx = MAX_CONTEXT_CHARS_MULTI_SEGMENT
        min_chunks_needed = n_segments + 3
        chars_per_chunk = max(max_ctx // min_chunks_needed, MIN_CHARS_PER_CHUNK)
    else:
        max_ctx = MAX_CONTEXT_CHARS_NORMAL
        chars_per_chunk = None
    
    context, _ = build_context(hits, max_chars=max_ctx, max_chars_per_chunk=chars_per_chunk)

    ql = (query or "").lower()
    
    # ================================================================
    # Detección de modo
    # ================================================================
    if multi_segment_info:
        # NUEVO: modo multi-segmento, siempre que haya info de multi-segmento
        mode = "multi_segment"
    else:
        # Lógica existente sin cambios
        is_comparative = any(k in ql for k in [" vs ", " versus", "compar", "octubre", "septiembre", "moM", "mom"])
        focused_segment = None
        for seg in SEGMENTS:
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
        "REGLA CRÍTICA - CONTEXTO DE LA MÉTRICA:\n"
        "- Algunos chunks incluyen un campo 'CONTEXTO DE LA MÉTRICA' que describe EXACTAMENTE\n"
        "  qué métrica representan los valores numéricos de esa tabla.\n"
        "- Si el CONTEXTO dice 'Resultado de gestión neto AxI', esos valores son SOLO esa métrica.\n"
        "  NO son 'resultado operativo', NI 'resultado neto contable', NI otra métrica distinta.\n"
        "- Si el usuario pregunta por una métrica diferente a la indicada en el CONTEXTO,\n"
        "  responde que los datos disponibles corresponden a la métrica del CONTEXTO,\n"
        "  no a la métrica preguntada.\n\n"
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

    # ================================================================
    # NUEVO: Formato multi-segmento con mapa de aliases
    # ================================================================
    multi_segment_format = ""
    if multi_segment_info:
        segments = multi_segment_info.get("segments", [])
        segment_list = ", ".join(segments)
        
        # Construir mapa de aliases
        alias_map = {}
        for alias, canonical in SEGMENT_ALIASES.items():
            if canonical in segments:
                alias_map.setdefault(canonical, []).append(alias)
        
        alias_hints = []
        for seg in segments:
            aliases = alias_map.get(seg, [seg])
            unique_aliases = sorted(set(a for a in aliases if a != seg))
            if unique_aliases:
                alias_hints.append(f"'{seg}' (variantes: {', '.join(unique_aliases)})")
            else:
                alias_hints.append(f"'{seg}'")
        
        alias_text = "\n".join(f"   - {h}" for h in alias_hints)
        
        multi_segment_format = (
            f"FORMATO OBLIGATORIO (MODO MULTI-SEGMENTO):\n"
            f"La pregunta pide información de CADA UNO de estos segmentos: [{segment_list}].\n\n"
            f"MAPA DE SEGMENTOS — busca estas variantes en el CONTEXTO:\n"
            f"{alias_text}\n\n"
            f"1) ONE-LINER (1 frase): responde directo qué métrica y período se muestra.\n"
            f"2) DETALLE POR SEGMENTO — incluye CADA segmento que tenga datos:\n"
            f"   - Segmento: valor [S#]\n"
            f"   Repite para CADA segmento del que tengas datos.\n"
            f"3) Si algún segmento NO tiene datos disponibles, menciónalo explícitamente.\n"
            f"4) LECTURA RÁPIDA (1 frase): síntesis general.\n\n"
            f"REGLA CRÍTICA: NO respondas solo con un par de segmentos. Tu respuesta está INCOMPLETA\n"
            f"si no incluyes datos de todos los segmentos disponibles en el CONTEXTO.\n"
            f"INSTRUCCIÓN: Revisa CADA chunk [S1] a [S{len(hits)}]. Cada página del Directorio\n"
            f"suele contener datos de UN segmento distinto. Busca en TODAS las páginas antes de\n"
            f"declarar que un segmento no tiene datos.\n"
        )

    # Seleccionar formato según modo
    if mode == "multi_segment":
        format_block = multi_segment_format
    elif mode == "comparative":
        format_block = comparative_format
    else:
        format_block = focused_format

    gloss = glossary_snippet(query)

    user = (
        f"Modo: {mode}\n"
        f"Pregunta: {query}\n\n"
        f"{gloss}\n\n"
        f"{business_rules}\n\n"
        f"EVIDENCE (JSON):\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"{format_block}"
        "\nREGLAS CRÍTICAS:\n"
        "- Si evidence.answerable es false: responde 'No encuentro esa información.'\n"
        "- Si evidence.answerable es true: extrae el valor de la entidad preguntada.\n"
        "- IMPORTANTE: El topic del chunk te dice qué métrica es. Úsalo para interpretar.\n"
        "- CONTEXTO DE LA MÉTRICA: Si un chunk incluye este campo, describe EXACTAMENTE qué métrica\n"
        "  representan los valores. Si la métrica del contexto no coincide con lo preguntado,\n"
        "  aclara qué datos tienes y qué se preguntó (ej: 'Los datos disponibles corresponden a\n"
        "  Resultado de gestión neto AxI, no a resultado operativo').\n"
        "- NO inventes cifras: si no está, no la pongas.\n"
        "- DEDUP POR FUENTE: Cada chunk está etiquetado como FUENTE PRIMARIA o SECUNDARIA. "
        "Si dos fuentes distintas reportan la misma métrica con valores diferentes, "
        "usa SOLO el valor de la FUENTE PRIMARIA. No menciones el dato duplicado de la fuente secundaria.\n"
    )

    # Para multi-segmento, más tokens para la respuesta completa
    max_tok = MAX_TOKENS_ANSWER_MULTI if multi_segment_info else MAX_TOKENS_ANSWER_NORMAL

    return call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE_ANSWER_GENERATION,
        max_tokens=max_tok
    )