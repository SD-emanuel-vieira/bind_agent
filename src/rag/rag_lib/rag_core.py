import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
import mlflow.deployments
from mlflow.utils.databricks_utils import get_databricks_host_creds
from databricks.vector_search.client import VectorSearchClient
from rag_lib.secret_functions import *
from rag_lib.config import *
from rag_lib.text_utils import *
from rag_lib.lexical_fallback import _get_dbx_auth, _vs_full_text_query, lexical_fallback
from rag_lib.business_glossary import BUSINESS_GLOSSARY_V2
from rag_lib.glossary_helper import glossary_snippet, glossary_expand_terms, prepare_rerank_candidates_glossary_aware
from rag_lib.llm import call_chat, expand_query_for_retrieval
from rag_lib.embeddings import embed_query
from rag_lib.rerank import query_anchors,hit_anchor_score,tie_break_by_date_in_blocks,enforce_anchor_priority,rerank_with_llm
# from rag_lib.trace import trace_stage

# Clients
vsc = VectorSearchClient()
index = vsc.get_index(VS_ENDPOINT, VS_INDEX_FULL_NAME)
client = mlflow.deployments.get_deploy_client("databricks")

print("Config OK")
print("VS endpoint:", VS_ENDPOINT)
print("VS index:", VS_INDEX_FULL_NAME)
print("Embedding endpoint:", EMBED_ENDPOINT)
print("LLM endpoint:", LLM_ENDPOINT)

# -------------------------
# CELL 5: RETRIEVER (Vector Search + expansion + lexical fallback)
# - Guarantees lexical fallback is merged even if vector search fails
# -------------------------
def retrieve_candidates(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    # ✅ default (por si falla el try)
    q_fulltext = query

    # 1) Try vector search (best effort)
    try:
        gl = glossary_expand_terms(query)
        terms = gl.get("terms", []) or []
        acronyms = gl.get("acronyms", []) or []

        q_llm = expand_query_for_retrieval(query) or query

        # Query para embeddings/vector: más rica (mejor semántica)
        q_embed = q_llm
        if terms:
            q_embed = q_embed + "\n\nGLOSSARY TERMS: " + " | ".join(terms)

        # Helper de quoting para FULL_TEXT
        def _qt(t: str) -> str:
            t = (t or "").strip()
            return f'"{t}"' if " " in t else t

        # ✅ FULL_TEXT "glossary-focused" si hay acrónimos relevantes (evita dilución)
        # (Ignoramos acrónimos de 1 letra tipo R/E)
        focus_acronyms = [a.strip() for a in acronyms if isinstance(a, str) and len(a.strip()) >= 2]

        if focus_acronyms:
            # FULL_TEXT más "afilado": buscar directo por ROA/ROE/etc
            q_fulltext = " ".join(_qt(a) for a in focus_acronyms)
        else:
            # FULL_TEXT estándar: query + términos (como estaba antes)
            q_fulltext = query
            if terms:
                q_fulltext = q_fulltext + " " + " ".join(_qt(t) for t in terms)

        qvec = embed_query(q_embed)

        if qvec:
            res = index.similarity_search(
                query_vector=qvec,  # direct access index requires query_vector
                columns=VS_COLUMNS,
                num_results=k
            )
            hits = parse_vs_similarity_response(res)

            for h in hits:
                raw = (h.get("chunk_text") or "").strip()
                h["chunk_text_clean"] = strip_chunk_prefix(raw)

    except Exception as e:
        print("Vector retrieval failed, fallback lexical only. Error:", repr(e))
        hits = []
        q_fulltext = query  # ✅ aseguramos valor válido

    # 2) Always add lexical fallback (FULL_TEXT)
    # hits = lexical_fallback(q_fulltext, hits, limit=LEX_FALLBACK_LIMIT)

    # 2.1) Exclude evidence that would be chart analysis
    hits = filter_hits_by_query_gates(query, hits, CHUNK_TYPE_QUERY_GATES)

    # 3) Merge + dedupe by chunk_id
    merged = []
    seen = set()
    for h in hits:
        cid = h.get("chunk_id")
        if cid and cid not in seen:
            merged.append(h)
            seen.add(cid)

    return merged

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

### -------------- DEBUGGING

import os, re, unicodedata
from collections import Counter

DEBUG_TRACE = os.getenv("RAG_DEBUG_TRACE", "0") == "1"

def _short_id(cid: str, n: int = 10) -> str:
    cid = cid or ""
    return cid[:n]

def trace_stage(stage: str, query: str, hits: list[dict], top: int = 8):
    """Tracer resumido para debuggear orden sin spamear output."""
    if not DEBUG_TRACE:
        return
    if hits is None:
        hits = []

    anchors = query_anchors(query)
    total = len(hits)

    # stats rápidos
    anchor_scores = [hit_anchor_score(h, anchors) for h in hits] if anchors else [0] * total
    n_anchor_pos = sum(1 for s in anchor_scores if s > 0)

    ctype_counts = Counter((h.get("chunk_type") or "NA") for h in hits)
    top_ctypes = ", ".join([f"{k}:{v}" for k, v in ctype_counts.most_common(3)])

    print("\n" + "-" * 110)
    print(f"[TRACE] {stage} | total={total} | anchors={anchors} | anchor_hits>0={n_anchor_pos} | chunk_type={top_ctypes}")

    # ✅ si querés imprimir TODOS, setear env var RAG_DEBUG_TRACE_ALL=1
    trace_all = os.getenv("RAG_DEBUG_TRACE_ALL", "0") == "1"
    limit = total if trace_all else min(top, total)

    for i in range(limit):
        h = hits[i]
        a = anchor_scores[i] if i < len(anchor_scores) else 0
        fd = h.get("file_date") or ""
        pg = h.get("page_num")
        ct = h.get("chunk_type") or ""
        gb = h.get("_glossary_bonus")
        
        # NUEVO: Mostrar el anchor_score calculado internamente
        internal_anchor = h.get("_anchor_score", "?")
        
        cid = _short_id(h.get("chunk_id"))

        path = h.get("path") or ""
        tail = path.split("/")[-1] if path else ""

        # ACTUALIZADO: formato con anchor_score interno
        print(f"  {i+1:02d}) a={a} | a_int={internal_anchor} | g={gb} | {fd} | p={pg} | {ct} | {tail} | cid={cid}")

# -------------------------
# CELL 10: Orchestrator (end-to-end RAG)
# -------------------------
def answer_with_rag(query: str) -> Dict[str, Any]:
    # Posibles candidatos para la respuesta, se filtran por lexical_fallback (importtante) y chunk_type:
    candidates = retrieve_candidates(query, k=TOP_K_CANDIDATES) or [] 
    trace_stage("1) retrieve_candidates", query, candidates)

    #Si no se menciona ningun segmento entonces descarta toda evidencia relacionada a cualquier segmentpo:
    candidates = drop_segment_topics_if_query_general(query, candidates)
    trace_stage("1.1) drop_segment_topics_if_query_general", query, candidates)

    # Soft ordering #1: anchors (gating suave por intención)
    # candidates_anchor_sorted = prefer_anchor_hits(query, hits_for_rerank)
    candidates_anchor_sorted = enforce_anchor_priority(query, candidates)
    trace_stage("2) enforce_anchor_priority", query, candidates_anchor_sorted)

    # Soft ordering #2: glossary-aware (estricto por frases)
    # Preprocesamiento de candidatos glossary-aware:
    TOP_K_RERANK_INPUT = max(TOP_K_FINAL * 3, TOP_K_FINAL + 12) 
    hits_for_rerank = prepare_rerank_candidates_glossary_aware(query, candidates, max_input=TOP_K_RERANK_INPUT) 
    trace_stage(f"3) prepare_rerank_candidates_glossary_aware(max_input={TOP_K_RERANK_INPUT})", query, hits_for_rerank)

    # Tie-break SOLO para empates (por file_date) — al final del pre-rerank
    hits_for_rerank_tiebroken = tie_break_by_date_in_blocks(hits_for_rerank, block_size=2)
    trace_stage("4) tie_break_by_date_in_blocks(block_size=2)", query, hits_for_rerank_tiebroken)

    # Reranking en base a las reglas definidas:
    top_hits = rerank_with_llm(query, hits_for_rerank_tiebroken, top_k=TOP_K_FINAL) or candidates[:TOP_K_FINAL] 
    trace_stage(f"5) rerank_with_llm(top_k={TOP_K_FINAL})", query, top_hits)

    # Se contruye la evidencia:
    evidence = extract_evidence(query, top_hits)
    
    # Se arma la respuesta final:
    answer = answer_from_evidence(query, top_hits, evidence)
    
    # Se construye el contexto:
    _, citations = build_context(top_hits, max_chars=MAX_CONTEXT_CHARS) 

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence,
        "citations": citations or [],
        "retrieved_candidates": candidates,
        "reranked_hits": top_hits,
    }
