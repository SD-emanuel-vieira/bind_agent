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
from rag_lib.business_glossary import *
from rag_lib.llm import *
from rag_lib.lexical_fallback import *

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
# CELL 4: Embeddings for query_vector (Model Serving friendly)
# -------------------------
def embed_query(text: str) -> Optional[List[float]]:
    """Return embedding vector for a single string using the Databricks embeddings endpoint."""
    t = (text or "").strip()
    if not t:
        return None

    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Most Databricks embeddings endpoints accept {"input": ["..."]} or {"input": "..."}
        try:
            res = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": [t]})
        except Exception:
            res = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": t})
    except Exception as e:
        raise

    # Normalize common response shapes
    if isinstance(res, dict):
        # OpenAI-like: {"data":[{"embedding":[...]}]}
        data = res.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
            if isinstance(first, list):
                return first

        # Some endpoints: {"predictions":[...]} or {"embeddings":[...]}
        for k in ("predictions", "embeddings", "output"):
            v = res.get(k)
            if isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]
                if isinstance(first, list):
                    return first

    # If already a list
    if isinstance(res, list) and res and isinstance(res[0], (float, int)):
        return res  # type: ignore[return-value]

    return None


# -------------------------
# CELL 5: RETRIEVER (Vector Search + expansion + lexical fallback)
# - Guarantees lexical fallback is merged even if vector search fails
# -------------------------
def retrieve_candidates(query: str, k: int = TOP_K_CANDIDATES) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    # 1) Try vector search (best effort)
    try:
        q_expanded = expand_query_for_retrieval(query) or query
        qvec = embed_query(q_expanded)

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

    # 2) Always add lexical fallback (FULL_TEXT)
    hits = lexical_fallback(query, hits, limit=LEX_FALLBACK_LIMIT)

    # 3) Merge + dedupe by chunk_id
    merged = []
    seen = set()
    for h in hits:
        cid = h.get("chunk_id")
        if cid and cid not in seen:
            merged.append(h)
            seen.add(cid)
    
    # Siempre se ordena las evidencias por fecha de archivo, tomando el más reciente primero
    merged.sort(key=lambda h: (h.get("file_date") is not None, h.get("file_date")), reverse=True)

    return merged

# -------------------------
# CELL 6: Reranking with LLM
# -------------------------
def rerank_with_llm(query: str, hits: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    if not hits:
        return []

    items = []
    for idx, h in enumerate(hits, start=1):
        sid = f"S{idx}"
        snippet = shorten(h.get("chunk_text_clean", ""), RERANK_SNIPPET_CHARS)
        meta = (
            f'file_date={h.get("file_date")}, '
            f'path="{h.get("path")}", page_num={h.get("page_num")}, topic="{h.get("topic")}"'
        )
        items.append({"sid": sid, "meta": meta, "snippet": snippet, "hit": h})

    system = (
        "Eres un motor de reranking para recuperación de información.\n"
        "Ordena extractos por relevancia para responder la pregunta.\n"
        "Devuelve SOLO JSON válido, sin texto adicional."
    )

    user_lines = [f"Pregunta:\n{query}\n", "Candidatos:"]
    for it in items:
        user_lines.append(f"{it['sid']} | {it['meta']}\n{it['snippet']}\n")

    user_lines.append(
        "Devuelve JSON EXACTO:\n"
        "{\n"
        '  "ranked_sids": ["S3","S1",...],\n'
        '  "reasons": {"S3":"...", "S1":"..."}\n'
        "}\n"
        f"- ranked_sids debe incluir como máximo {top_k} ids.\n"
        "- Prioriza coincidencia literal con palabras clave del query si existe.\n"
        "- En caso de empate de relevancia, prioriza file_date más reciente.\n"
    )

    content = call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role":"system","content":system},{"role":"user","content":"\n".join(user_lines)}],
        temperature=TEMPERATURE_RERANK,
        max_tokens=650
    )

    parsed = safe_json_load(content)
    ranked = parsed.get("ranked_sids", [])

    if not ranked or not isinstance(ranked, list):
        return hits[:top_k]

    sid_to_hit = {f"S{i+1}": items[i]["hit"] for i in range(len(items))}
    reranked = [sid_to_hit[sid] for sid in ranked if sid in sid_to_hit]

    # fill up if needed
    if len(reranked) < top_k:
        seen = set(h.get("chunk_id") for h in reranked)
        for h in hits:
            if h.get("chunk_id") not in seen:
                reranked.append(h)
                if len(reranked) >= top_k:
                    break
    
    # Siempre se ordena las evidencias por fecha de archivo, tomando el más reciente primero
    reranked.sort(
        key=lambda h: (h.get("file_date") is not None, h.get("file_date")),
        reverse=True
    )

    return reranked[:top_k]

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
        "- Si la pregunta pide un segmento específico (ej: 'Empresas') y no aparece explícito, answerable=false.\n"
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
        "Eres una base de conocimiento sobre reportes corporativos (RAG). "
        "Usa el glosario solo para interpretar términos, pero no lo cites como evidencia. Las afirmaciones sobre hechos deben salir del CONTEXTO/EVIDENCE."
        "Responde de forma directa, detallada y 100% basada en evidencia. "
        "NO des recomendaciones, NO incluyas próximos pasos, NO inventes datos. "
        "Usa SOLO CONTEXTO y EVIDENCE."
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

# -------------------------
# CELL 10: Orchestrator (end-to-end RAG)
# -------------------------
def answer_with_rag(query: str) -> Dict[str, Any]:
    candidates = retrieve_candidates(query, k=TOP_K_CANDIDATES) or []
    top_hits = rerank_with_llm(query, candidates, top_k=TOP_K_FINAL) or candidates[:TOP_K_FINAL]
    evidence = extract_evidence(query, top_hits)
    answer = answer_from_evidence(query, top_hits, evidence)

    _, citations = build_context(top_hits, max_chars=MAX_CONTEXT_CHARS)

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence,
        "citations": citations or [],
        "retrieved_candidates": candidates,
        "reranked_hits": top_hits,
    }
