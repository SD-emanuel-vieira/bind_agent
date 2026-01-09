import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient

# ====== REQUIRED CONFIG ======
def _get_env(name: str, default: str | None = None) -> str:
    import os
    v = os.getenv(name, default)
    if v is None:
        raise ValueError(f"Missing env var: {name}")
    return v

# --- Vector Search ---
VS_ENDPOINT = _get_env("RAG_VS_ENDPOINT")
VS_INDEX_FULL_NAME = _get_env("RAG_VS_INDEX_FULL_NAME")

# --- Endpoints ---
EMBED_ENDPOINT = _get_env("RAG_EMBED_ENDPOINT")  # for query_vector
EMB_ENDPOINT = EMBED_ENDPOINT  # backward-compatible alias
LLM_ENDPOINT = _get_env("RAG_LLM_ENDPOINT")
# LLM_ENDPOINT = _get_env("RAG_LLM_ENDPOINT", default="databricks-gpt-5-2")

# --- Retrieval params ---
TOP_K_CANDIDATES = int(_get_env("RAG_TOP_K_CANDIDATES"))
TOP_K_FINAL = int(_get_env("RAG_TOP_K_FINAL"))
LEX_FALLBACK_LIMIT = int(_get_env("RAG_LEX_FALLBACK_LIMIT"))

# --- Prompt/context limits ---
MAX_CONTEXT_CHARS = int(_get_env("RAG_MAX_CONTEXT_CHARS"))
RERANK_SNIPPET_CHARS = int(_get_env("RAG_RERANK_SNIPPET_CHARS"))

# --- LLM params ---
TEMPERATURE_RERANK = float(_get_env("RAG_TEMPERATURE_RERANK"))
TEMPERATURE_ANSWER = float(_get_env("RAG_TEMPERATURE_ANSWER"))
MAX_TOKENS_ANSWER = int(_get_env("RAG_MAX_TOKENS_ANSWER"))

# --- Retry params ---
MAX_RETRIES = int(_get_env("RAG_MAX_RETRIES"))
RETRY_SLEEP_SECS = float(_get_env("RAG_RETRY_SLEEP_SECS"))


# Columns we want back from Vector Search (both vector + FULL_TEXT fallback)
VS_COLUMNS = [
    "chunk_id","doc_id","path","file_date","file_type",
    "page_id","page_num","topic","chunk_text"
]

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
# CELL 3: Helper functions (text cleaning + parsing + LLM call)
# -------------------------
def strip_chunk_prefix(text: str) -> str:
    """Remove [SOURCE]/[TOPIC] prefix (if your gold chunks included it)."""
    if not text:
        return ""
    return re.sub(r"(?s)^\[SOURCE:[^\]]*\]\s*\n\[TOPIC:[^\]]*\]\s*\n\s*", "", text).strip()

def shorten(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else (t[:n].rstrip() + "…")

def safe_json_load(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        s2 = s[i:j+1]
        try:
            return json.loads(s2)
        except Exception:
            return {}
    return {}

def extract_chat_content(resp: Any) -> str:
    """Best-effort extraction of chat content from serving response."""
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            msg = resp["choices"][0].get("message", {})
            return msg.get("content", "") or ""
        if "predictions" in resp and resp["predictions"]:
            p0 = resp["predictions"][0]
            if isinstance(p0, dict) and "content" in p0:
                return p0["content"]
            if isinstance(p0, str):
                return p0
    return str(resp)

def call_chat(endpoint: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = client.predict(endpoint=endpoint, inputs=payload)
    return extract_chat_content(resp)

def parse_vs_similarity_response(res: Any) -> List[Dict[str, Any]]:
    """Normalize Vector Search similarity_search response to list[dict]."""
    if isinstance(res, dict):
        r = res.get("result") or res
        cols = r.get("columns")
        data = r.get("data_array") or r.get("data") or []
        if cols and data:
            out = []
            for row in data:
                out.append({c: row[i] for i, c in enumerate(cols)})
            return out
    if isinstance(res, list):
        return res
    return []

# -------------------------
# CELL 5: Query expansion (ES -> keywords + EN)
# -------------------------
def expand_query_for_retrieval(query: str) -> str:
    """
    Expand query to keywords/synonyms including EN terms because embeddings are -en.
    Returns a SINGLE LINE.
    """
    system = (
        "Reescribe consultas para mejorar recuperación semántica. "
        "Devuelve SOLO una línea de keywords/sinónimos, sin explicaciones."
    )
    user = (
        f"Consulta original (ES): {query}\n"
        "Reescribe como query de búsqueda con keywords y sinónimos ES+EN. "
        "Incluye términos relacionados.\n"
        "Ejemplos ES/EN: previsiones/proyecciones/estimaciones; forecast/outlook/guidance/projections/estimates.\n"
        "Si hay fechas (mes/año), mantenlas."
    )
    return call_chat(
        endpoint=LLM_ENDPOINT,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.0,
        max_tokens=120
    ).strip()

# -------------------------
# CELL 6: Lexical fallback via Vector Search FULL_TEXT (serving-friendly)
# - No Spark required (works in Model Serving)
# - Queries the same Vector Search index using query_type=FULL_TEXT
# - Merges + dedupes with existing hits
# -------------------------
def _get_dbx_auth() -> Tuple[str, str]:
    """Get Databricks host + token.

    Priority:
      1) Env vars: DATABRICKS_HOST/WORKSPACE_URL + DATABRICKS_TOKEN/TOKEN
      2) MLflow Databricks creds (works in Databricks notebooks)
      3) Notebook context token (best-effort)
    """
    host = (os.getenv("DATABRICKS_HOST") or os.getenv("WORKSPACE_URL") or "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN") or os.getenv("TOKEN") or ""

    # 2) MLflow helper (usually works in Databricks notebooks)
    if not host or not token:
        try:
            from mlflow.utils.databricks_utils import get_databricks_host_creds
            creds = get_databricks_host_creds()
            host = host or (getattr(creds, "host", "") or "").rstrip("/")
            token = token or (getattr(creds, "token", "") or "")
        except Exception:
            pass

    # 3) Notebook context fallback (best-effort)
    if not host or not token:
        try:
            # Only import pyspark in notebook clusters; in serving this may not exist
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            if not host:
                host = ("https://" + spark.conf.get("spark.databricks.workspaceUrl")).rstrip("/")
            if not token:
                # dbutils available in notebooks
                token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        except Exception:
            pass

    if not host or not token:
        raise RuntimeError(
            "Missing Databricks auth. Set DATABRICKS_HOST + DATABRICKS_TOKEN (or run inside a Databricks notebook)."
        )

    return host, token

def _vs_full_text_query(
    index_name: str,
    query_text: str,
    columns: List[str],
    num_results: int,
    filters: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Call Vector Search REST API for FULL_TEXT queries and return rows as list[dict]."""
    host, token = _get_dbx_auth()
    url = f"{host}/api/2.0/vector-search/indexes/{index_name}/query"

    payload: Dict[str, Any] = {
        "query_text": query_text,
        "query_type": "FULL_TEXT",
        "columns": columns,
        "num_results": min(int(num_results), 200),
    }
    if filters is not None:
        payload["filters"] = filters

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result") or data
    colnames = result.get("column_names") or columns
    rows = result.get("data_array") or result.get("data") or []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = {colnames[i]: row[i] for i in range(min(len(colnames), len(row)))}
        out.append(d)
    return out

def lexical_fallback(query: str, hits: List[Dict[str, Any]], limit: int = LEX_FALLBACK_LIMIT) -> List[Dict[str, Any]]:
    """Best-effort: agrega matches lexicográficos desde Vector Search index (FULL_TEXT)."""
    q = (query or "").strip()
    if not q:
        return hits

    try:
        rows = _vs_full_text_query(
            index_name=VS_INDEX_FULL_NAME,
            query_text=q,
            columns=VS_COLUMNS,
            num_results=limit,
            filters=None,
        )
    except Exception as e:
        # If serving env doesn't have auth configured, do not break main flow.
        print("Lexical FULL_TEXT fallback failed. Error:", repr(e))
        return hits

    lex_hits: List[Dict[str, Any]] = []
    for d in rows:
        raw = (d.get("chunk_text") or "").strip()
        d["chunk_text_clean"] = strip_chunk_prefix(raw)
        lex_hits.append(d)

    # merge + dedupe por chunk_id (primero mantiene orden de hits existentes)
    merged: List[Dict[str, Any]] = []
    seen = set()
    for h in hits + lex_hits:
        cid = h.get("chunk_id")
        key = cid if cid is not None else (h.get("path"), h.get("page_id"), h.get("page_num"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)

    return merged


# -------------------------
# Embeddings for query_vector (Model Serving friendly)
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
# CELL 7: RETRIEVER (Vector Search + expansion + lexical fallback)
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

    return merged

# -------------------------
# CELL 8: Reranking with LLM
# -------------------------
def rerank_with_llm(query: str, hits: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    if not hits:
        return []

    items = []
    for idx, h in enumerate(hits, start=1):
        sid = f"S{idx}"
        snippet = shorten(h.get("chunk_text_clean", ""), RERANK_SNIPPET_CHARS)
        meta = f'path="{h.get("path")}", page_num={h.get("page_num")}, topic="{h.get("topic")}"'
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

    return reranked[:top_k]

# -------------------------
# CELL 9: Build context (with [S#] citations)
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
# CELL 10: Extracción de evidencia (GENÉRICA)
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
# CELL 11: Respuesta con evidencia
# -------------------------
def answer_from_evidence(query: str, hits: List[Dict[str, Any]], evidence: Dict[str, Any]) -> str:
    """
    Respuesta directa y factual.
    - Modo 'comparativo/generico': resumen por segmento + desglose + lectura rápida.
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
        "Eres un asistente de QA sobre reportes corporativos (RAG). "
        "Responde de forma directa, detallada y 100% basada en evidencia. "
        "NO des recomendaciones, NO incluyas próximos pasos, NO inventes datos. "
        "Usa SOLO CONTEXTO y EVIDENCE."
    )

    # Plantilla para modo comparativo (más parecido a tu ejemplo)
    comparative_format = (
        "FORMATO OBLIGATORIO (MODO COMPARATIVO):\n"
        "1) ONE-LINER (1 frase): responde directo qué muestra el documento sobre la comparación solicitada.\n"
        "2) 'Resumen por segmento' (si el documento lo trae):\n"
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

    user = (
        f"Modo: {mode}\n"
        f"Pregunta: {query}\n\n"
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
# CELL 12: Orchestrator (end-to-end RAG)
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

# -------------------------
# MLflow PyFunc Model
# -------------------------
class RagAgent(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):  # type: ignore[override]
        try:
            rows = model_input.to_dict(orient="records")
        except Exception:
            rows = model_input if isinstance(model_input, list) else [model_input]

        outputs = []
        for row in rows:
            q = None
            if isinstance(row, dict) and row.get("messages"):
                q = row["messages"][-1].get("content")
            if not q and isinstance(row, dict):
                q = row.get("query")

            if not q:
                outputs.append({"answer": "", "sources_json": "[]", "error": "Missing 'messages' or 'query'"})
                continue

            try:
                out = answer_with_rag(q)
                # sources_json: lista simple (compatible con tu smoke test)
                sources = []
                for i, h in enumerate(out.get("reranked_hits", []) or [], start=1):
                    sources.append({
                        "sid": f"S{i}",
                        "path": h.get("path"),
                        "page_num": h.get("page_num"),
                        "chunk_id": h.get("chunk_id"),
                        "topic": h.get("topic"),
                    })
                outputs.append({
                    "answer": out.get("answer", ""),
                    "sources_json": json.dumps(sources, ensure_ascii=False),
                    "error": "",
                })
            except Exception as e:
                outputs.append({"answer": "", "sources_json": "[]", "error": str(e)})

        return outputs


# ✅ requerido para “code-based logging”
mlflow.models.set_model(RagAgent())
