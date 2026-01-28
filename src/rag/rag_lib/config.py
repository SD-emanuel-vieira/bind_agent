import os

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
# LLM_ENDPOINT = "databricks-gpt-5-2"

# --- Retrieval params ---
TOP_K_CANDIDATES = int(_get_env("RAG_TOP_K_CANDIDATES"))
TOP_K_FINAL = int(_get_env("RAG_TOP_K_FINAL"))
LEX_FALLBACK_LIMIT = int(_get_env("RAG_LEX_FALLBACK_LIMIT","20"))

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
    "chunk_id","doc_id","path","file_date","file_type","page_id","page_num",
    "topic_heuristic",
    "topic_llm","topic_content",
    "chunk_text","chunk_type"
]

# # Para estos chunk_type, se EXIGE que la query contenga al menos 1 keyword.
CHUNK_TYPE_QUERY_GATES = {
    "figure_enriched": ["evolución", "evolucion", "tendencia", "trend", "evolution", "variación", "variacion"],
}

SEGMENTS = ["empresa", "corporate", "institucional", "minorista", "baas", "segmento"]





