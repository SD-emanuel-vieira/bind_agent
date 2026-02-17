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
TABLE_ = 'bind_agent.docs.silver_excel'
LLM_ENDPOINT_SQL = _get_env("RAG_LLM_ENDPOINT")
# LLM_ENDPOINT_SQL = 'databricks-llama-4-maverick' 
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
    "chunk_id","doc_id","path","file_date","file_type","page_id","page_num","page_segment",
    "topic_heuristic",
    "topic_llm","topic_content","context_text",
    "chunk_text","chunk_type","metadata_enrich"
]

# # Para estos chunk_type, se EXIGE que la query contenga al menos 1 keyword.
CHUNK_TYPE_QUERY_GATES = {
    "figure_enriched": ["evolución", "evolucion", "tendencia", "trend", "evolution", "variación", "variacion"],
}

# Segmentos conocidos
SEGMENTS = {"empresas", "corporate", "institucional", "minorista", "baas", "pyme", "pymes"}

# Lista ordenada de segmentos canónicos (sin duplicados, sin aliases)
# Usada por decompose_multi_segment_query cuando detecta "cada segmento" / "por banca"
CANONICAL_SEGMENTS = ["empresas", "corporate", "institucional", "minorista", "baas"]

# Mapeo de aliases a segmento canónico
SEGMENT_ALIASES = {
    "empresas": "empresas",
    "empresa": "empresas",
    "corporate": "corporate",
    "corp": "corporate",
    "institucional": "institucional",
    "institucionales": "institucional",
    "minorista": "minorista",
    "retail": "minorista",
    "individuos": "minorista",
    "baas": "baas",
    "banca as a service": "baas",
    "pyme": "pyme",
    "pymes": "pyme",
}

# ====================================================================
# CONTEXT LIMITS (evidence + answer)
# ====================================================================
MAX_CONTEXT_CHARS_MULTI_SEGMENT = int(_get_env("RAG_MAX_CONTEXT_CHARS_MULTI", "30000"))
MAX_CONTEXT_CHARS_NORMAL        = MAX_CONTEXT_CHARS  # reutiliza la existente
MIN_CHARS_PER_CHUNK             = int(_get_env("RAG_MIN_CHARS_PER_CHUNK", "2000"))

# ====================================================================
# TOKEN LIMITS POR FUNCIÓN LLM
# ====================================================================
MAX_TOKENS_EVIDENCE_MULTI  = int(_get_env("RAG_MAX_TOKENS_EVIDENCE_MULTI", "1500"))
MAX_TOKENS_EVIDENCE_NORMAL = int(_get_env("RAG_MAX_TOKENS_EVIDENCE_NORMAL", "800"))
MAX_TOKENS_ANSWER_MULTI    = int(_get_env("RAG_MAX_TOKENS_ANSWER_MULTI", "1200"))
MAX_TOKENS_ANSWER_NORMAL   = int(_get_env("RAG_MAX_TOKENS_ANSWER_NORMAL", "700"))
MAX_TOKENS_RERANK          = int(_get_env("RAG_MAX_TOKENS_RERANK", "650"))
MAX_TOKENS_QUERY_EXPANSION = int(_get_env("RAG_MAX_TOKENS_QUERY_EXPANSION", "120"))

# ====================================================================
# TEMPERATURAS ESPECÍFICAS
# ====================================================================
TEMPERATURE_EVIDENCE          = float(_get_env("RAG_TEMPERATURE_EVIDENCE", "0.0"))
TEMPERATURE_ANSWER_GENERATION = float(_get_env("RAG_TEMPERATURE_ANSWER_GENERATION", "0.05"))

# ====================================================================
# EVIDENCE POST-PROCESSING LIMITS
# ====================================================================
EVIDENCE_LIMIT_MULTI    = int(_get_env("RAG_EVIDENCE_LIMIT_MULTI", "12"))
EVIDENCE_LIMIT_NORMAL   = int(_get_env("RAG_EVIDENCE_LIMIT_NORMAL", "6"))
KEY_POINTS_LIMIT_MULTI  = int(_get_env("RAG_KEY_POINTS_LIMIT_MULTI", "12"))
KEY_POINTS_LIMIT_NORMAL = int(_get_env("RAG_KEY_POINTS_LIMIT_NORMAL", "6"))

# ====================================================================
# RERANK PARAMS
# ====================================================================
RERANK_TIE_BREAK_BLOCK_SIZE = int(_get_env("RAG_RERANK_TIE_BREAK_BLOCK", "2"))
ANCHOR_PROTECTION_RATIO     = float(_get_env("RAG_ANCHOR_PROTECTION_RATIO", "0.8"))
SOURCE_PRIORITY_DEFAULT     = int(_get_env("RAG_SOURCE_PRIORITY_DEFAULT", "9"))
TRACE_DEFAULT_TOP           = int(_get_env("RAG_TRACE_DEFAULT_TOP", "12"))
RERANK_INPUT_MULTIPLIER     = int(_get_env("RAG_RERANK_INPUT_MULTIPLIER", "3"))
RERANK_INPUT_OFFSET         = int(_get_env("RAG_RERANK_INPUT_OFFSET", "12"))

# ====================================================================
# RETRIEVER
# ====================================================================
VS_FULL_TEXT_MAX_RESULTS = int(_get_env("RAG_VS_FULL_TEXT_MAX_RESULTS", "200"))
VS_REQUEST_TIMEOUT_SECS  = int(_get_env("RAG_VS_REQUEST_TIMEOUT_SECS", "15"))
HYDRATION_MAX_IDS        = int(_get_env("RAG_HYDRATION_MAX_IDS", "500"))
LLM_CALL_TIMEOUT_SECS    = int(_get_env("RAG_LLM_CALL_TIMEOUT_SECS", "30"))

# ====================================================================
# SQL PARAMS
# ====================================================================
SQL_RESULT_LIMIT  = int(_get_env("RAG_SQL_RESULT_LIMIT", "20"))
SQL_TEMPERATURE   = float(_get_env("RAG_SQL_TEMPERATURE", "0.2"))
SQL_MAX_TOKENS    = int(_get_env("RAG_SQL_MAX_TOKENS", "500"))
SCHEMA_CACHE_TTL  = int(_get_env("RAG_SCHEMA_CACHE_TTL", "3600"))
