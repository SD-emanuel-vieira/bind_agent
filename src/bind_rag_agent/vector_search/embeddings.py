import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from mlflow.deployments import get_deploy_client

from bind_rag_agent.config import EMBED_ENDPOINT
# -------------------------
# CELL 4: Embeddings for query_vector (Model Serving friendly)
# -------------------------
def embed_query(text: str) -> Optional[List[float]]:
    """Return embedding vector for a single string using the Databricks embeddings endpoint."""
    t = (text or "").strip()
    if not t:
        return None

    try:
        client = get_deploy_client("databricks")
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

    raise ValueError(
        f"Formato de respuesta inesperado del endpoint {EMBED_ENDPOINT}: "
        f"{type(res).__name__}"
    )