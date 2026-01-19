import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
import mlflow.deployments
from rag_lib.text_utils import extract_chat_content
from rag_lib.config import *

# LLM call to chat
def call_chat(endpoint: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    client = mlflow.deployments.get_deploy_client("databricks")
    resp = client.predict(endpoint=endpoint, inputs=payload)
    return extract_chat_content(resp)

# -------------------------
# CELL 2: Query expansion (ES -> keywords + EN)
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