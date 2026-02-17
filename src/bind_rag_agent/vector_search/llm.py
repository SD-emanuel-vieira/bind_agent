import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
import mlflow.deployments
import signal
from contextlib import contextmanager
from bind_rag_agent.text_utils import extract_chat_content
from bind_rag_agent.config import (
    LLM_ENDPOINT,
    LLM_CALL_TIMEOUT_SECS,
    TEMPERATURE_EVIDENCE,
    MAX_TOKENS_QUERY_EXPANSION,
)

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    def handler(signum, frame):
        raise TimeoutError(f"Operación excedió {seconds} segundos")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# LLM call to chat
def call_chat(endpoint: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, timeout_seconds: int = LLM_CALL_TIMEOUT_SECS) -> str:
    with timeout(timeout_seconds):
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
        temperature=TEMPERATURE_EVIDENCE,
        max_tokens=MAX_TOKENS_QUERY_EXPANSION
    ).strip()