import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import requests
import mlflow.deployments

def call_chat(endpoint: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    client = mlflow.deployments.get_deploy_client("databricks")
    resp = client.predict(endpoint=endpoint, inputs=payload)
    return extract_chat_content(resp)