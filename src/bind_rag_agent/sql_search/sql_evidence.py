import os
import re
import logging
import mlflow
from mlflow.deployments import get_deploy_client
from typing import List, Dict, Any
from contextlib import contextmanager
from pyspark.sql import SparkSession
import time
from bind_rag_agent.config import TABLE_, LLM_ENDPOINT_SQL

# Obtener la sesión de Spark activa
spark = SparkSession.builder.getOrCreate()


# =========================================================================
# Utilidad para suprimir logs de error de PySpark/gRPC
# =========================================================================

@contextmanager
def suppress_spark_errors():
    """Suprime temporalmente los logs de error de PySpark y gRPC."""
    loggers_to_suppress = [
        'pyspark.sql.connect.client',
        'pyspark.sql.connect.client.logging', 
        'grpc._channel',
        'py4j',
    ]
    
    original_levels = {}
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.CRITICAL + 1)  # Suprime todo
    
    try:
        yield
    finally:
        # Restaurar niveles originales
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def schema_text(table: str) -> str:
    fields = spark.table(table).schema.fields
    return "\n".join([f"- {f.name}: {f.dataType.simpleString()}" for f in fields])

_SCHEMA_CACHE = {"text": None, "timestamp": 0}
SCHEMA_CACHE_TTL = 3600  # 1 hora

def get_schema_text() -> str:
    global _SCHEMA_CACHE
    now = time.time()
    
    if _SCHEMA_CACHE["text"] is None or (now - _SCHEMA_CACHE["timestamp"]) > SCHEMA_CACHE_TTL:
        _SCHEMA_CACHE["text"] = schema_text(TABLE_)
        _SCHEMA_CACHE["timestamp"] = now
    
    return _SCHEMA_CACHE["text"]

def sql_tool(sql: str, n: int = 20) -> str:
    df = spark.sql(sql)
    return df.limit(n).toPandas().to_string(index=False)

def run_sql_preview(sql: str, n: int = 20) -> dict:
    """Ejecuta SQL y retorna resultado con metadata. Suprime logs de error."""
    with suppress_spark_errors():
        try:
            df = spark.sql(sql)
            has_rows = df.limit(1).count() > 0
            preview = df.limit(n).toPandas().to_string(index=False)
            return {'ok': True, 'has_rows': has_rows, 'preview': preview, 'error': None}
        except Exception as e:
            # Extraer solo el mensaje relevante del error
            error_msg = str(e)
            # Simplificar mensaje de error si es muy largo
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return {'ok': False, 'has_rows': False, 'preview': '', 'error': error_msg}
    
def extract_chat_content(resp) -> str:
    if isinstance(resp, dict):
        d = resp
    else:
        try:
            d = resp.__dict__
        except Exception:
            d = resp

    try:
        return d['choices'][0]['message']['content']
    except Exception:
        pass

    try:
        return d['predictions'][0]['content']
    except Exception:
        pass

    return str(resp)


def call_llm(endpoint: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 500) -> str:
    payload = {'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
    client = get_deploy_client('databricks')
    resp = client.predict(endpoint=endpoint, inputs=payload)
    return extract_chat_content(resp)

def call_llm_simple(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    return call_llm(LLM_ENDPOINT_SQL, messages, temperature=0.2, max_tokens=500)

def clean_sql(sql: str) -> str:
    s = sql.strip()

    if s.startswith('```'):
        lines = s.splitlines()

        if len(lines) >= 1 and lines[0].strip().startswith('```'):
            lines = lines[1:]

        if len(lines) >= 1 and lines[-1].strip().startswith('```'):
            lines = lines[:-1]

        s = '\n'.join(lines).strip()

    return s

def enforce_limit(sql: str, n: int = 20) -> str:
    s = sql.strip().rstrip(';')
    if 'limit' not in s.lower():
        return s + f'\nLIMIT {n}'
    return s

FORBIDDEN_PATTERNS = [
    r'\bdrop\b', r'\bdelete\b', r'\bupdate\b', r'\binsert\b', 
    r'\balter\b', r'\btruncate\b', r'\bcreate\b', r'\bgrant\b',
    r'--', r'/\*', r'\*/', r';(?!$)'  # Permitir ; solo al final
]

def validate_sql(sql: str, table: str):
    s = sql.strip().lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, s):
            raise ValueError(f'SQL bloqueada por patrón: {pattern}')
    
    if table.lower() not in s:
        raise ValueError(f'La query SQL debe referenciar {table}')
    
    if not s.startswith('select'):
        raise ValueError('Solo se permiten consultas SELECT')

def text_to_sql(question: str, table: str, schema_txt: str) -> str:
    # Prompt mejorado: más explícito sobre funciones en inglés
    system = '''Sos experto en SQL Spark (Databricks). Devolvé SOLO SQL válido. Nada de explicación.
    IMPORTANTE: Usá funciones de Spark SQL en INGLÉS (contains, lower, trim, etc). NO uses funciones en español.'''
        
    user = f"""
        Generá una query SQL Spark para responder la pregunta usando SOLO esta tabla: {table}

        Esquema:
        {schema_txt}

        Reglas ESTRICTAS:
        - Usá nombres exactos de columnas del esquema.
        - No uses DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE.
        - Para filtros por texto (cuenta_bt, cuit, cliente, producto, sub_producto, moneda y oficial), siempre usá búsqueda PARCIAL case-insensitive con la función CONTAINS (en inglés):
        Sintaxis correcta: contains(lower(columna), lower('valor'))
        Ejemplo: contains(lower(cliente), lower('santander'))
        - NUNCA uses "contiene", "contener" u otras funciones en español. Solo funciones Spark SQL en inglés.
        - No uses '=' para filtrar valores de texto.
        - Cuando se pregunte por "resultado neto", "resultado bruto", "IIBB", "interes cobrado" o "interes pagado" se tiene que hacer una suma por ese campo agrupado por el campo "moneda".
        - Si preguntan por valor de un campo como "tasa activa" en los filtros se deben excluir registros null para ese campo:
        Ejemplo: tasa_activa IS NOT NULL
        - Devolvé SOLO el SQL (sin ```).

        Pregunta: {question}
        """

    raw = call_llm_simple(system, user)   
    sql = clean_sql(raw)
    sql = enforce_limit(sql, 20)                  
    validate_sql(sql, table)
    return sql


# =========================================================================
# FUNCIÓN PRINCIPAL: Evidencia estructurada para integración con RAG
# =========================================================================
def get_sql_evidence(question: str) -> Dict[str, Any]:
    """
    Genera evidencia estructurada a partir de una query SQL.
    Nunca lanza excepciones - siempre retorna un dict con el estado.
    
    Returns:
        Dict con estructura:
        {
            "success": bool,
            "has_data": bool,
            "answer": str | None,
            "raw_data": str | None,
            "source": str,
            "query": str | None,           # ← NUEVO
            "error_type": str | None,
            "error_message": str | None
        }
    """
    result = {
        "success": False,
        "has_data": False,
        "answer": None,
        "raw_data": None,
        "source": "tabla_excel_financiera",
        "query": None,                      # ← NUEVO
        "error_type": None,
        "error_message": None
    }
    
    # Paso 1: Generar SQL
    try:
        schema_txt = get_schema_text()
        sql = text_to_sql(question, TABLE_, schema_txt)
        result["query"] = sql               # ← NUEVO: Guardar la query generada
    except Exception as e:
        result["error_type"] = "sql_generation"
        result["error_message"] = str(e)[:200] if len(str(e)) > 200 else str(e)
        return result
    
    # Paso 2: Ejecutar SQL (con logs suprimidos)
    r = run_sql_preview(sql, n=20)
    
    if not r['ok']:
        result["error_type"] = "sql_execution"
        result["error_message"] = r['error']
        return result                       # ← Ya tiene result["query"] = sql
    
    if not r['has_rows']:
        result["success"] = True
        result["error_type"] = "no_rows"
        result["error_message"] = "La consulta no retornó resultados"
        return result                       # ← Ya tiene result["query"] = sql
    
    # Paso 3: Generar respuesta en lenguaje natural
    result["raw_data"] = r['preview']
    
    try:
        system = 'Sos un asistente de datos. Respondé en español de forma concisa y directa. No inventes. Si falta info, decilo.'
        prompt = f'''
            Pregunta: {question}

            Datos obtenidos:
            {r["preview"]}

            Instrucciones:
            - Respondé de forma directa y concisa.
            - Mencioná los valores numéricos relevantes.
            - No expliques cómo se obtuvo la información.
            - No menciones SQL ni consultas.
            '''
        
        answer = call_llm_simple(system, prompt)
        result["answer"] = answer
        result["success"] = True
        result["has_data"] = True
        
    except Exception as e:
        result["success"] = True
        result["has_data"] = True
        result["error_type"] = "answer_generation"
        result["error_message"] = str(e)[:200] if len(str(e)) > 200 else str(e)
    
    return result


def is_evidence_usable(evidence: Dict[str, Any]) -> bool:
    """
    Helper para determinar si la evidencia SQL es utilizable.
    
    Retorna True SOLO si:
    - evidence no es None
    - success es True
    - has_data es True  
    - answer no es None ni vacío
    """
    if evidence is None:
        return False
    
    if not isinstance(evidence, dict):
        return False
    
    success = evidence.get("success", False)
    has_data = evidence.get("has_data", False)
    answer = evidence.get("answer")
    
    # Verificar que answer existe y no está vacío
    has_valid_answer = answer is not None and str(answer).strip() != ""
    
    return success and has_data and has_valid_answer


# =========================================================================
# FUNCIÓN ORIGINAL (mantenida por compatibilidad)
# =========================================================================

def answer_sql(question: str) -> str:
    """Genera y ejecuta SQL para responder una pregunta, retorna respuesta en lenguaje natural."""
    schema_txt = get_schema_text()
    sql = text_to_sql(question, TABLE_, schema_txt)

    print('---SQL GENERADA---')
    print(sql)
    print('------------------')

    r = run_sql_preview(sql, n=20)
    if not r['ok']:
        return f'No pude ejecutar la SQL generada.\nError: {r["error"]}\nSQL:\n{sql}'

    if not r['has_rows']:
        return f'La SQL se ejecutó pero no devolvió filas.\nSQL:\n{sql}'

    system = 'Sos un asistente de datos. Respondé en español. No inventes. Si falta info, decilo.'
    prompt = f'''
        Pregunta: {question}

        Evidencia SQL [T1]:
        SQL:
        {sql}

        Resultado:
        {r["preview"]}

        Instrucciones:
        - Usá [T1] para justificar números.
        - No inventes nada fuera del resultado.
        '''
    
    return call_llm_simple(system, prompt)

def build_sql_response(query: str, sql_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye el dict de respuesta para el caso de SQL early exit.
    
    Args:
        query: La pregunta original del usuario
        sql_evidence: El resultado de get_sql_evidence()
        
    Returns:
        Dict con la estructura estándar de answer_with_rag
    """
    return {
        "query": query,
        "answer": sql_evidence["answer"],
        "evidence": {
            "answerable": True,
            "source": sql_evidence["source"],
            "source_type": "structured_sql",
            "raw_data": sql_evidence["raw_data"],
            "key_points": [sql_evidence["answer"]],
            "missing": [],
        },
        "citations": [{
            "source": sql_evidence["source"],
            "source_type": "tabla_excel",
            "content": sql_evidence["raw_data"],
        }],
        "retrieved_candidates": [],  # No se usó retrieval
        "reranked_hits": [],          # No se usó reranking
        "response_source": "sql_table",  # Indicador de origen
    }