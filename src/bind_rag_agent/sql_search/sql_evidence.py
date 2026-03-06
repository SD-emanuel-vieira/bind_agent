import os
import re
import logging
import mlflow
import json
from mlflow.deployments import get_deploy_client
from typing import List, Dict, Any
from contextlib import contextmanager
import time
from bind_rag_agent.config import (
    TABLE_, 
    LLM_ENDPOINT_SQL,
    SQL_RESULT_LIMIT,
    SQL_TEMPERATURE,
    SQL_MAX_TOKENS,
    SCHEMA_CACHE_TTL,
)
from bind_rag_agent.token_counter import token_counter

logger = logging.getLogger("sql_evidence")

# =========================================================================
# SQL Execution: SparkSession (notebook) → Statement Execution API (serving)
# =========================================================================

_SQL_WAREHOUSE_ID = os.getenv("RAG_SQL_WAREHOUSE_ID", "")
_spark_available = None  # None = no probado aún, True/False = resultado


def _get_warehouse_id() -> str:
    """Obtiene el warehouse ID desde env var o autodescubrimiento."""
    if _SQL_WAREHOUSE_ID:
        return _SQL_WAREHOUSE_ID
    
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        warehouses = list(w.warehouses.list())
        for wh in warehouses:
            if wh.state and wh.state.value == "RUNNING":
                return wh.id
        if warehouses:
            return warehouses[0].id
    except Exception as e:
        logger.warning(f"[sql_evidence] No se pudo autodescubrir warehouse: {e}")
    
    raise ValueError(
        "No se encontró RAG_SQL_WAREHOUSE_ID. "
        "Configurá la env var RAG_SQL_WAREHOUSE_ID con el ID del SQL Warehouse."
    )


def _try_spark():
    """Intenta usar SparkSession. Cachea el resultado para no reintentar."""
    global _spark_available
    
    if _spark_available is False:
        return None
    
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark.sql("SELECT 1").collect()  # verificar que funciona
        _spark_available = True
        return spark
    except Exception as e:
        logger.warning(f"[sql_evidence] SparkSession no disponible ({str(e)[:80]}), usando Statement Execution API")
        _spark_available = False
        return None


def _execute_sql_via_api(sql: str, max_rows: int = SQL_RESULT_LIMIT) -> Dict[str, Any]:
    """Ejecuta SQL via Statement Execution API (funciona en serving endpoints)."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        warehouse_id = _get_warehouse_id()
        
        logger.warning(f"[sql_evidence] API exec: warehouse={warehouse_id}, sql={sql[:120]}...")
        
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="30s",
        )
        
        # Verificar estado
        status = response.status
        if status and status.state:
            state_val = status.state.value if hasattr(status.state, 'value') else str(status.state)
            logger.warning(f"[sql_evidence] API response state: {state_val}")
            if state_val == "FAILED":
                error_msg = ""
                if status.error:
                    error_msg = getattr(status.error, 'message', str(status.error))
                logger.error(f"[sql_evidence] SQL FAILED: {error_msg}")
                return {
                    'ok': False, 'has_rows': False, 'preview': '',
                    'error': error_msg or "SQL execution failed"
                }
        
        # Extraer columnas y filas
        columns = []
        if response.manifest and response.manifest.schema and response.manifest.schema.columns:
            columns = [col.name for col in response.manifest.schema.columns]
        
        rows = []
        if response.result and response.result.data_array:
            rows = response.result.data_array[:max_rows]
        
        has_rows = len(rows) > 0
        
        # Generar preview en formato tabla (compatible con pandas.to_string)
        preview = ""
        if has_rows and columns:
            col_widths = [len(c) for c in columns]
            for row in rows:
                for i, val in enumerate(row):
                    if i < len(col_widths):
                        col_widths[i] = max(col_widths[i], len(str(val) if val is not None else "None"))
            
            header = "  ".join(str(c).rjust(w) for c, w in zip(columns, col_widths))
            preview = header + "\n"
            for row in rows:
                row_str = "  ".join(
                    str(v if v is not None else "None").rjust(w) 
                    for v, w in zip(row, col_widths)
                )
                preview += row_str + "\n"
            preview = preview.rstrip()
        
        return {'ok': True, 'has_rows': has_rows, 'preview': preview, 'error': None}
        
    except Exception as e:
        logger.error(f"[sql_evidence] API exception: {type(e).__name__}: {str(e)[:300]}")
        return {'ok': False, 'has_rows': False, 'preview': '', 'error': str(e)[:200]}


def _get_schema_via_api(table: str) -> str:
    """Obtiene el schema via Statement Execution API."""
    # --- DEBUG: identificar identidad del endpoint ---
    try:
        from databricks.sdk import WorkspaceClient
        _w = WorkspaceClient()
        logger.warning(f"[sql_evidence] Auth: type={_w.config.auth_type}, host={_w.config.host}")
        try:
            me = _w.current_user.me()
            logger.warning(f"[sql_evidence] Identity: {me.user_name} (id={me.id})")
        except Exception as e2:
            logger.warning(f"[sql_evidence] current_user.me() failed: {e2}")
    except Exception as e:
        logger.warning(f"[sql_evidence] WorkspaceClient failed: {e}")
    # --- FIN DEBUG ---

    result = _execute_sql_via_api(f"DESCRIBE TABLE {table}")
    if not result['ok']:
        logger.error(f"[sql_evidence] DESCRIBE TABLE failed: {result['error']}")
        raise ValueError(f"No se pudo obtener schema de {table}: {result['error']}")
    
    # Parsear el preview (formato tabla) para extraer col_name y data_type
    lines = []
    for line in result['preview'].strip().splitlines()[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 2 and not parts[0].startswith("#"):
            lines.append(f"- {parts[0]}: {parts[1]}")
    
    return "\n".join(lines)


# =========================================================================
# Funciones públicas (con fallback automático)
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
        _logger = logging.getLogger(logger_name)
        original_levels[logger_name] = _logger.level
        _logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def schema_text(table: str) -> str:
    """Obtiene schema. Intenta Spark primero, fallback a API."""
    spark = _try_spark()
    if spark:
        try:
            fields = spark.table(table).schema.fields
            return "\n".join([f"- {f.name}: {f.dataType.simpleString()}" for f in fields])
        except Exception:
            pass
    return _get_schema_via_api(table)


_SCHEMA_CACHE = {"text": None, "timestamp": 0}

def get_schema_text() -> str:
    global _SCHEMA_CACHE
    now = time.time()
    
    if _SCHEMA_CACHE["text"] is None or (now - _SCHEMA_CACHE["timestamp"]) > SCHEMA_CACHE_TTL:
        _SCHEMA_CACHE["text"] = schema_text(TABLE_)
        _SCHEMA_CACHE["timestamp"] = now
    
    return _SCHEMA_CACHE["text"]


def sql_tool(sql: str, n: int = SQL_RESULT_LIMIT) -> str:
    result = run_sql_preview(sql, n)
    if result['ok']:
        return result['preview']
    raise Exception(f"SQL execution failed: {result['error']}")


def run_sql_preview(sql: str, n: int = SQL_RESULT_LIMIT) -> dict:
    """Ejecuta SQL. Intenta Spark primero, fallback a Statement Execution API."""
    spark = _try_spark()
    if spark:
        with suppress_spark_errors():
            try:
                df = spark.sql(sql)
                has_rows = df.limit(1).count() > 0
                preview = df.limit(n).toPandas().to_string(index=False)
                return {'ok': True, 'has_rows': has_rows, 'preview': preview, 'error': None}
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                logger.warning(f"[sql_evidence] Spark SQL falló, intentando API: {error_msg[:100]}")
    
    # Fallback: Statement Execution API
    logger.warning(f"[sql_evidence] Usando Statement Execution API para ejecutar SQL")
    return _execute_sql_via_api(sql, max_rows=n)

    
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


def call_llm(endpoint: str, messages: List[Dict[str, str]], temperature: float = SQL_TEMPERATURE, max_tokens: int = SQL_MAX_TOKENS) -> str:
    payload = {'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
    client = get_deploy_client('databricks')
    resp = client.predict(endpoint=endpoint, inputs=payload)
    token_counter.add_from_response(resp)
    return extract_chat_content(resp)

def call_llm_simple(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    return call_llm(LLM_ENDPOINT_SQL, messages, temperature=SQL_TEMPERATURE, max_tokens=SQL_MAX_TOKENS)

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

def enforce_limit(sql: str, n: int = SQL_RESULT_LIMIT) -> str:
    s = sql.strip().rstrip(';')
    if 'limit' not in s.lower():
        return s + f'\nLIMIT {n}'
    return s

FORBIDDEN_PATTERNS = [
    r'\bdrop\b', r'\bdelete\b', r'\bupdate\b', r'\binsert\b', 
    r'\balter\b', r'\btruncate\b', r'\bcreate\b', r'\bgrant\b',
    r'--', r'/\*', r'\*/', r';(?!$)'
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
    system = '''Sos experto en SQL Spark (Databricks). Devolvé SOLO SQL válido. Nada de explicación.
    IMPORTANTE: Usá funciones de Spark SQL en INGLÉS (contains, lower, trim, etc). NO uses funciones en español.'''
        
    user = f"""
        Generá una query SQL Spark para responder la pregunta usando SOLO esta tabla: {table}

        Esquema:
        {schema_txt}

        MAPEO DE NEGOCIO (usar siempre estas equivalencias):
        - "ingresos", "ingresos netos", "ingresos del cliente" → columna: resultado_neto_iibb
        - "ingresos brutos" → columna: resultado_bruto
        - "resultado neto", "res neto" → columna: resultado_neto_iibb
        - "resultado bruto", "res bruto" → columna: resultado_bruto
        Ejemplo:
        Pregunta: "Dame los ingresos del cliente X para julio 2025"
        SQL: SELECT moneda, SUM(resultado_neto_iibb) AS ingresos_netos FROM {table} WHERE contains(lower(cliente), lower('X')) AND year = 2025 AND month = 7 GROUP BY moneda

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
    sql = enforce_limit(sql, SQL_RESULT_LIMIT)                  
    validate_sql(sql, table)
    return sql


# =========================================================================
# FUNCIÓN PRINCIPAL: Evidencia estructurada para integración con RAG
# =========================================================================
def get_sql_evidence(question: str) -> Dict[str, Any]:
    """
    Genera evidencia estructurada a partir de una query SQL.
    Nunca lanza excepciones - siempre retorna un dict con el estado.
    """
    result = {
        "success": False,
        "has_data": False,
        "answer": None,
        "raw_data": None,
        "source": "tabla_excel_financiera",
        "query": None,
        "error_type": None,
        "error_message": None
    }
    
    # Paso 1: Generar SQL
    try:
        schema_txt = get_schema_text()
        sql = text_to_sql(question, TABLE_, schema_txt)
        result["query"] = sql
        logger.warning(f"[sql_evidence] SQL generado: {sql[:200]}")
    except Exception as e:
        result["error_type"] = "sql_generation"
        result["error_message"] = str(e)[:200] if len(str(e)) > 200 else str(e)
        logger.error(f"[sql_evidence] SQL generation FAILED: {result['error_message']}")
        return result
    
    # Paso 2: Ejecutar SQL
    r = run_sql_preview(sql, n=SQL_RESULT_LIMIT)
    
    if not r['ok']:
        result["error_type"] = "sql_execution"
        result["error_message"] = r['error']
        logger.error(f"[sql_evidence] SQL execution FAILED: {r['error']}")
        return result
    
    if not r['has_rows']:
        result["success"] = True
        result["error_type"] = "no_rows"
        result["error_message"] = "La consulta no retornó resultados"
        logger.warning(f"[sql_evidence] SQL OK pero sin filas")
        return result
    
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
        logger.warning(f"[sql_evidence] SQL evidence OK: has_data=True, answer_len={len(answer or '')}")
        
    except Exception as e:
        result["success"] = True
        result["has_data"] = True
        result["error_type"] = "answer_generation"
        result["error_message"] = str(e)[:200] if len(str(e)) > 200 else str(e)
    
    return result


def is_evidence_usable(evidence: Dict[str, Any]) -> bool:
    if evidence is None:
        return False
    if not isinstance(evidence, dict):
        return False
    
    success = evidence.get("success", False)
    has_data = evidence.get("has_data", False)
    answer = evidence.get("answer")
    has_valid_answer = answer is not None and str(answer).strip() != ""
    
    return success and has_data and has_valid_answer


# =========================================================================
# FUNCIÓN ORIGINAL (mantenida por compatibilidad)
# =========================================================================

def answer_sql(question: str) -> str:
    schema_txt = get_schema_text()
    sql = text_to_sql(question, TABLE_, schema_txt)

    logger.warning('---SQL GENERADA---')
    logger.warning(sql)
    logger.warning('------------------')

    r = run_sql_preview(sql, n=SQL_RESULT_LIMIT)
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
        "retrieved_candidates": [],
        "reranked_hits": [],
        "response_source": "sql_table",
    }