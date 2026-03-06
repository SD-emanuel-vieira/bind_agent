"""
Módulo para registrar cada ejecución del RAG en una tabla Delta.
Crea la tabla automáticamente si no existe.

Logging asíncrono: las escrituras se encolan y un hilo daemon las
procesa en background, sin bloquear la respuesta al usuario.

Compatible con Model Serving (donde pyspark no está disponible):
todos los imports de pyspark son lazy y el writer thread solo se
inicia si hay una SparkSession activa.
"""
import atexit
import json
import threading
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, Any, Optional
from bind_rag_agent.config import DELTA_LOG_TABLE

# ---------------------------------------------------------------------------
# Estado a nivel de módulo
# ---------------------------------------------------------------------------
_table_checked: bool = False
_table_check_lock = threading.Lock()

_LOG_SCHEMA = None  # se inicializa lazy en _get_log_schema()

# Cola y worker
_log_queue: Queue = Queue()
_SENTINEL = object()  # señal de shutdown

_writer_thread: Optional[threading.Thread] = None
_writer_started = False
_writer_start_lock = threading.Lock()


# ---------------------------------------------------------------------------
# PySpark availability check
# ---------------------------------------------------------------------------
def _pyspark_available() -> bool:
    """Retorna True si pyspark está instalado."""
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_log_schema():
    """Construye el schema lazy (solo cuando pyspark está disponible)."""
    global _LOG_SCHEMA
    if _LOG_SCHEMA is not None:
        return _LOG_SCHEMA

    from pyspark.sql.types import (
        StructType, StructField, StringType, LongType, TimestampType,
    )
    _LOG_SCHEMA = StructType([
        StructField("query",                StringType(),    True),
        StructField("answer",               StringType(),    True),
        StructField("evidence",             StringType(),    True),
        StructField("citations",            StringType(),    True),
        StructField("retrieved_candidates", StringType(),    True),
        StructField("reranked_hits",        StringType(),    True),
        StructField("timestamp",            TimestampType(), True),
        StructField("input_tokens",         LongType(),      True),
        StructField("output_tokens",        LongType(),      True),
        StructField("response_source",      StringType(),    True),
    ])
    return _LOG_SCHEMA


def _get_spark():
    """Obtiene la SparkSession activa en Databricks."""
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession()
    except ImportError:
        return None


def _ensure_table_exists(spark) -> None:
    """Crea la tabla Delta una sola vez por ciclo de vida del proceso."""
    global _table_checked
    if _table_checked:
        return
    with _table_check_lock:
        if _table_checked:          # double-check bajo lock
            return
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {DELTA_LOG_TABLE} (
                query                STRING,
                answer               STRING,
                evidence             STRING,
                citations            STRING,
                retrieved_candidates STRING,
                reranked_hits        STRING,
                timestamp            TIMESTAMP,
                input_tokens         BIGINT,
                output_tokens        BIGINT,
                response_source      STRING
            )
            USING DELTA
        """)
        _table_checked = True


def _serialize(value) -> str:
    """Serializa un valor a JSON string de forma segura."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _build_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Construye el dict de la fila a insertar (se ejecuta en el caller
    para capturar el timestamp exacto de la respuesta)."""
    return {
        "query":                result.get("query", ""),
        "answer":               result.get("answer", ""),
        "evidence":             _serialize(result.get("evidence", "")),
        "citations":            _serialize(result.get("citations", [])),
        "retrieved_candidates": _serialize(result.get("retrieved_candidates", [])),
        "reranked_hits":        _serialize(result.get("reranked_hits", [])),
        "timestamp":            datetime.now(),
        "input_tokens":         int(result.get("total_input_tokens", 0)),
        "output_tokens":        int(result.get("total_output_tokens", 0)),
        "response_source":      result.get("response_source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Worker de escritura (hilo daemon)
# ---------------------------------------------------------------------------
def _writer_loop() -> None:
    """Consume filas de la cola y las escribe en Delta.
    Se detiene al recibir el sentinel o cuando el proceso termina."""
    while True:
        try:
            item = _log_queue.get(timeout=5)
        except Empty:
            continue

        if item is _SENTINEL:
            _log_queue.task_done()
            break

        try:
            spark = _get_spark()
            if spark is None:
                print("[delta_logger] No hay SparkSession activa, descartando fila.")
                continue

            _ensure_table_exists(spark)

            schema = _get_log_schema()
            df = spark.createDataFrame([item], schema=schema)
            df.writeTo(DELTA_LOG_TABLE).append()

        except Exception as e:
            print(f"[delta_logger] Error al escribir en Delta: {e}")
        finally:
            _log_queue.task_done()


def _ensure_writer_started() -> None:
    """Inicia el writer thread solo la primera vez que se necesita
    y solo si pyspark está disponible."""
    global _writer_thread, _writer_started
    if _writer_started:
        return
    with _writer_start_lock:
        if _writer_started:
            return
        if not _pyspark_available():
            print("[delta_logger] pyspark no disponible, logging a Delta deshabilitado.")
            _writer_started = True  # no reintentar
            return
        _writer_thread = threading.Thread(
            target=_writer_loop, daemon=True, name="delta-log-writer"
        )
        _writer_thread.start()
        atexit.register(_shutdown_writer)
        _writer_started = True


def _shutdown_writer() -> None:
    """Intenta vaciar la cola antes de que el proceso muera."""
    if _writer_thread is not None and _writer_thread.is_alive():
        _log_queue.put(_SENTINEL)
        _writer_thread.join(timeout=30)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def log_rag_to_delta(result: Dict[str, Any]) -> None:
    """
    Encola una fila para escritura asíncrona en la tabla Delta.

    La serialización del resultado se hace de forma sincrónica (es barata)
    para capturar el timestamp exacto; la escritura a Delta ocurre en
    background sin bloquear la respuesta al usuario.

    En entornos sin pyspark (e.g. Model Serving), la llamada es un no-op
    silencioso.

    Parameters
    ----------
    result : dict
        El diccionario que devuelve answer_with_rag().
    """
    try:
        _ensure_writer_started()
        if _writer_thread is None or not _writer_thread.is_alive():
            return  # no hay writer (sin pyspark), skip silencioso
        row = _build_row(result)
        _log_queue.put_nowait(row)
    except Exception as e:
        print(f"[delta_logger] Error al encolar log: {e}")