"""
Módulo para registrar cada ejecución del RAG en una tabla Delta.
Crea la tabla automáticamente si no existe.

Logging asíncrono: las escrituras se encolan y un hilo daemon las
procesa en background, sin bloquear la respuesta al usuario.
"""
import atexit
import json
import threading
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, Any

from bind_rag_agent.config import DELTA_LOG_TABLE

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType,
)

# ---------------------------------------------------------------------------
# Estado a nivel de módulo
# ---------------------------------------------------------------------------
_table_checked: bool = False
_table_check_lock = threading.Lock()

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

# Cola y worker
_log_queue: Queue = Queue()
_SENTINEL = object()  # señal de shutdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_spark() -> SparkSession:
    """Obtiene la SparkSession activa en Databricks."""
    return SparkSession.getActiveSession()


def _ensure_table_exists(spark: SparkSession) -> None:
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

            df = spark.createDataFrame([item], schema=_LOG_SCHEMA)
            df.writeTo(DELTA_LOG_TABLE).append()

        except Exception as e:
            print(f"[delta_logger] Error al escribir en Delta: {e}")
        finally:
            _log_queue.task_done()


_writer_thread = threading.Thread(target=_writer_loop, daemon=True, name="delta-log-writer")
_writer_thread.start()


def _shutdown_writer() -> None:
    """Intenta vaciar la cola antes de que el proceso muera."""
    _log_queue.put(_SENTINEL)
    _writer_thread.join(timeout=30)


atexit.register(_shutdown_writer)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def log_rag_to_delta(result: Dict[str, Any]) -> None:
    """
    Encola una fila para escritura asíncrona en la tabla Delta.

    La serialización del resultado se hace de forma sincrónica (es barata)
    para capturar el timestamp exacto; la escritura a Delta ocurre en
    background sin bloquear la respuesta al usuario.

    Parameters
    ----------
    result : dict
        El diccionario que devuelve answer_with_rag().
    """
    try:
        row = _build_row(result)
        _log_queue.put_nowait(row)
    except Exception as e:
        print(f"[delta_logger] Error al encolar log: {e}")