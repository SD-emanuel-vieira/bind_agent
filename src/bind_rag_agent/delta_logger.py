"""
Módulo para registrar cada ejecución del RAG en una tabla Delta.
Crea la tabla automáticamente si no existe.
"""
import json
from datetime import datetime
from typing import Dict, Any
from bind_rag_agent.config import DELTA_LOG_TABLE

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType
)


def _get_spark() -> SparkSession:
    """Obtiene la SparkSession activa en Databricks."""
    return SparkSession.getActiveSession()


def _ensure_table_exists(spark: SparkSession) -> None:
    """Crea la tabla Delta si no existe."""
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


def _serialize(value) -> str:
    """Serializa un valor a JSON string de forma segura."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def log_rag_to_delta(result: Dict[str, Any]) -> None:
    """
    Inserta una fila en la tabla Delta con los datos de la ejecución.
    Crea la tabla automáticamente en la primera invocación.

    Parameters
    ----------
    result : dict
        El diccionario que devuelve answer_with_rag().
    """
    spark = _get_spark()
    if spark is None:
        print("[delta_logger] No hay SparkSession activa, skip logging.")
        return

    try:
        _ensure_table_exists(spark)

        row = {
            "query": result.get("query", ""),
            "answer": result.get("answer", ""),
            "evidence": _serialize(result.get("evidence", "")),
            "citations": _serialize(result.get("citations", [])),
            "retrieved_candidates": _serialize(result.get("retrieved_candidates", [])),
            "reranked_hits": _serialize(result.get("reranked_hits", [])),
            "timestamp": datetime.now(),
            "input_tokens": int(result.get("total_input_tokens", 0)),
            "output_tokens": int(result.get("total_output_tokens", 0)),
            "response_source": result.get("response_source", "unknown"),
        }

        schema = StructType([
            StructField("query", StringType(), True),
            StructField("answer", StringType(), True),
            StructField("evidence", StringType(), True),
            StructField("citations", StringType(), True),
            StructField("retrieved_candidates", StringType(), True),
            StructField("reranked_hits", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("input_tokens", LongType(), True),
            StructField("output_tokens", LongType(), True),
            StructField("response_source", StringType(), True),
        ])

        df = spark.createDataFrame([row], schema=schema)
        df.write.format("delta").mode("append").saveAsTable(DELTA_LOG_TABLE)

    except Exception as e:
        print(f"[delta_logger] Error al escribir en Delta: {e}")