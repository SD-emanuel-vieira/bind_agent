import os
from typing import Dict, Any, Optional

# ============================================================
# SQL DEBUGGING / TRACING
# ============================================================

DEBUG_TRACE_SQL = os.getenv("RAG_DEBUG_TRACE_SQL", "0") == "1"


def trace_sql(
    stage: str,
    *,
    query: Optional[str] = None,
    metric: Optional[str] = None,
    dimensions: Optional[list] = None,
    skip_reason: Optional[str] = None,
    sql: Optional[str] = None,
    sql_result: Optional[Dict[str, Any]] = None,
    sql_evidence: Optional[Dict[str, Any]] = None,
    routing: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Tracer para debuggear el flujo SQL completo.

    Solo se ejecuta si RAG_DEBUG_TRACE_SQL=1.

    Args:
        stage: Nombre de la etapa (ej: "1) should_try_sql")
        query: Query del usuario
        metric: Métrica detectada
        dimensions: Dimensiones detectadas
        skip_reason: Razón por la que se saltó SQL
        sql: Query SQL generada
        sql_result: Resultado de run_sql_preview
        sql_evidence: Resultado de get_sql_evidence
        routing: Resultado de validate_and_route
        extra: Cualquier dato adicional para debug
    """
    if not DEBUG_TRACE_SQL:
        return

    print("\n" + "=" * 90)
    print(f"[SQL TRACE] {stage}")
    print("-" * 90)

    if query is not None:
        print(f"  query: {query}")

    if metric is not None:
        print(f"  metric: {metric}")

    if dimensions is not None:
        print(f"  dimensions: {dimensions}")

    if skip_reason is not None:
        print(f"  skip_reason: {skip_reason}")

    if sql is not None:
        print(f"  sql_generated:")
        for line in sql.strip().splitlines():
            print(f"    {line}")

    if sql_result is not None:
        print(f"  execution: ok={sql_result.get('ok')} | has_rows={sql_result.get('has_rows')}")
        if sql_result.get('error'):
            print(f"  exec_error: {sql_result['error']}")
        if sql_result.get('preview'):
            preview_lines = sql_result['preview'].strip().splitlines()
            max_preview = 5
            for line in preview_lines[:max_preview]:
                print(f"    {line}")
            if len(preview_lines) > max_preview:
                print(f"    ... ({len(preview_lines) - max_preview} filas más)")

    if sql_evidence is not None:
        print(
            f"  evidence: success={sql_evidence.get('success')} | "
            f"has_data={sql_evidence.get('has_data')} | "
            f"error_type={sql_evidence.get('error_type')} | "
            f"error_msg={sql_evidence.get('error_message')}"
        )
        if sql_evidence.get('raw_data'):
            data_lines = sql_evidence['raw_data'].strip().splitlines()
            max_data = 5
            for line in data_lines[:max_data]:
                print(f"    {line}")
            if len(data_lines) > max_data:
                print(f"    ... ({len(data_lines) - max_data} filas más)")

    if routing is not None:
        print(
            f"  routing: use_sql={routing.get('use_sql')} | "
            f"strategy={routing.get('strategy')} | "
            f"reason={routing.get('reason')}"
        )
        validation = routing.get("validation")
        if validation:
            print(
                f"  validation: is_valid={validation.get('is_valid')} | "
                f"select_fields={validation.get('sql_select_fields')} | "
                f"where_fields={validation.get('sql_where_fields')}"
            )

    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")

    print("=" * 90)
