"""
RAG Smart Routing para BIND
===========================

Reglas de routing:
- SQL SOLO para consultas sobre "resultado neto" o "resultado bruto"
- SQL SOLO puede filtrar por: oficial, producto, cliente
- TODO LO DEMÁS → Base vectorial
"""

import re
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# REGLAS DE NEGOCIO BIND
# =============================================================================

# Métricas permitidas en SQL (únicas que se pueden consultar vía SQL)
SQL_ALLOWED_METRICS = {
    "resultado neto": ["resultado_neto", "resultado_neto_iibb", "res_neto"],
    "resultado bruto": ["resultado_bruto", "res_bruto"],
    "IIBB": ["iibb", "impuesto a las ganancias"],
}

# Dimensiones permitidas en SQL (únicas por las que se puede filtrar)
SQL_ALLOWED_DIMENSIONS = {
    "oficial",
    "producto", 
    "cliente",
    "sub_producto",
    "moneda",
}

# Alias comunes que los usuarios pueden usar
METRIC_ALIASES = {
    "resultado neto": ["resultado neto", "res neto", "neto"],
    "resultado bruto": ["resultado bruto", "res bruto", "bruto"],
}

# Términos que SIEMPRE van a vector (nunca SQL)
VECTOR_ONLY_TERMS = [
    # Otras métricas financieras
    "previsiones", "prevision", "provision",
    "resultado comercial",
    "resultado gestion",
    "roe", "roa",
    "tna", "tasa",
    "margen",
    "ingresos", "gastos",
    # "activos", "pasivos", 
    "patrimonio",
    "axi", "ajuste",
    "waiver",
    "presupuesto", "presupuestado",
    
    # Términos contextuales/evolutivos
    "evolución", "evolucion",
    "tendencia",
    # "comparar", "comparación", "comparacion",
    "histórico", "historico",
    "acumulado",
    # "variación", "variacion",
    "diferencia",
    
    # Segmentos (no son oficial/producto/cliente)
    "segmento",
    "empresas",
    "corporate",
    "leasing" #- puede ser un valor válido de producto, no bloquear
    "retail",
    "pyme",
]

# =============================================================================
# FUNCIONES DE DETECCIÓN
# =============================================================================

def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    return text.lower().strip()


def detect_allowed_metric(question: str) -> Optional[str]:
    """
    Detecta si la pregunta es sobre una métrica permitida en SQL.
    Returns: nombre de la métrica si está permitida, None si no.
    """
    q = normalize_text(question)
    
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return metric
    
    return None


def detect_vector_only_terms(question: str) -> List[str]:
    """
    Detecta términos que fuerzan búsqueda vectorial.
    Returns: lista de términos encontrados que requieren vector.
    """
    q = normalize_text(question)
    found = []
    
    for term in VECTOR_ONLY_TERMS:
        if term in q:
            found.append(term)
    
    return found


def detect_dimensions(question: str) -> List[str]:
    """
    Detecta dimensiones mencionadas en la pregunta.
    Returns: lista de dimensiones encontradas.
    """
    q = normalize_text(question)
    found = []
    
    for dim in SQL_ALLOWED_DIMENSIONS:
        if dim in q:
            found.append(dim)
    
    return found


def extract_sql_select_fields(sql_query: str) -> List[str]:
    """Extrae los campos del SELECT de una query SQL."""
    if not sql_query:
        return []
    
    # Encontrar parte entre SELECT y FROM
    match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    
    select_clause = match.group(1)
    fields = []
    
    # Extraer campos de funciones agregadas
    agg_pattern = r'(?:SUM|AVG|COUNT|MIN|MAX)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
    for m in re.finditer(agg_pattern, sql_query, re.IGNORECASE):
        fields.append(m.group(1).lower())
    
    # Extraer campos simples
    for part in select_clause.split(','):
        part = part.strip()
        if '(' not in part and part.upper() != '*':
            field_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', part)
            if field_match:
                fields.append(field_match.group(1).lower())
    
    return list(set(fields))


def extract_sql_where_fields(sql_query: str) -> List[str]:
    """Extrae los campos usados en el WHERE de una query SQL."""
    if not sql_query:
        return []
    
    # Encontrar parte después de WHERE
    match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql_query, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    
    where_clause = match.group(1)
    fields = []
    
    # Buscar patrones: field = value, field IN (...), field LIKE ...
    patterns = [
        r'(\w+)\s*=',
        r'(\w+)\s+IN\s*\(',
        r'(\w+)\s+LIKE',
        r'(\w+)\s+BETWEEN',
        r'(\w+)\s*[<>]',
    ]
    
    for pattern in patterns:
        for m in re.finditer(pattern, where_clause, re.IGNORECASE):
            field = m.group(1).lower()
            # Ignorar campos de fecha que siempre están permitidos
            if field not in ['year', 'month', 'fecha', 'date']:
                fields.append(field)
    
    return list(set(fields))


# =============================================================================
# VALIDACIÓN PRINCIPAL
# =============================================================================

def validate_sql_for_bind(question: str, sql_query: str) -> Dict[str, Any]:
    """
    Valida si una query SQL es apropiada según las reglas de BIND.
    
    Reglas:
    1. La pregunta debe ser sobre "resultado neto" o "resultado bruto"
    2. Los filtros deben ser solo por oficial, producto, o cliente (+ fecha)
    3. Si hay términos que requieren vector, rechazar SQL
    
    Returns:
        {
            "is_valid": bool,
            "reason": str,
            "detected_metric": str or None,
            "vector_terms_found": list,
            "dimensions_found": list,
            "sql_select_fields": list,
            "sql_where_fields": list,
        }
    """
    # 1. Detectar si hay términos que fuerzan vector
    vector_terms = detect_vector_only_terms(question)
    if vector_terms:
        return {
            "is_valid": False,
            "reason": f"La pregunta contiene términos que requieren búsqueda documental: {vector_terms}",
            "detected_metric": None,
            "vector_terms_found": vector_terms,
            "dimensions_found": [],
            "sql_select_fields": extract_sql_select_fields(sql_query),
            "sql_where_fields": extract_sql_where_fields(sql_query),
        }
    
    # 2. Detectar si la pregunta es sobre una métrica permitida
    allowed_metric = detect_allowed_metric(question)
    if not allowed_metric:
        return {
            "is_valid": False,
            "reason": "La pregunta no es sobre 'resultado neto' ni 'resultado bruto'. Usar búsqueda vectorial.",
            "detected_metric": None,
            "vector_terms_found": [],
            "dimensions_found": detect_dimensions(question),
            "sql_select_fields": extract_sql_select_fields(sql_query),
            "sql_where_fields": extract_sql_where_fields(sql_query),
        }
    
    # 3. Validar que la query SQL use los campos correctos
    sql_select = extract_sql_select_fields(sql_query)
    sql_where = extract_sql_where_fields(sql_query)
    
    # Verificar que el SELECT traiga campos de la métrica permitida
    valid_select_fields = SQL_ALLOWED_METRICS.get(allowed_metric, [])
    select_ok = any(f in valid_select_fields or any(vf in f for vf in valid_select_fields) 
                    for f in sql_select)
    
    if not select_ok and sql_select:
        return {
            "is_valid": False,
            "reason": f"La query selecciona '{sql_select}' pero debería traer campos de '{allowed_metric}': {valid_select_fields}",
            "detected_metric": allowed_metric,
            "vector_terms_found": [],
            "dimensions_found": detect_dimensions(question),
            "sql_select_fields": sql_select,
            "sql_where_fields": sql_where,
        }
    
    # 4. Verificar que los filtros WHERE sean solo dimensiones permitidas
    invalid_filters = [f for f in sql_where if f not in SQL_ALLOWED_DIMENSIONS]
    if invalid_filters:
        return {
            "is_valid": False,
            "reason": f"La query filtra por '{invalid_filters}' pero solo se permite filtrar por: {list(SQL_ALLOWED_DIMENSIONS)}",
            "detected_metric": allowed_metric,
            "vector_terms_found": [],
            "dimensions_found": detect_dimensions(question),
            "sql_select_fields": sql_select,
            "sql_where_fields": sql_where,
        }
    
    # 5. Todo OK
    return {
        "is_valid": True,
        "reason": f"Query válida para '{allowed_metric}' con filtros permitidos",
        "detected_metric": allowed_metric,
        "vector_terms_found": [],
        "dimensions_found": detect_dimensions(question),
        "sql_select_fields": sql_select,
        "sql_where_fields": sql_where,
    }


# =============================================================================
# FUNCIÓN PRINCIPAL DE ROUTING
# =============================================================================

def validate_and_route(
    question: str,
    sql_query: Optional[str] = None,
    sql_result: Any = None,
    sql_error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Función principal para validar y decidir el routing.
    
    Args:
        question: Pregunta del usuario
        sql_query: Query SQL generada (puede ser None)
        sql_result: Resultado de ejecutar la query
        sql_error: Error si la query falló
    
    Returns:
        {
            "use_sql": bool,
            "use_vector": bool,
            "strategy": "sql_only" | "vector_only",
            "reason": str,
            "validation": dict
        }
    """
    # Caso 1: No hay query SQL
    if not sql_query:
        return {
            "use_sql": False,
            "use_vector": True,
            "strategy": "vector_only",
            "reason": "No se generó query SQL",
            "validation": None
        }
    
    # Caso 2: Error en SQL
    if sql_error:
        return {
            "use_sql": False,
            "use_vector": True,
            "strategy": "vector_only",
            "reason": f"Error en SQL: {sql_error}",
            "validation": None
        }
    
    # Caso 3: Validar la query
    validation = validate_sql_for_bind(question, sql_query)
    
    # Caso 4: SQL inválido
    if not validation["is_valid"]:
        return {
            "use_sql": False,
            "use_vector": True,
            "strategy": "vector_only",
            "reason": validation["reason"],
            "validation": validation
        }
    
    # Caso 5: SQL válido pero sin resultados
    sql_empty = sql_result is None or (hasattr(sql_result, '__len__') and len(sql_result) == 0)
    if sql_empty:
        return {
            "use_sql": False,
            "use_vector": True,
            "strategy": "vector_only",
            "reason": "Query SQL válida pero sin resultados, buscando en documentos",
            "validation": validation
        }
    
    # Caso 6: SQL válido con resultados
    return {
        "use_sql": True,
        "use_vector": False,
        "strategy": "sql_only",
        "reason": validation["reason"],
        "validation": validation
    }


# =============================================================================
# FUNCIÓN HELPER PARA DECIDIR ANTES DE GENERAR SQL
# =============================================================================

def has_detail_dimension(question: str) -> Tuple[bool, List[str]]:
    """
    Verifica si la pregunta menciona al menos una dimensión de detalle.
    SQL solo aplica para consultas a nivel de detalle (cliente, producto, oficial).
    
    Returns:
        (has_dimension: bool, dimensions_found: list)
    """
    q = normalize_text(question)
    found = []
    
    # Patrones para detectar menciones de dimensiones
    # Incluye variaciones comunes
    dimension_patterns = {
        "cliente": ["cliente", "clientes", "santander", "bbva", "galicia"],  # Agregar nombres de clientes conocidos
        "producto": ["producto", "productos"],
        "oficial": ["oficial", "oficiales"],
    }
    
    for dim, patterns in dimension_patterns.items():
        for pattern in patterns:
            if pattern in q:
                found.append(dim)
                break  # No duplicar la misma dimensión
    
    return len(found) > 0, found


def should_try_sql(question: str) -> Tuple[bool, str]:
    """
    Decide rápidamente si vale la pena intentar SQL, ANTES de generarlo.
    
    Reglas:
    1. No debe tener términos que fuercen vector (previsiones, evolución, etc.)
    2. Debe ser sobre 'resultado neto' o 'resultado bruto'
    3. NUEVO: Debe mencionar al menos una dimensión de detalle (cliente, producto, oficial)
    
    Returns:
        (should_try: bool, reason: str)
    """
    # Verificar términos que fuerzan vector
    vector_terms = detect_vector_only_terms(question)
    if vector_terms:
        return False, f"Términos que requieren documentos: {vector_terms}"
    
    # Verificar si es sobre métrica permitida
    metric = detect_allowed_metric(question)
    if not metric:
        return False, "No es sobre 'resultado neto' ni 'resultado bruto'"
    
    # NUEVO: Verificar que mencione al menos una dimensión de detalle
    has_dimension, dimensions = has_detail_dimension(question)
    if not has_dimension:
        return False, f"Pregunta sobre '{metric}' pero sin dimensión de detalle (cliente/producto/oficial). Usar vector para datos agregados."
    
    return True, f"Pregunta sobre '{metric}' con dimensión [{', '.join(dimensions)}], intentar SQL"