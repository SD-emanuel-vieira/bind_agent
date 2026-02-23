# =============================================================================
# REGLAS DE NEGOCIO BIND
# =============================================================================

# Métricas permitidas en SQL (únicas que se pueden consultar vía SQL)
SQL_ALLOWED_METRICS = {
    "resultado neto": ["resultado_neto", "resultado_neto_iibb", "res_neto"],
    "resultado bruto": ["resultado_bruto", "res_bruto", "ingresos brutos"],
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
    "resultado neto": ["resultado neto", "res neto", "ingresos netos", "ingresos del cliente", "ingresos netos del cliente"],
    "resultado bruto": ["resultado bruto", "res bruto", "ingresos brutos del cliente"],
}

# Términos que SIEMPRE van a vector (nunca SQL)
VECTOR_ONLY_TERMS = [
    # Otras métricas financieras
    "previsiones", "prevision",
    "resultado comercial",
    "resultado gestion",
    "roe", "roa",
    "tna", "tasa",
    "margen",
    # "ingresos", "gastos", #revisar
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
    "leasing", #- puede ser un valor válido de producto, no bloquear
    "retail",
    "pyme",
]