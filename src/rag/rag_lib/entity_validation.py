"""
entity_validation.py - Validación de entidades críticas en evidencias

OBJETIVO:
Detectar cuando la query pide información sobre una entidad específica (cliente, empresa)
y validar que esa entidad REALMENTE aparezca en los chunks recuperados.

Esto previene el problema de "asociación espuria" donde el LLM encuentra una métrica
(como "Resultado Neto") y la asocia incorrectamente a una entidad (como "Grimoldi")
que no aparece en ninguna evidencia.
"""

import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple, Optional
from rag_lib.config import SEGMENTS, SEGMENT_ALIASES

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Palabras que NO son entidades críticas (stop words extendidas para este contexto)
_ENTITY_STOP_WORDS = {
    # Stop words generales
    "cual", "cuál", "cuales", "cuáles", "como", "cómo", 
    "que", "qué", "quien", "quién", "donde", "dónde", "cuando", "cuándo",
    "son", "es", "fue", "fueron", "ser", "estar", "sido", "siendo",
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "para", "por", "en", "con", "sin", "sobre", "entre", "hacia",
    "y", "o", "ni", "pero", "sino", "aunque",
    "se", "le", "lo", "les", "nos", "me", "te",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
    "al", "a", "ha", "han", "hay", "he", "has",
    "muy", "mas", "más", "menos", "tan", "tanto", "mucho", "poco",
    "si", "no", "ya", "aun", "todavia", "tambien", "solo", "sólo",
    # Meses
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    # Términos financieros genéricos (NO son entidades)
    "resultado", "resultados", "neto", "netos", "operativo", "operativos",
    "ingreso", "ingresos", "egreso", "egresos", "gasto", "gastos",
    "activo", "activos", "pasivo", "pasivos", "patrimonio",
    "margen", "margenes", "financiero", "financiera", "financieros",
    "cliente", "clientes", "banco", "bancos", "banca",
    "prestamo", "prestamos", "préstamo", "préstamos",
    "deposito", "depositos", "depósito", "depósitos",
    "cartera", "mora", "morosidad", "provision", "provisiones",
    "capital", "fondeo", "liquidez", "solvencia", "roe", "roa",
    "ytd", "mom", "yoy", "mtd", "acumulado",
    "total", "totales", "promedio", "promedios",
    "mayor", "menor", "maximo", "minimo",
    # Años y períodos
    "2024", "2025", "2026", "año", "años", "mes", "meses", "trimestre",
    # Verbos comunes
    "dame", "mostrar", "muestra", "ver", "obtener", "calcular",
    "comparar", "analizar", "listar", "detallar",
}

# Patrones que indican que un token es parte de una entidad (cliente/empresa)
_ENTITY_CONTEXT_PATTERNS = [
    r"cliente\s+(\w+)",           # "cliente X"
    r"para\s+(?:el\s+)?(\w+)",    # "para X" o "para el X"
    r"de\s+(?:la\s+empresa\s+)?(\w+)",  # "de X" o "de la empresa X"
    r"empresa\s+(\w+)",           # "empresa X"
]

# ============================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================

def _norm(s: str) -> str:
    """Normaliza texto para comparación."""
    s = (s or "").strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def _get_segment_aliases() -> Set[str]:
    """Obtiene todos los aliases de segmentos normalizados."""
    aliases = set()
    for alias in SEGMENT_ALIASES.keys():
        aliases.add(_norm(alias))
    for seg in SEGMENTS:
        aliases.add(_norm(seg))
    return aliases


# ============================================================
# EXTRACCIÓN DE ENTIDADES CRÍTICAS
# ============================================================

def extract_critical_entities(query: str) -> List[str]:
    """
    Extrae entidades críticas de la query.
    
    Una "entidad crítica" es un nombre propio o identificador específico
    que DEBE aparecer en las evidencias para que la respuesta sea válida.
    
    Ejemplos:
    - "Dame el resultado neto para el cliente grimoldi" → ["grimoldi"]
    - "Ingresos de Banco Santander octubre 2025" → ["santander"]
    - "Resultado operativo octubre 2025" → [] (no hay entidad específica)
    
    Returns:
        Lista de entidades críticas normalizadas
    """
    qn = _norm(query)
    segment_aliases = _get_segment_aliases()
    
    entities = []
    
    # Método 1: Buscar patrones de contexto
    for pattern in _ENTITY_CONTEXT_PATTERNS:
        for match in re.finditer(pattern, qn):
            entity = match.group(1)
            if _is_valid_entity(entity, segment_aliases):
                entities.append(entity)
    
    # Método 2: Buscar tokens que parecen nombres propios
    # (palabras no-stop que no son métricas financieras ni segmentos)
    words = re.findall(r"\b[a-z]+\b", qn)
    
    for word in words:
        if _is_valid_entity(word, segment_aliases):
            # Verificar si parece un nombre propio (heurística)
            if _looks_like_proper_name(word, qn):
                entities.append(word)
    
    # Deduplicar
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    
    return unique


def _is_valid_entity(word: str, segment_aliases: Set[str]) -> bool:
    """
    Verifica si una palabra puede ser una entidad válida.
    """
    word = _norm(word)
    
    # Muy corta
    if len(word) < 3:
        return False
    
    # Es stop word
    if word in _ENTITY_STOP_WORDS:
        return False
    
    # Es un segmento conocido
    if word in segment_aliases:
        return False
    
    # Es solo números
    if word.isdigit():
        return False
    
    return True


def _looks_like_proper_name(word: str, query_context: str) -> bool:
    """
    Heurística para determinar si una palabra parece un nombre propio.
    
    Criterios:
    - No es una palabra común del español
    - Aparece después de "cliente", "para", "de la empresa", etc.
    - No es un término financiero conocido
    """
    word = _norm(word)
    
    # Si aparece después de patrones de contexto de entidad, es más probable que sea un nombre
    context_patterns = [
        rf"cliente\s+{re.escape(word)}",
        rf"para\s+(?:el\s+)?{re.escape(word)}",
        rf"empresa\s+{re.escape(word)}",
    ]
    
    for pattern in context_patterns:
        if re.search(pattern, query_context):
            return True
    
    # Heurística adicional: palabras que no son comunes en español financiero
    # y tienen cierta longitud
    if len(word) >= 5 and word not in _ENTITY_STOP_WORDS:
        # Verificar que no sea un término financiero común
        financial_terms = {
            "balance", "estado", "cuenta", "flujo", "caja",
            "intereses", "comisiones", "dividendos", "reservas",
            "spread", "spread", "funding", "trading",
        }
        if word not in financial_terms:
            return True
    
    return False


# ============================================================
# VALIDACIÓN DE ENTIDADES EN CHUNKS
# ============================================================

def validate_entities_in_chunks(
    query: str, 
    hits: List[Dict[str, Any]], 
    required_threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Valida que las entidades críticas de la query aparezcan en los chunks.
    
    Args:
        query: La pregunta del usuario
        hits: Lista de chunks recuperados
        required_threshold: Proporción mínima de chunks que deben contener la entidad
                          (0.0 = al menos 1 chunk debe tenerla)
    
    Returns:
        Dict con:
        - critical_entities: Lista de entidades críticas detectadas
        - entity_coverage: Dict[entity -> count de chunks que la contienen]
        - missing_entities: Lista de entidades que no aparecen en ningún chunk
        - is_valid: True si todas las entidades críticas aparecen en al menos 1 chunk
        - warning_message: Mensaje de advertencia si hay entidades faltantes
    """
    critical_entities = extract_critical_entities(query)
    
    if not critical_entities:
        return {
            "critical_entities": [],
            "entity_coverage": {},
            "missing_entities": [],
            "is_valid": True,
            "warning_message": None
        }
    
    # Contar en cuántos chunks aparece cada entidad
    entity_coverage = {entity: 0 for entity in critical_entities}
    
    for hit in hits:
        chunk_text = _norm(
            (hit.get("chunk_text_clean") or hit.get("chunk_text") or "") + " " +
            (hit.get("topic_heuristic") or "") + " " +
            (hit.get("topic_llm") or "") + " " +
            (hit.get("topic_content") or "")
        )
        
        for entity in critical_entities:
            # Buscar la entidad como palabra completa
            pattern = rf"\b{re.escape(entity)}\b"
            if re.search(pattern, chunk_text):
                entity_coverage[entity] += 1
    
    # Identificar entidades faltantes
    min_required = max(1, int(len(hits) * required_threshold))
    missing_entities = [
        entity for entity, count in entity_coverage.items() 
        if count < min_required
    ]
    
    is_valid = len(missing_entities) == 0
    
    warning_message = None
    if missing_entities:
        warning_message = (
            f"ADVERTENCIA: La(s) entidad(es) {missing_entities} no aparece(n) "
            f"en ninguna de las {len(hits)} evidencias recuperadas. "
            f"No es posible asociar valores a esta(s) entidad(es)."
        )
    
    return {
        "critical_entities": critical_entities,
        "entity_coverage": entity_coverage,
        "missing_entities": missing_entities,
        "is_valid": is_valid,
        "warning_message": warning_message
    }


def get_entity_validation_prompt_snippet(validation_result: Dict[str, Any]) -> str:
    """
    Genera un snippet para el prompt del LLM basado en la validación de entidades.
    
    Si hay entidades críticas faltantes, genera instrucciones explícitas
    para que el LLM NO asocie valores a esas entidades.
    """
    if not validation_result.get("critical_entities"):
        return ""
    
    missing = validation_result.get("missing_entities", [])
    
    if not missing:
        # Todas las entidades están presentes
        entities_str = ", ".join(validation_result["critical_entities"])
        return (
            f"\nENTIDADES CONFIRMADAS EN EVIDENCIA: {entities_str}\n"
            f"Puedes asociar valores a estas entidades si aparecen en el contexto.\n"
        )
    
    # Hay entidades faltantes - instrucciones estrictas
    missing_str = ", ".join(missing)
    return (
        f"\n⚠️ ADVERTENCIA CRÍTICA - ENTIDADES NO ENCONTRADAS:\n"
        f"La query menciona: {missing_str}\n"
        f"PERO estas entidades NO aparecen en NINGUNA de las evidencias.\n"
        f"REGLA OBLIGATORIA:\n"
        f"- NO asocies ningún valor a estas entidades.\n"
        f"- Responde que NO encuentras información sobre {missing_str} en las evidencias disponibles.\n"
        f"- answerable DEBE ser false para preguntas sobre {missing_str}.\n"
    )


# ============================================================
# HOOK PARA INTEGRACIÓN CON RAG CORE
# ============================================================

def create_entity_aware_evidence_response(
    query: str,
    hits: List[Dict[str, Any]],
    evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Post-procesa la evidencia para manejar entidades faltantes.
    
    Si la query pregunta por una entidad que no está en las evidencias,
    modifica la respuesta para indicar que no se encontró información.
    """
    validation = validate_entities_in_chunks(query, hits)
    
    if not validation["is_valid"]:
        missing_str = ", ".join(validation["missing_entities"])
        
        # Modificar la evidencia
        evidence["answerable"] = False
        evidence["missing"] = evidence.get("missing", []) + [
            f"No se encontró información sobre: {missing_str} en las evidencias disponibles"
        ]
        evidence["_entity_validation"] = validation
        evidence["key_points"] = []  # Limpiar key_points que podrían tener asociaciones incorrectas
    
    return evidence
