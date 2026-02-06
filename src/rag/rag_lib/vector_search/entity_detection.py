"""
entity_detection.py - Detección de entidades nombradas (NER simplificado).

Este módulo detecta entidades nombradas en queries para evitar que se
fragmenten incorrectamente durante el procesamiento.

Ejemplo del problema que resuelve:
- Query: "¿Cuál fue el ROA del Banco Santander?"
- SIN este módulo: anchors = ["banco", "santander", "roa"] 
  → "banco" matchea con "BIND Banco" incorrectamente
- CON este módulo: anchors = ["banco santander", "roa"]
  → Match correcto solo con chunks de Banco Santander

Estrategias de detección:
1. Prefijo + Nombre: "Banco Santander", "Grupo Galicia"
2. Capitalización consecutiva: "Mercado Pago"
3. Entidades conocidas: "BBVA", "Macro", etc.
"""

import re
from typing import List, Set, Tuple

# from rag_lib.vector_search.text_normalization import normalize_text as _norm
from rag_lib.text_utils import normalize_text as _norm


# ============================================================
# CONSTANTES NLP - STOPWORDS Y MESES
# ============================================================

STOPWORDS_ES: Set[str] = {
    # Interrogativos
    "cual", "cuál", "cuales", "cuáles", "como", "cómo", 
    "que", "qué", "quien", "quién", "donde", "dónde", "cuando", "cuándo",
    
    # Verbos auxiliares
    "son", "es", "fue", "fueron", "ser", "estar", "sido", "siendo",
    
    # Artículos
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    
    # Preposiciones
    "para", "por", "en", "con", "sin", "sobre", "entre", "hacia",
    
    # Conjunciones
    "y", "o", "ni", "pero", "sino", "aunque",
    
    # Pronombres
    "se", "le", "lo", "les", "nos", "me", "te",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra",
    
    # Otros
    "al", "a", "ha", "han", "hay", "he", "has",
    "muy", "mas", "más", "menos", "tan", "tanto", "mucho", "poco",
    "si", "no", "ya", "aun", "todavia", "tambien", "solo", "sólo",
}

MONTHS_ES: Set[str] = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
}


# ============================================================
# CONSTANTES DE ENTIDADES
# ============================================================

# Prefijos que indican inicio de nombre propio de empresa/institución
ENTITY_PREFIXES: Set[str] = {
    # Español
    "banco", "grupo", "empresa", "compañía", "fondo", 
    "corporación", "sociedad", "caja",
    
    # Inglés
    "bank", "banque", "group", "company", "fund", 
    "corporation", "corp",
    
    # Siglas comunes
    "fci",  # Fondo Común de Inversión
}

# Palabras que NO deben iniciar una entidad (falsos positivos comunes)
NON_ENTITY_STARTERS: Set[str] = {
    # Verbos de consulta
    "comparar", "mostrar", "buscar", "encontrar", "ver", "dame", "dime",
    
    # Interrogativos
    "cual", "cuales", "como", "que", "donde", "cuando",
    "cuanto", "cuanta", "cuantos", "cuantas",
}

# Entidades financieras conocidas en Argentina
# Si una palabra está aquí, se considera entidad incluso sin prefijo
KNOWN_FINANCIAL_ENTITIES: Set[str] = {
    # Bancos tradicionales
    "santander", "galicia", "bbva", "macro", "hsbc", "icbc", "patagonia",
    "supervielle", "hipotecario", "comafi", "itau", "itaú", "credicoop",
    
    # Bancos públicos
    "nacion", "nación", "provincia", "ciudad",
    
    # Bancos internacionales
    "citi", "citibank",
    
    # Fintechs y digitales
    "mercado", "mercadolibre", "mercadopago",
    "uala", "ualá", "prex", "naranja", "brubank",
    "rebanking", "openbank", "wilobank",
    
    # El propio banco
    "bind",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _is_capitalized_word(word: str) -> bool:
    """
    Verifica si una palabra está capitalizada (primera letra mayúscula).
    
    Args:
        word: Palabra a verificar
        
    Returns:
        True si la palabra tiene más de 1 caracter y empieza con mayúscula
    """
    return bool(word) and word[0].isupper() and len(word) > 1


def _is_likely_entity_word(word: str, word_norm: str) -> bool:
    """
    Determina si una palabra es probablemente parte de una entidad.
    
    Args:
        word: Palabra original (con capitalización)
        word_norm: Palabra normalizada
        
    Returns:
        True si la palabra parece ser parte de un nombre de entidad
    """
    # Entidad conocida
    if word_norm in KNOWN_FINANCIAL_ENTITIES:
        return True
    
    # Palabra capitalizada (no al inicio de oración)
    if _is_capitalized_word(word):
        return True
    
    return False


# ============================================================
# DETECCIÓN PRINCIPAL
# ============================================================

def detect_named_entities(query: str) -> List[Tuple[str, Set[str]]]:
    """
    Detecta entidades nombradas en la query.
    
    Aplica tres estrategias en orden:
    1. Patrón "Prefijo + Nombre": "Banco Santander", "Grupo Galicia"
    2. Palabras capitalizadas consecutivas: "Mercado Pago"
    3. Indicadores conocidos sueltos: "Santander", "BBVA"
    
    Args:
        query: Query del usuario
        
    Returns:
        Lista de tuplas (entidad_normalizada, set_de_palabras_consumidas)
        
    Examples:
        >>> detect_named_entities("¿Cuál fue el ROA del Banco Santander?")
        [("banco santander", {"banco", "santander"})]
        
        >>> detect_named_entities("Comparar Galicia vs BBVA")
        [("galicia", {"galicia"}), ("bbva", {"bbva"})]
    """
    entities: List[Tuple[str, Set[str]]] = []
    consumed_words: Set[str] = set()
    
    # Limpiar puntuación pero mantener capitalización
    query_clean = re.sub(r'[^\w\s]', '', query)
    
    words_original = query_clean.split()
    words_norm = [_norm(w) for w in words_original]
    
    i = 0
    while i < len(words_norm):
        word = words_norm[i]
        word_orig = words_original[i]
        
        # ============================================================
        # Estrategia 1: Prefijo + Nombre(s)
        # Ejemplo: "Banco Santander", "Grupo Financiero Galicia"
        # ============================================================
        if word in ENTITY_PREFIXES and i + 1 < len(words_norm):
            entity_parts = [word]
            j = i + 1
            
            # Consumir palabras siguientes mientras sean parte de la entidad
            while j < len(words_norm):
                next_word = words_norm[j]
                next_orig = words_original[j]
                
                # Continuar si es capitalizada o entidad conocida
                if _is_likely_entity_word(next_orig, next_word):
                    entity_parts.append(next_word)
                    j += 1
                else:
                    break
            
            # Solo crear entidad si hay más de una palabra
            if len(entity_parts) > 1:
                entity = " ".join(entity_parts)
                consumed = set(entity_parts)
                entities.append((entity, consumed))
                consumed_words.update(consumed)
                i = j
                continue
        
        # ============================================================
        # Estrategia 2: Palabras capitalizadas consecutivas
        # Ejemplo: "Mercado Pago", "JP Morgan"
        # ============================================================
        if _is_capitalized_word(word_orig) and word not in ENTITY_PREFIXES:
            # Ignorar si es un falso positivo conocido
            if word in NON_ENTITY_STARTERS:
                i += 1
                continue
            
            entity_parts = [word]
            j = i + 1
            
            # Consumir palabras capitalizadas consecutivas
            while j < len(words_norm):
                next_orig = words_original[j]
                next_word = words_norm[j]
                
                if _is_capitalized_word(next_orig):
                    entity_parts.append(next_word)
                    j += 1
                else:
                    break
            
            # Crear entidad si hay 2+ palabras O es una entidad conocida
            if len(entity_parts) >= 2 or word in KNOWN_FINANCIAL_ENTITIES:
                entity = " ".join(entity_parts)
                consumed = set(entity_parts)
                
                # Evitar superposición con entidades ya detectadas
                if not consumed.intersection(consumed_words):
                    entities.append((entity, consumed))
                    consumed_words.update(consumed)
                    i = j
                    continue
        
        # ============================================================
        # Estrategia 3: Entidades conocidas como palabra suelta
        # Ejemplo: "santander", "bbva", "macro"
        # ============================================================
        if word in KNOWN_FINANCIAL_ENTITIES and word not in consumed_words:
            entities.append((word, {word}))
            consumed_words.add(word)
        
        i += 1
    
    return entities


def get_entity_words(query: str) -> Set[str]:
    """
    Obtiene todas las palabras que forman parte de entidades detectadas.
    
    Útil para saber qué palabras ya fueron "consumidas" por entidades
    y no deben procesarse como keywords individuales.
    
    Args:
        query: Query del usuario
        
    Returns:
        Set de palabras normalizadas que son parte de entidades
        
    Example:
        >>> get_entity_words("ROA del Banco Santander")
        {"banco", "santander"}
    """
    entities = detect_named_entities(query)
    
    all_words: Set[str] = set()
    for _, words in entities:
        all_words.update(words)
    
    return all_words


def extract_entity_strings(query: str) -> List[str]:
    """
    Extrae solo los strings de entidades (sin el set de palabras).
    
    Convenience function para cuando solo necesitas los nombres.
    
    Args:
        query: Query del usuario
        
    Returns:
        Lista de strings de entidades detectadas
        
    Example:
        >>> extract_entity_strings("Comparar Banco Santander vs BBVA")
        ["banco santander", "bbva"]
    """
    entities = detect_named_entities(query)
    return [entity for entity, _ in entities]
