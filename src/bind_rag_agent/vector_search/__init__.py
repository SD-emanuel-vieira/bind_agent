"""
vector_search - Módulo de búsqueda vectorial y procesamiento de queries.

Este paquete contiene los componentes para:
- Normalización de texto
- Búsqueda en glosario de negocio
- Detección de entidades nombradas
- Preparación de candidatos para reranking

Estructura:
- text_normalization.py: Funciones de normalización centralizadas
- glossary_lookup.py: Búsqueda y expansión de términos del glosario
- entity_detection.py: Detección de entidades nombradas (NER)
- glossary_helper.py: Orquestador principal

Uso típico:
    from bind_rag_agent.vector_search.glossary_helper import (
        glossary_snippet,
        prepare_rerank_candidates_glossary_aware,
    )
"""

# Re-exportar funciones principales para conveniencia
from bind_rag_agent.text_utils import (
    normalize_text,
    normalize_for_search,
)

from bind_rag_agent.vector_search.glossary_lookup import (
    glossary_snippet,
    glossary_expand_terms,
    glossary_bonus,
)

from bind_rag_agent.vector_search.entity_detection import (
    detect_named_entities,
    STOPWORDS_ES,
    MONTHS_ES,
)

from bind_rag_agent.vector_search.glossary_helper import (
    prepare_rerank_candidates_glossary_aware,
    extract_query_anchors,
    compute_anchor_score,
)

__all__ = [
    # Normalización
    'normalize_text',
    'normalize_for_search',
    
    # Glosario
    'glossary_snippet',
    'glossary_expand_terms',
    'glossary_bonus',
    
    # Entidades
    'detect_named_entities',
    'STOPWORDS_ES',
    'MONTHS_ES',
    
    # Orquestador
    'prepare_rerank_candidates_glossary_aware',
    'extract_query_anchors',
    'compute_anchor_score',
]
