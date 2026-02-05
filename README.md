# BIND RAG Agent

Sistema de Retrieval-Augmented Generation (RAG) para consultas sobre documentos financieros corporativos de BIND Banco. Desplegado en Databricks con MLflow y Model Serving.

---

## 📁 Estructura del Proyecto

```
bind_agent/
├── databricks.yml              # Configuración de Databricks Asset Bundle
├── pyproject.toml              # Dependencias del proyecto
├── resources/
│   └── deploy_rag_agent.job.yml  # Job de deployment
├── src/
│   ├── questions.json          # Preguntas de prueba
│   ├── notebooks/
│   │   └── deploy_rag_agent.py   # Notebook de deployment
│   └── rag/
│       ├── rag_agent.py        # Agente MLflow (entry point)
│       └── rag_lib/            # Librería del RAG
│           ├── __init__.py
│           ├── config.py           # Configuración y variables de entorno
│           ├── rag_core.py         # Orquestador principal del RAG
│           ├── retriever.py        # Recuperación de candidatos (Vector Search + Lexical)
│           ├── rerank.py           # Reranking con LLM y protección de anchors
│           ├── evidence_handling.py # Extracción de evidencia y generación de respuestas
│           ├── glossary_helper.py  # Glosario y extracción de anchors/entidades
│           ├── business_glossary.py # Definiciones de términos financieros
│           ├── business_rules.py   # Reglas de negocio y fórmulas del P&L
│           ├── text_utils.py       # Utilidades de texto y filtrado de segmentos
│           ├── llm.py              # Llamadas al LLM
│           ├── embeddings.py       # Generación de embeddings
│           └── secret_functions.py # Funciones de secretos
└── fixtures/                   # Archivos de prueba
```

---

## 🔄 Pipeline del RAG

El flujo completo de una consulta se ejecuta en `rag_core.py`:

```
Query del usuario
        │
        ▼
┌───────────────────────────────────────┐
│ 1) retrieve_candidates()              │
│    - Vector Search (embeddings)       │
│    - Lexical fallback (FULL_TEXT)     │
│    - Filtrado por chunk_type gates    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 1.1) drop_segment_topics_if_query_    │
│      general()                        │
│    - Filtrado inteligente por         │
│      segmentos (Empresas, Minorista,  │
│      Institucional, etc.)             │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 2) enforce_anchor_priority()          │
│    - Prioriza chunks con keywords     │
│      de la query                      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 3) prepare_rerank_candidates_         │
│    glossary_aware()                   │
│    - Scoring por glosario             │
│    - Detección de entidades nombradas │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4) tie_break_by_date_in_blocks()      │
│    - Desempate por fecha más reciente │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 5) rerank_with_llm()                  │
│    - Reranking semántico con LLM      │
│    - Protección de chunks con alto    │
│      anchor_score                     │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 6) extract_evidence()                 │
│    - LLM extrae evidencia relevante   │
│    - Determina si es answerable       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 7) answer_from_evidence()             │
│    - Genera respuesta final           │
│    - Inyecta business_rules y         │
│      glosario                         │
└───────────────────────────────────────┘
        │
        ▼
    Respuesta + Citations
```

---

## 📦 Módulos Principales

### `config.py`
Configuración centralizada mediante variables de entorno:

| Variable | Descripción |
|----------|-------------|
| `RAG_VS_ENDPOINT` | Endpoint de Vector Search |
| `RAG_VS_INDEX_FULL_NAME` | Nombre completo del índice |
| `RAG_EMBED_ENDPOINT` | Endpoint de embeddings |
| `RAG_LLM_ENDPOINT` | Endpoint del LLM |
| `RAG_TOP_K_CANDIDATES` | Candidatos iniciales (default: 40) |
| `RAG_TOP_K_FINAL` | Candidatos finales (default: 8) |
| `RAG_MAX_CONTEXT_CHARS` | Máximo caracteres de contexto (default: 14000) |

### `retriever.py`
Recuperación de candidatos con estrategia híbrida:

1. **Vector Search**: Búsqueda semántica usando embeddings
2. **Lexical Fallback**: Búsqueda FULL_TEXT para términos exactos
3. **Query Gates**: Filtrado de chunk_types (ej: excluir gráficos si no aplica)

```python
def retrieve_candidates(query: str, k: int) -> List[Dict]:
    # 1. Expandir query con glosario + LLM
    # 2. Vector Search con embeddings
    # 3. Lexical fallback (FULL_TEXT)
    # 4. Filtrar por query gates
    # 5. Deduplicar por chunk_id
```

### `rerank.py`
Reranking inteligente que combina:

- **Anchor Score**: Match exacto de keywords/entidades
- **LLM Reranking**: Reordenamiento semántico
- **Protección de Anchors**: Chunks con alto anchor_score no pueden ser relegados

```python
def rerank_with_llm(query: str, hits: List[Dict], top_k: int) -> List[Dict]:
    # 1. Calcular anchor_score para cada hit
    # 2. Identificar hits "protegidos" (score >= 80% del máximo)
    # 3. Enviar al LLM con keyword_match_score en metadata
    # 4. Post-procesar: garantizar protegidos en top posiciones
```

### `evidence_handling.py`
Extracción de evidencia y generación de respuestas:

- **build_context()**: Construye contexto con tratamiento especial para tablas
- **extract_evidence()**: LLM identifica claims y evidencia
- **answer_from_evidence()**: Genera respuesta con business_rules y glosario

### `glossary_helper.py`
Manejo del glosario y detección de entidades:

- **Entidades Nombradas**: "Banco Santander" se trata como frase completa
- **Frases del Glosario**: "retorno sobre activos" → ROA
- **Anchor Scoring**: Frases = 5 puntos, palabras = 1 punto

### `text_utils.py`
Filtrado inteligente de segmentos con 3 comportamientos:

| Tipo de Query | Ejemplo | Acción |
|---------------|---------|--------|
| **Comparativa** | "¿Qué segmento generó más?" | Mantener TODOS los segmentos |
| **Específica** | "Ingresos de institucional" | Solo hits de ese segmento |
| **General** | "Resultado operativo total" | Descartar hits con segmentos |

### `business_rules.py`
Definiciones de métricas financieras y fórmulas del P&L:

```
Jerarquía del P&L:
1. Ingresos = MF Préstamos + MF Depósitos + Comisiones + ...
2. Gastos = Directos + Indirectos (signo negativo)
3. Resultado Operativo = Ingresos + Gastos
4. Resultado Comercial Nominal = Resultado Operativo + IIGG
5. Resultado Comercial Neto AxI = Resultado Comercial Nominal + AxI
6. Resultado Gestión Neto AxI = Resultado Comercial Neto AxI + Waiver
```

### `business_glossary.py`
Glosario de términos financieros con aliases:

```python
"YTD": {"desc": "Year to Date", "aliases": ["ytd", "acumulado", "acumulado del año"]}
"ROA": {"desc": "Return on Assets", "aliases": ["roa", "retorno sobre activos"]}
"MF":  {"desc": "Margen Financiero", "aliases": ["margen financiero", "nim"]}
```

---

## 🚀 Deployment

### Notebook: `deploy_rag_agent.py`

El deployment se realiza mediante el notebook `src/notebooks/deploy_rag_agent.py`:

1. **Configuración de Variables**: Lee parámetros del Asset Bundle o usa defaults
2. **Exportación de Código**: Copia `rag_agent.py` y `rag_lib/` a directorio temporal
3. **Log + Register en MLflow/UC**: Registra el modelo en Unity Catalog
4. **Crear/Actualizar Serving Endpoint**: Despliega como Model Serving

```python
# Ejemplo de invocación del endpoint
payload = {
    "dataframe_split": {
        "columns": ["query"],
        "data": [["¿Cuál fue el resultado operativo de octubre 2025?"]],
    }
}
resp = dc.predict(endpoint="bind_agent_rag_agent", inputs=payload)
```

### Asset Bundle (`databricks.yml`)

Configuración de targets:

| Target | Modo | Uso |
|--------|------|-----|
| `dev` | development | Desarrollo con prefijos `[dev]` |
| `prod` | production | Producción sin prefijos |

Variables configurables:
- `vs_endpoint`, `vs_index_full_name`
- `embed_endpoint`, `llm_endpoint`
- `top_k_candidates`, `top_k_final`
- `temperature_rerank`, `temperature_answer`

---

## 🔧 Configuración de Debugging

Activar trazas detalladas:

```bash
export RAG_DEBUG_TRACE=1      # Habilitar trace básico
export RAG_DEBUG_TRACE_ALL=1  # Mostrar todos los hits (no solo top 12)
```

Output del trace:
```
[TRACE] 5) rerank_with_llm(top_k=8) | total=8 | anchors=['resultado', 'operativo', '2025'] | anchor_hits>0=5
  01) a=3 | a_int=5 | g=5 | 2025-11-18 | p=12.0 | table | Directorio.pdf | cid=bb658864d1
  02) a=2 | a_int=3 | g=0 | 2025-11-18 | p=8.0  | table | Directorio.pdf | cid=fedd3617da
```

---

## 📊 Segmentos de Negocio

El sistema reconoce los siguientes segmentos:

| Segmento | Aliases |
|----------|---------|
| `empresas` | empresas, empresa |
| `corporate` | corporate, corp |
| `institucional` | institucional, institucionales |
| `minorista` | minorista, retail, individuos |
| `baas` | baas, banca as a service |
| `pyme` | pyme, pymes |

---

## 🧪 Ejemplos de Queries

```python
# Query general (sin segmento)
"¿Cuál fue el resultado operativo de octubre 2025?"

# Query con segmento específico
"¿Cuáles son los ingresos de institucional en octubre 2025?"

# Query comparativa
"¿Qué segmento ha generado más ingresos por MF préstamos?"

# Query sobre entidad externa
"¿Cuál fue el resultado integral YTD julio 2025 del Banco Santander?"

# Query con métricas compuestas
"¿Cómo fue el resultado comercial nominal sin ajuste por inflación en septiembre?"
```

---

## 📝 Notas Técnicas

### Chunks y Contexto
- Tamaño máximo de chunk recomendado: **7000 caracteres**
- Contexto máximo para LLM: **14000 caracteres**
- Si un chunk excede el límite, se trunca (no se descarta)

### Tablas sin Headers
Las tablas OCR frecuentemente no tienen headers. El sistema usa el campo `topic` para interpretar qué representan los valores:
```
[S1] TABLA | Contenido: Resultados financieros bancos (YTD julio 2025)
NOTA: Los valores corresponden a 'Resultados financieros bancos (YTD julio 2025)'.
```

### Protección de Anchors
Chunks con alto anchor_score (>= 80% del máximo) están "protegidos" y no pueden ser relegados por el LLM reranker.

---

## 🔗 Dependencias Principales

- `databricks-vectorsearch`
- `mlflow`
- `databricks-sdk`
- `pyspark`

---

## 👥 Contacto

Desarrollado para BIND Banco.