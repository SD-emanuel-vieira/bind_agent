"""
business_rules.py - Reglas y definiciones de negocio para el RAG

Este módulo contiene:
1. Definiciones de métricas financieras
2. Fórmulas de cálculo
3. Relaciones entre conceptos
4. Contexto que ayuda al LLM a dar mejores respuestas

Se inyecta en el prompt de generación de respuestas (answer_from_evidence)
para que el LLM tenga contexto sobre cómo interpretar los datos.
"""

from typing import Dict, List, Optional
import re
import unicodedata


# ============================================================
# DEFINICIONES DE MÉTRICAS Y FÓRMULAS
# ============================================================

BUSINESS_RULES: Dict[str, Dict] = {
    
    # ==================== P&L / RESULTADOS ====================
    # JERARQUÍA DE RESULTADOS (de arriba hacia abajo):
    # 1. Resultado Operativo = Ingresos + Gastos (gastos vienen con signo negativo)
    # 2. Resultado Comercial Nominal = Resultado Operativo + IIGG
    # 3. Resultado Comercial Nominal Neto AxI = Resultado Comercial Nominal + AxI
    # 4. Resultado Gestión Neto AxI = Resultado Comercial Nominal Neto AxI + Ajuste Waiver
    # 5. "Resultado" a secas = Resultado Gestión Neto AxI
    
    "Resultado Operativo": {
        "definicion": "Resultado de las operaciones del negocio antes de impuestos y ajustes. Es la suma de Ingresos y Gastos en el P&L (los gastos ya vienen con signo negativo).",
        "formula": "Resultado Operativo = Ingresos + Gastos",
        "componentes": ["Ingresos", "Gastos"],
        "aliases": ["resultado operativo", "rdo operativo", "operating result", "utilidad operativa"],
        "notas": [
            "En el P&L los Gastos aparecen con signo negativo, por eso la fórmula es una suma",
            "No incluye impuesto a las ganancias (IIGG)",
            "No incluye ajustes por inflación (AxI)",
        ]
    },
    
    "Resultado Comercial Nominal": {
        "definicion": "Resultado operativo más el impuesto a las ganancias. Es el resultado antes de ajuste por inflación.",
        "formula": "Resultado Comercial Nominal = Resultado Operativo + IIGG",
        "componentes": ["Resultado Operativo", "IIGG"],
        "aliases": ["resultado comercial nominal", "resultado comercial", "rdo comercial nominal", "nominal commercial result"],
        "notas": [
            "IIGG viene con signo negativo en el P&L, por eso se suma",
            "Este resultado aún no está ajustado por inflación",
        ]
    },
    
    "Resultado Comercial Neto AxI": {
        "definicion": "Resultado comercial nominal neto del ajuste por inflación. Es el resultado 'real' en términos de poder adquisitivo.",
        "formula": "Resultado Comercial Neto AxI = Resultado Comercial Nominal + AxI",
        "componentes": ["Resultado Comercial Nominal", "AxI"],
        "aliases": ["resultado comercial neto axi", "resultado comercial nominal neto axi", "rdo comercial neto axi", "resultado neto ajustado"],
        "notas": [
            "AxI (Ajuste por Inflación) típicamente es negativo en contextos inflacionarios",
            "Este es el resultado 'real' que descuenta el efecto de la inflación",
        ]
    },
    
    "Resultado Gestión Neto AxI": {
        "definicion": "Resultado total de gestión del negocio. Es el Resultado Comercial Nominal Neto AxI más el Ajuste Waiver. Cuando se habla de 'Resultado' a secas, generalmente se refiere a esta métrica.",
        "formula": "Resultado Gestión Neto AxI = Resultado Comercial Neto AxI + Ajuste Waiver",
        "componentes": ["Resultado Comercial Neto AxI", "Ajuste Waiver"],
        "aliases": ["resultado gestion neto axi", "resultado gestion", "rdo gestion", "resultado de gestion", "resultado", "rdo"],
        "notas": [
            "'Resultado' a secas típicamente se refiere a Resultado Gestión Neto AxI",
            "Es la métrica más completa de rentabilidad del negocio",
        ]
    },
    
    "Resultado Total con Vinculadas": {
        "definicion": "Resultado final incluyendo el cross-sell (XSell) generado con empresas vinculadas del grupo.",
        "formula": "Resultado Total con Vinculadas = Resultado Gestión Neto AxI + XSell Vinculadas",
        "componentes": ["Resultado Gestión Neto AxI", "XSell Vinculadas"],
        "aliases": ["resultado total con vinculadas", "resultado con xsell", "resultado total"],
    },

    "Resultados Integrales": {
        "desc": "Incluye Ingresos, Gastos y Resultado Neto de una entidad",
        "aliases": ["resultados integrales", "ingresos", "estado de resultados"],
    },
    
    # ==================== INGRESOS ====================
    
    "Ingresos": {
        "definicion": "Suma de todos los ingresos del banco: margen financiero de préstamos, margen financiero de depósitos, comisiones, y resultados por operaciones de cambio.",
        "formula": "Ingresos = MF Préstamos + MF Depósitos + MF AFIP + NDF/FX + Comisiones Netas + Previsiones & Otros",
        "componentes": ["MF Préstamos", "MF Depósitos", "MF AFIP", "NDF + FX", "Comisiones Netas", "Previsiones & Otros"],
        "aliases": ["ingresos", "ingresos totales", "revenue", "income"],
        "notas": [
            "Previsiones & Otros puede ser positivo o negativo dependiendo del período",
        ]
    },
    
    "MF Préstamos": {
        "definicion": "Margen Financiero por Préstamos. Diferencia entre intereses cobrados por préstamos y costo de fondeo asignado.",
        "formula": "MF Préstamos = Intereses Cobrados por Préstamos - Costo de Fondeo (proporción préstamos)",
        "aliases": ["mf prestamos", "margen financiero prestamos", "margen de préstamos", "loan margin"],
        "notas": [
            "Mayor spread entre tasa activa y costo de fondeo genera mayor MF",
        ]
    },
    
    "MF Depósitos": {
        "definicion": "Margen Financiero por Depósitos. Incluye el margen generado por encaje remunerado.",
        "formula": "MF Depósitos = Rendimiento de Colocación - Intereses Pagados a Depositantes",
        "aliases": ["mf depositos", "margen financiero depositos", "mf depositos + encaje remunerado", "deposit margin"],
    },
    
    "MF AFIP": {
        "definicion": "Margen Financiero por operaciones relacionadas con AFIP (recaudación impositiva).",
        "aliases": ["mf afip", "margen afip"],
    },
    
    "Comisiones Netas": {
        "definicion": "Ingresos por servicios menos comisiones pagadas. Incluye mantenimiento de cuentas, transferencias, tarjetas, etc.",
        "formula": "Comisiones Netas = Comisiones Cobradas - Comisiones Pagadas",
        "aliases": ["comisiones netas", "comisiones", "fees", "net fees", "net commissions"],
    },
    
    "NDF + FX": {
        "definicion": "Resultado por operaciones de derivados (Non-Deliverable Forwards) y tipo de cambio (Foreign Exchange).",
        "aliases": ["ndf fx", "ndf + fx", "resultado cambiario", "forex", "ndf"],
        "notas": [
            "Volátil y dependiente de movimientos del tipo de cambio",
            "Puede ser positivo o negativo según el período",
        ]
    },
    
    "Previsiones & Otros": {
        "definicion": "Provisiones para pérdidas crediticias y otros resultados menores. Dentro de la línea de Ingresos.",
        "aliases": ["previsiones & otros", "previsiones y otros", "previsiones"],
        "notas": [
            "Puede ser positivo (recupero de previsiones) o negativo (constitución de previsiones)",
        ]
    },
    
    # ==================== GASTOS ====================
    
    "Gastos": {
        "definicion": "Total de gastos operativos del banco. Aparecen con signo negativo en el P&L.",
        "formula": "Gastos = Directos + Indirectos",
        "componentes": ["Directos", "Indirectos"],
        "aliases": ["gastos", "gastos totales", "expenses", "costos operativos", "egresos"],
        "notas": [
            "Los gastos aparecen con signo NEGATIVO en el P&L",
            "Al sumar Ingresos + Gastos, los gastos restan automáticamente",
        ]
    },
    
    "Directos": {
        "definicion": "Gastos directamente atribuibles a una línea de negocio o segmento.",
        "aliases": ["directos", "gastos directos", "direct costs", "costos directos"],
        "ejemplos": ["Comisiones a vendedores", "Costos de originación", "Marketing directo"],
    },
    
    "Indirectos": {
        "definicion": "Gastos compartidos que se asignan a las líneas de negocio por algún criterio de distribución (overhead).",
        "aliases": ["indirectos", "gastos indirectos", "overhead", "indirect costs", "costos indirectos"],
        "ejemplos": ["Alquiler oficinas", "IT compartido", "RRHH", "Compliance", "Management"],
    },
    
    # ==================== AJUSTES Y OTROS ====================
    
    "IIGG": {
        "definicion": "Impuesto a las Ganancias. Carga fiscal sobre el resultado. Aparece con signo negativo en el P&L.",
        "aliases": ["iigg", "impuesto a las ganancias", "income tax", "impuesto"],
        "notas": [
            "Viene con signo NEGATIVO en el P&L",
            "Se suma al Resultado Operativo para obtener el Resultado Comercial Nominal",
        ]
    },
    
    "AxI": {
        "definicion": "Ajuste por Inflación. Corrección monetaria para reflejar el impacto de la inflación en el resultado.",
        "aliases": ["axi", "ajuste por inflacion", "ajuste por inflación", "inflation adjustment"],
        "notas": [
            "Típicamente NEGATIVO en contextos de alta inflación",
            "Representa la pérdida de poder adquisitivo del capital",
            "Se calcula sobre posiciones monetarias netas",
        ]
    },
    
    "Ajuste Waiver": {
        "definicion": "Ajuste relacionado con waivers de deuda o condiciones especiales. Se suma al Resultado Comercial Neto AxI.",
        "formula": "Ajuste Waiver (Neto IIGG) - Se presenta neto del efecto impositivo",
        "aliases": ["ajuste waiver", "waiver", "ajuste waive"],
        "notas": [
            "Puede ser positivo o negativo",
            "Relacionado con renegociaciones de deuda o condiciones especiales",
        ]
    },
    
    "RECPAM": {
        "definicion": "Resultado por Exposición a los Cambios en el Poder Adquisitivo de la Moneda. Metodología específica de BCRA.",
        "aliases": ["recpam", "resultado por exposicion monetaria"],
        "notas": [
            "Similar conceptualmente a AxI pero con metodología regulatoria",
        ]
    },
    
    # ==================== RATIOS DE RENTABILIDAD ====================
    
    "ROA": {
        "definicion": "Return on Assets. Rentabilidad sobre activos totales. Mide la eficiencia en el uso de activos.",
        "formula": "ROA = Resultado Neto / Activos Totales Promedio × 100",
        "aliases": ["roa", "return on assets", "retorno sobre activos", "rentabilidad sobre activos"],
        "interpretacion": "ROA > 1% es bueno para bancos. ROA > 2% es excelente.",
    },
    
    "ROE": {
        "definicion": "Return on Equity. Rentabilidad sobre patrimonio neto. Mide el retorno para los accionistas.",
        "formula": "ROE = Resultado Neto / Patrimonio Neto Promedio × 100",
        "aliases": ["roe", "return on equity", "retorno sobre patrimonio", "rentabilidad sobre patrimonio"],
        "interpretacion": "ROE > 15% es bueno. ROE > 20% es excelente.",
    },
    
    # ==================== CROSS-SELL Y VINCULADAS ====================
    
    "XSell Vinculadas": {
        "definicion": "Cross-sell con empresas vinculadas del grupo. Ingresos generados por referir clientes o servicios entre empresas del grupo económico.",
        "aliases": ["xsell vinculadas", "cross sell vinculadas", "xsell", "cross-sell", "vinculadas"],
        "notas": [
            "Representa sinergias del grupo",
            "No es resultado directo del banco pero se atribuye a la relación comercial",
        ]
    },
    
    # ==================== SEGMENTOS ====================
    
    "Empresas": {
        "definicion": "Segmento de banca para empresas medianas (middle market, PyMEs grandes).",
        "aliases": ["empresas", "banca empresas", "middle market", "pymes"],
    },
    
    "Corporate": {
        "definicion": "Segmento de banca para grandes empresas y corporaciones multinacionales.",
        "aliases": ["corporate", "banca corporate", "banca corporativa", "grandes empresas", "large corporate"],
    },
    
    "Institucional": {
        "definicion": "Segmento de banca para instituciones financieras, fondos de inversión, aseguradoras y organismos públicos.",
        "aliases": ["institucional", "banca institucional", "institutional"],
    },
    
    "Minorista": {
        "definicion": "Segmento de banca para personas físicas (individuos).",
        "aliases": ["minorista", "retail", "banca minorista", "personas", "red minorista"],
    },
    
    "BaaS": {
        "definicion": "Banking as a Service / Digital. Segmento que ofrece infraestructura bancaria a fintechs y empresas no financieras.",
        "aliases": ["baas", "digital", "banking as a service"],
    },
    
    "Zafiro": {
        "definicion": "Segmento de banca premium o private banking para clientes de alto patrimonio.",
        "aliases": ["zafiro", "private banking", "banca privada"],
    },
    
    # ==================== OTROS ====================
    
    "Tesorería": {
        "definicion": "Área que gestiona la liquidez, inversiones y posición de mercado del banco.",
        "aliases": ["tesoreria", "treasury"],
        "notas": [
            "Resultado de Tesorería puede ser muy volátil",
            "Incluye resultado por posiciones en títulos y pases",
        ]
    },
    
    "Cuota a Cuota": {
        "definicion": "Línea de negocio de financiamiento en cuotas (consumo, automotor, etc.).",
        "aliases": ["cuota a cuota", "cuotas"],
    },
}


# ============================================================
# RELACIONES JERÁRQUICAS (para entender composición)
# ============================================================

METRIC_HIERARCHY = {
    "Resultado Total con Vinculadas": {
        "formula": "= Resultado Gestión Neto AxI + XSell Vinculadas",
        "components": ["Resultado Gestión Neto AxI", "XSell Vinculadas"]
    },
    "Resultado Gestión Neto AxI": {
        "formula": "= Resultado Comercial Neto AxI + Ajuste Waiver",
        "components": ["Resultado Comercial Neto AxI", "Ajuste Waiver"],
        "nota": "'Resultado' a secas típicamente se refiere a esta métrica"
    },
    "Resultado Comercial Neto AxI": {
        "formula": "= Resultado Comercial Nominal + AxI",
        "components": ["Resultado Comercial Nominal", "AxI"]
    },
    "Resultado Comercial Nominal": {
        "formula": "= Resultado Operativo + IIGG",
        "components": ["Resultado Operativo", "IIGG"]
    },
    "Resultado Operativo": {
        "formula": "= Ingresos + Gastos (gastos con signo negativo)",
        "components": ["Ingresos", "Gastos"]
    },
    "Ingresos": {
        "formula": "= MF Préstamos + MF Depósitos + MF AFIP + NDF/FX + Comisiones Netas + Previsiones & Otros",
        "components": ["MF Préstamos", "MF Depósitos", "MF AFIP", "NDF + FX", "Comisiones Netas", "Previsiones & Otros"]
    },
    "Gastos": {
        "formula": "= Directos + Indirectos (con signo negativo)",
        "components": ["Directos", "Indirectos"]
    },
}


# ============================================================
# REGLAS DE INTERPRETACIÓN GENERAL
# ============================================================

INTERPRETATION_RULES = """
## Jerarquía del P&L (de arriba hacia abajo):

1. **Ingresos** = MF Préstamos + MF Depósitos + MF AFIP + NDF/FX + Comisiones Netas + Previsiones & Otros
2. **Gastos** = Directos + Indirectos (aparecen con signo NEGATIVO)
3. **Resultado Operativo** = Ingresos + Gastos
4. **IIGG** = Impuesto a las Ganancias (signo NEGATIVO)
5. **Resultado Comercial Nominal** = Resultado Operativo + IIGG
6. **AxI** = Ajuste por Inflación (típicamente NEGATIVO)
7. **Resultado Comercial Neto AxI** = Resultado Comercial Nominal + AxI
8. **Ajuste Waiver** = Ajuste por waivers (puede ser + o -)
9. **Resultado Gestión Neto AxI** = Resultado Comercial Neto AxI + Ajuste Waiver
10. **Resultado Total con Vinculadas** = Resultado Gestión Neto AxI + XSell Vinculadas

## Nota importante sobre signos:
- Gastos, IIGG y AxI aparecen con signo NEGATIVO en el P&L
- Por eso las fórmulas son SUMAS (el signo ya está incorporado)
- "Resultado" a secas = Resultado Gestión Neto AxI

## Comparaciones temporales:
- **MoM**: Month over Month (vs mes anterior)
- **YTD**: Year to Date (acumulado del año)
- **i.a.**: Interanual (vs mismo período año anterior)
- **vs Budget**: Comparación contra presupuesto
"""


# ============================================================
# FUNCIONES PARA EXTRAER REGLAS RELEVANTES
# ============================================================

def _norm(s: str) -> str:
    """Normaliza texto para matching."""
    s = (s or "").strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def get_relevant_rules(query: str, evidence_text: str = "", max_rules: int = 10) -> List[Dict]:
    """
    Extrae las reglas de negocio relevantes basándose en la query y la evidencia.
    
    Args:
        query: Pregunta del usuario
        evidence_text: Texto de la evidencia recuperada (opcional)
        max_rules: Máximo de reglas a retornar
        
    Returns:
        Lista de reglas relevantes con sus definiciones y fórmulas
    """
    combined_text = _norm(query + " " + evidence_text)
    
    relevant = []
    matched_names = set()
    
    for metric_name, rule in BUSINESS_RULES.items():
        # Verificar si el nombre o algún alias aparece en el texto
        aliases = [metric_name.lower()] + [a.lower() for a in rule.get("aliases", [])]
        
        for alias in aliases:
            alias_norm = _norm(alias)
            if alias_norm and alias_norm in combined_text:
                if metric_name not in matched_names:
                    relevant.append({
                        "nombre": metric_name,
                        **rule
                    })
                    matched_names.add(metric_name)
                break
    
    # Si se menciona "resultado" a secas, asegurar que se incluya Resultado Gestión Neto AxI
    if "resultado" in combined_text and "Resultado Gestión Neto AxI" not in matched_names:
        # Verificar que no sea parte de otra frase como "resultado operativo"
        resultado_patterns = ["resultado operativo", "resultado comercial", "resultado total"]
        is_specific = any(p in combined_text for p in resultado_patterns)
        
        if not is_specific:
            relevant.insert(0, {
                "nombre": "Resultado Gestión Neto AxI",
                **BUSINESS_RULES["Resultado Gestión Neto AxI"]
            })
    
    # Priorizar reglas con fórmulas
    relevant.sort(key=lambda r: (
        1 if "formula" in r else 0,
        len(r.get("componentes", []))
    ), reverse=True)
    
    return relevant[:max_rules]


def format_rules_for_prompt(rules: List[Dict], include_hierarchy: bool = True) -> str:
    """
    Formatea las reglas para incluir en el prompt del LLM.
    
    Args:
        rules: Lista de reglas relevantes
        include_hierarchy: Si incluir la jerarquía de P&L cuando hay métricas de resultados
        
    Returns:
        String formateado para incluir en el prompt
    """
    if not rules:
        return ""
    
    lines = [
        "REGLAS DE NEGOCIO (usar para interpretar métricas y dar contexto):"
    ]
    
    # Verificar si hay métricas de P&L para incluir la jerarquía
    pl_keywords = {"resultado", "ingresos", "gastos", "operativo", "comercial", "axi", "iigg"}
    has_pl_metrics = any(
        any(kw in _norm(r.get("nombre", "")) for kw in pl_keywords)
        for r in rules
    )
    
    for rule in rules:
        nombre = rule.get("nombre", "")
        definicion = rule.get("definicion", "")
        formula = rule.get("formula", "")
        notas = rule.get("notas", [])
        
        lines.append(f"\n• {nombre}:")
        if definicion:
            lines.append(f"  Definición: {definicion}")
        if formula:
            lines.append(f"  Fórmula: {formula}")
        for nota in notas[:2]:  # Máximo 2 notas por regla
            lines.append(f"  Nota: {nota}")
    
    # Agregar jerarquía de P&L si es relevante
    if include_hierarchy and has_pl_metrics:
        lines.append("\n" + INTERPRETATION_RULES)
    
    return "\n".join(lines)


def get_rules_snippet(query: str, evidence_text: str = "") -> str:
    """
    Función principal para obtener el snippet de reglas para el prompt.
    
    Args:
        query: Pregunta del usuario
        evidence_text: Texto de la evidencia (opcional)
        
    Returns:
        String listo para inyectar en el prompt
    """
    rules = get_relevant_rules(query, evidence_text)
    return format_rules_for_prompt(rules)


# ============================================================
# PARA DEBUG / TESTING
# ============================================================

if __name__ == "__main__":
    # Test
    test_queries = [
        "Cuál fue el resultado operativo de octubre?",
        "Cuál fue el resultado de octubre?",  # Debería mapear a Resultado Gestión Neto AxI
        "Cómo se calcula el ROA?",
        "Cuáles son los gastos del segmento empresas?",
        "Qué es el resultado comercial neto axi?",
        "Cómo fue el ajuste por inflación?",
    ]
    
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        snippet = get_rules_snippet(q)
        print(snippet if snippet else "(No se encontraron reglas relevantes)")