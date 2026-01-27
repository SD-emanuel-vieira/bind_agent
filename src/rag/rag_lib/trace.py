# ============================================================
# ACTUALIZACIÓN PARA trace_stage() en rag_core.py
# ============================================================
# Reemplazar la función trace_stage existente con esta versión
# que muestra el nuevo _anchor_score calculado internamente.
import os, re, unicodedata
from collections import Counter

DEBUG_TRACE = os.getenv("RAG_DEBUG_TRACE", "0") == "1"

def _short_id(cid: str, n: int = 10) -> str:
    cid = cid or ""
    return cid[:n]

def trace_stage(stage: str, query: str, hits: list[dict], top: int = 8):
    """Tracer resumido para debuggear orden sin spamear output."""
    if not DEBUG_TRACE:
        return
    if hits is None:
        hits = []

    anchors = query_anchors(query)
    total = len(hits)

    # stats rápidos
    anchor_scores = [hit_anchor_score(h, anchors) for h in hits] if anchors else [0] * total
    n_anchor_pos = sum(1 for s in anchor_scores if s > 0)

    ctype_counts = Counter((h.get("chunk_type") or "NA") for h in hits)
    top_ctypes = ", ".join([f"{k}:{v}" for k, v in ctype_counts.most_common(3)])

    print("\n" + "-" * 110)
    print(f"[TRACE] {stage} | total={total} | anchors={anchors} | anchor_hits>0={n_anchor_pos} | chunk_type={top_ctypes}")

    # ✅ si querés imprimir TODOS, setear env var RAG_DEBUG_TRACE_ALL=1
    trace_all = os.getenv("RAG_DEBUG_TRACE_ALL", "0") == "1"
    limit = total if trace_all else min(top, total)

    for i in range(limit):
        h = hits[i]
        a = anchor_scores[i] if i < len(anchor_scores) else 0
        fd = h.get("file_date") or ""
        pg = h.get("page_num")
        ct = h.get("chunk_type") or ""
        gb = h.get("_glossary_bonus")
        
        # NUEVO: Mostrar el anchor_score calculado internamente
        internal_anchor = h.get("_anchor_score", "?")
        
        cid = _short_id(h.get("chunk_id"))

        path = h.get("path") or ""
        tail = path.split("/")[-1] if path else ""

        # ACTUALIZADO: formato con anchor_score interno
        print(f"  {i+1:02d}) a={a} | a_int={internal_anchor} | g={gb} | {fd} | p={pg} | {ct} | {tail} | cid={cid}")