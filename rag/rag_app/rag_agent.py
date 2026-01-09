from rag_lib.secret_functions import *
from rag_lib.config import *
from rag_core import *

# -------------------------
# MLflow PyFunc Model
# -------------------------
class RagAgent(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):  # type: ignore[override]
        try:
            rows = model_input.to_dict(orient="records")
        except Exception:
            rows = model_input if isinstance(model_input, list) else [model_input]

        outputs = []
        for row in rows:
            q = None
            if isinstance(row, dict) and row.get("messages"):
                q = row["messages"][-1].get("content")
            if not q and isinstance(row, dict):
                q = row.get("query")

            if not q:
                outputs.append({"answer": "", "sources_json": "[]", "error": "Missing 'messages' or 'query'"})
                continue

            try:
                out = answer_with_rag(q)
                # sources_json: lista simple (compatible con tu smoke test)
                sources = []
                for i, h in enumerate(out.get("reranked_hits", []) or [], start=1):
                    sources.append({
                        "sid": f"S{i}",
                        "path": h.get("path"),
                        "page_num": h.get("page_num"),
                        "chunk_id": h.get("chunk_id"),
                        "topic": h.get("topic"),
                    })
                outputs.append({
                    "answer": out.get("answer", ""),
                    "sources_json": json.dumps(sources, ensure_ascii=False),
                    "error": "",
                })
            except Exception as e:
                outputs.append({"answer": "", "sources_json": "[]", "error": str(e)})

        return outputs


# ✅ requerido para “code-based logging”
mlflow.models.set_model(RagAgent())