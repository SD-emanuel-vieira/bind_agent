"""
Contador acumulativo de tokens para todas las llamadas al LLM.

Uso:
    from bind_rag_agent.token_counter import token_counter

    token_counter.reset()                          # al inicio de answer_with_rag
    token_counter.add_from_response(resp)          # dentro de call_chat / call_llm
    totals = token_counter.get_totals()            # al final de answer_with_rag
"""

from typing import Any, Dict


class TokenCounter:
    """Acumula prompt_tokens y completion_tokens de cada llamada al LLM."""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Reinicia contadores a cero (llamar al inicio de cada request)."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    # ------------------------------------------------------------------
    def add_from_response(self, resp: Any):
        """
        Extrae usage del response de client.predict() y lo acumula.

        Formato esperado (OpenAI-compatible):
          {
            "usage": {
              "prompt_tokens": 123,
              "completion_tokens": 45
            },
            ...
          }
        """
        usage = self._extract_usage(resp)
        if usage:
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.call_count += 1

    # ------------------------------------------------------------------
    def get_totals(self) -> Dict[str, int]:
        """Devuelve los totales acumulados."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "llm_call_count": self.call_count,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_usage(resp: Any) -> Dict[str, int] | None:
        """Intenta extraer el dict 'usage' del response, tolerando varios formatos."""
        if isinstance(resp, dict):
            d = resp
        else:
            try:
                d = resp.__dict__
            except Exception:
                return None

        usage = d.get("usage")
        if isinstance(usage, dict):
            return usage
        return None


# Instancia global (singleton de módulo)
token_counter = TokenCounter()
