from typing import Dict, Iterable, Optional

def get_dbx_secret(key: str, scope: str = "bind_agent_scope") -> str:
    """
    Lee un secret de Databricks Secret Scope usando dbutils.
    Requiere ejecutarse dentro de un Databricks notebook/cluster donde exista `dbutils`.

    Args:
        key: nombre del secret (key) dentro del scope.
        scope: nombre del secret scope (default: bind_agent_scope).

    Returns:
        El valor del secret como string.

    Raises:
        RuntimeError: si no está dbutils disponible (no estás en un notebook de Databricks).
        Exception: si el scope/key no existe o no tenés permisos.
    """
    try:
        # dbutils suele estar inyectado en el notebook; este try evita NameError fuera de Databricks
        _dbutils = dbutils  # type: ignore[name-defined]
    except Exception as e:
        raise RuntimeError("dbutils no está disponible. Esta función debe ejecutarse en un Databricks notebook.") from e

    return _dbutils.secrets.get(scope=scope, key=key)


def get_dbx_secrets(keys: Iterable[str], scope: str = "bind_agent_scope") -> Dict[str, Optional[str]]:
    """
    Lee múltiples secrets. Devuelve dict {key: value} y pone None si no pudo leer alguno.

    Útil para smoke tests donde querés validar qué está configurado sin romper todo.
    """
    out: Dict[str, Optional[str]] = {}
    for k in keys:
        try:
            out[k] = get_dbx_secret(k, scope=scope)
        except Exception:
            out[k] = None
    return out