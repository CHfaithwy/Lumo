"""Compatibility exports for provider clients."""

from .providers import clients as _clients

for _name in dir(_clients):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_clients, _name)

__all__ = [_name for _name in globals() if not _name.startswith("__")]
