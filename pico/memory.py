"""Compatibility exports for runtime memory."""

from .features import memory as _memory

for _name in dir(_memory):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_memory, _name)

__all__ = [_name for _name in globals() if not _name.startswith("__")]
