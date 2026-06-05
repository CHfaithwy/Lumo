"""Compatibility exports for evaluation metrics."""

from .evaluation import metrics as _metrics

for _name in dir(_metrics):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_metrics, _name)

__all__ = [_name for _name in globals() if not _name.startswith("__")]
