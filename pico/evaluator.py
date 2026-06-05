"""Compatibility exports for evaluation helpers."""

from .evaluation import evaluator as _evaluator

for _name in dir(_evaluator):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_evaluator, _name)

__all__ = [_name for _name in globals() if not _name.startswith("__")]
