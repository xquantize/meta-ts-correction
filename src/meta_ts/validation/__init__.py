__all__ = ["GroupResult", "validate_harness"]


def __getattr__(name: str):
    if name in __all__:
        from meta_ts.validation import harness

        return getattr(harness, name)
    raise AttributeError(name)
