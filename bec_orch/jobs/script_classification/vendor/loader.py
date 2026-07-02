from .pipeline import Pipeline

_pipeline = None


def get_pipeline(cfg=None):
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
