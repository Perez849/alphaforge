"""AlphaForge — predicción direccional intradía->overnight con validación
anti-sobreajuste.

Uso rápido:
    from alphaforge import Config, run_experiment, predict_next
    cfg = Config(); cfg.data.ticker = "AAPL"
    res = run_experiment(cfg)
    print(predict_next(cfg, res))
"""
from .config import (BacktestConfig, Config, DataConfig, FeatureConfig,
                     LabelConfig, ModelConfig, ValidationConfig)

__version__ = "1.2.0"
ARTIFACT_FORMAT = 3
__all__ = ["Config", "DataConfig", "LabelConfig", "FeatureConfig", "ModelConfig",
           "ValidationConfig", "BacktestConfig", "run_experiment", "predict_next",
           "run_all_checks", "__version__", "ARTIFACT_FORMAT"]


def __getattr__(name):
    # importación perezosa: `import alphaforge` no debe arrastrar sklearn entero
    if name in ("run_experiment", "predict_next"):
        from . import pipeline
        return getattr(pipeline, name)
    if name == "run_all_checks":
        from .selfcheck import run_all
        return run_all
    raise AttributeError(name)
