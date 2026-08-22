"""Utilidades transversales: logging, guardas numéricas, reproducibilidad."""
from __future__ import annotations

import logging
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

_LOGGER_NAME = "alphaforge"


class DataIntegrityError(RuntimeError):
    """Los datos no cumplen las invariantes mínimas."""


class LeakageError(RuntimeError):
    """Se ha detectado fuga de información del futuro."""


def get_logger(verbose: int = 1) -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    log.setLevel({0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.INFO))
    log.propagate = False
    return log


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextmanager
def timer(name: str, log: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    msg = f"{name} completado en {dt:.1f}s"
    (log.info(msg) if log else print(msg))


# ---------------------------------------------------------------------------
# Guardas numéricas
# ---------------------------------------------------------------------------
def sanitize(df: pd.DataFrame | pd.Series, name: str = "matriz",
             clip_sigma: float = 8.0) -> pd.DataFrame | pd.Series:
    """Sustituye inf por NaN y winsoriza colas extremas.

    NO rellena NaN: eso se decide aguas arriba con conocimiento del contexto
    temporal (rellenar hacia delante es legítimo, hacia atrás es fuga).
    """
    out = df.replace([np.inf, -np.inf], np.nan)
    if isinstance(out, pd.Series):
        out = out.to_frame()
        was_series = True
    else:
        was_series = False
    num = out.select_dtypes(include=[np.number])
    if len(num.columns) and clip_sigma > 0:
        mu = num.median()
        sd = num.std(ddof=0).replace(0, np.nan)
        lo, hi = mu - clip_sigma * sd, mu + clip_sigma * sd
        out[num.columns] = num.clip(lower=lo, upper=hi, axis=1)
    return out.iloc[:, 0] if was_series else out


def assert_finite(a: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(a)):
        n_nan = int(np.isnan(a).sum())
        n_inf = int(np.isinf(a).sum())
        raise DataIntegrityError(
            f"{name} contiene valores no finitos (NaN={n_nan}, Inf={n_inf})."
        )


def safe_div(a, b, fill: float = 0.0):
    """División protegida elemento a elemento."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.divide(a, b)
    if isinstance(r, np.ndarray):
        r[~np.isfinite(r)] = fill
        return r
    return fill if not np.isfinite(r) else r


def pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    prev = s.shift(periods)
    out = (s / prev.where(prev.abs() > 1e-12) - 1.0)
    return out.replace([np.inf, -np.inf], np.nan)


def zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(5, window // 2)
    m = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std(ddof=0)
    return (s - m) / sd.where(sd > 1e-12)


def suppress_noisy_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
    warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")
    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


def fmt_pct(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def fmt_num(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
