"""Validación honesta: el módulo que decide si el modelo vale algo o no.

Contiene las tres defensas contra el autoengaño:

1. PurgedWalkForward  - splits temporales con purga y embargo. Sin esto, las
   etiquetas solapadas entre train y test inflan cualquier métrica.

2. PBO vía CSCV       - Probability of Backtest Overfitting (Bailey, Borwein,
   López de Prado & Zhu, 2015). Responde: "de todas las configuraciones que
   probé, ¿con qué probabilidad la mejor en muestra queda por debajo de la
   mediana fuera de muestra?" Si sale > 0.35, el proceso de selección está
   pescando ruido y el número bonito del backtest no significa nada.

3. Deflated Sharpe    - ajusta el Sharpe por el número de pruebas realizadas,
   la asimetría y la curtosis. Un Sharpe de 1.5 tras 500 intentos vale menos
   que un 0.8 al primer intento.

Más el test de permutación: se reentrena con etiquetas barajadas y se compara
la métrica real contra la distribución nula. Da un p-valor empírico directo.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .utils import get_logger


# ---------------------------------------------------------------------------
# 1. Splits temporales purgados
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    fold: int
    train_span: tuple[pd.Timestamp, pd.Timestamp]
    test_span: tuple[pd.Timestamp, pd.Timestamp]


class PurgedWalkForward:
    """Walk-forward con purga y embargo.

    Purga: elimina del train las observaciones cuyo periodo de etiqueta se
    solapa con el test. Con horizonte h, la etiqueta de t usa información hasta
    t+h, luego cualquier train con índice en [test_start - h, test_start) está
    contaminado.

    Embargo: además retira una fracción de observaciones inmediatamente
    posteriores al test, porque la autocorrelación serial de las features
    (medias móviles de 200 días, por ejemplo) crea dependencia hacia atrás.
    """

    def __init__(self, n_splits: int = 8, purge: int = 3, embargo_pct: float = 0.01,
                 min_train: int = 500, anchored: bool = True):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo_pct = embargo_pct
        self.min_train = min_train
        self.anchored = anchored

    def split(self, index: pd.DatetimeIndex) -> list[Split]:
        n = len(index)
        if n < self.min_train + self.n_splits * 20:
            raise ValueError(
                f"Insuficientes datos: {n} filas para {self.n_splits} folds con "
                f"min_train={self.min_train}. Reduce n_splits o amplía el histórico.")

        usable = n - self.min_train
        test_size = usable // self.n_splits
        embargo = int(np.ceil(self.embargo_pct * n))
        out: list[Split] = []

        for k in range(self.n_splits):
            t0 = self.min_train + k * test_size
            t1 = n if k == self.n_splits - 1 else t0 + test_size
            test_idx = np.arange(t0, t1)
            if len(test_idx) < 10:
                continue

            train_end = t0 - self.purge
            train_start = 0 if self.anchored else max(0, train_end - self.min_train)
            if train_end - train_start < max(100, self.min_train // 2):
                continue
            train_idx = np.arange(train_start, train_end)

            # Embargo. En walk-forward anclado el train siempre precede al test,
            # así que un embargo "posterior" no elimina nada: era código muerto
            # que daba falsa sensación de protección. Lo que sí protege aquí es
            # ampliar la separación hacia atrás, porque las features con ventanas
            # largas (medias de 200 días) crean dependencia del train con el
            # inicio del test.
            if embargo > 0:
                cut = t0 - self.purge - embargo
                train_idx = train_idx[train_idx < cut] if cut > 0 else train_idx[:0]
                if len(train_idx) < max(100, self.min_train // 2):
                    continue

            out.append(Split(
                train=train_idx, test=test_idx, fold=k,
                train_span=(index[train_idx[0]], index[train_idx[-1]]),
                test_span=(index[test_idx[0]], index[test_idx[-1]]),
            ))
        if not out:
            raise ValueError("PurgedWalkForward no generó ningún fold válido.")
        return out


def assert_no_overlap(splits: list[Split], purge: int) -> None:
    """Guarda de seguridad: ningún índice de train puede invadir el test."""
    for s in splits:
        if len(np.intersect1d(s.train, s.test)):
            raise AssertionError(f"fold {s.fold}: train y test se solapan")
        if len(s.train) and s.train.max() >= s.test.min() - purge + 1:
            raise AssertionError(
                f"fold {s.fold}: purga insuficiente "
                f"(train_max={s.train.max()}, test_min={s.test.min()}, purge={purge})")


# ---------------------------------------------------------------------------
# 2. PBO por CSCV
# ---------------------------------------------------------------------------
@dataclass
class PBOResult:
    pbo: float
    n_configs: int
    n_combinations: int
    logits: np.ndarray = field(repr=False)
    is_oos_slope: float = np.nan
    is_oos_corr: float = np.nan
    median_oos_of_best: float = np.nan


def compute_pbo(perf_matrix: np.ndarray, n_blocks: int = 12) -> PBOResult:
    """Probability of Backtest Overfitting (CSCV).

    perf_matrix: (T, N) con el retorno (o métrica por periodo) de N
    configuraciones a lo largo de T periodos.

    Procedimiento: se parte T en S bloques, se recorren todas las C(S, S/2)
    formas de dividirlos en (entrenamiento, prueba), se elige el mejor
    candidato en entrenamiento y se mira su rango relativo en prueba. El logit
    de ese rango, agregado, da la probabilidad de que el ganador en muestra sea
    mediocre fuera de ella.
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return PBOResult(pbo=np.nan, n_configs=int(M.shape[1] if M.ndim == 2 else 0),
                         n_combinations=0, logits=np.array([]))
    T, N = M.shape
    S = int(n_blocks)
    if S % 2:
        S -= 1
    S = max(4, min(S, T // 3 if T >= 12 else 4))
    if T < S * 2:
        S = max(4, (T // 2) // 2 * 2)
    if S < 4 or T < 8:
        return PBOResult(pbo=np.nan, n_configs=N, n_combinations=0, logits=np.array([]))

    blocks = np.array_split(np.arange(T), S)
    combos = list(itertools.combinations(range(S), S // 2))
    if len(combos) > 2000:                     # control de coste
        rng = np.random.default_rng(0)
        sel = rng.choice(len(combos), 2000, replace=False)
        combos = [combos[i] for i in sel]

    logits, is_perf, oos_perf = [], [], []
    for c in combos:
        tr_b = set(c)
        tr = np.concatenate([blocks[i] for i in range(S) if i in tr_b])
        te = np.concatenate([blocks[i] for i in range(S) if i not in tr_b])
        if len(tr) < 3 or len(te) < 3:
            continue
        r_is = _sharpe_cols(M[tr])
        r_oos = _sharpe_cols(M[te])
        if not np.isfinite(r_is).any():
            continue
        best = int(np.nanargmax(r_is))
        # rango relativo del ganador fuera de muestra
        finite = np.isfinite(r_oos)
        if finite.sum() < 2:
            continue
        rank = stats.rankdata(np.where(finite, r_oos, -np.inf))[best]
        w = rank / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
        is_perf.append(r_is[best])
        oos_perf.append(r_oos[best])

    if not logits:
        return PBOResult(pbo=np.nan, n_configs=N, n_combinations=0, logits=np.array([]))

    logits = np.asarray(logits)
    pbo = float(np.mean(logits <= 0))

    slope, corr = np.nan, np.nan
    is_perf, oos_perf = np.asarray(is_perf), np.asarray(oos_perf)
    ok = np.isfinite(is_perf) & np.isfinite(oos_perf)
    if ok.sum() > 5 and np.std(is_perf[ok]) > 1e-9:
        slope = float(np.polyfit(is_perf[ok], oos_perf[ok], 1)[0])
        corr = float(np.corrcoef(is_perf[ok], oos_perf[ok])[0, 1])

    return PBOResult(pbo=pbo, n_configs=N, n_combinations=len(logits), logits=logits,
                     is_oos_slope=slope, is_oos_corr=corr,
                     median_oos_of_best=float(np.nanmedian(oos_perf)))


def _sharpe_cols(a: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=1)
    out = np.divide(mu, sd, out=np.full(a.shape[1], np.nan), where=sd > 1e-12)
    return out


# ---------------------------------------------------------------------------
# 3. Deflated / Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------
def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float:
    """PSR: probabilidad de que el Sharpe verdadero supere el benchmark."""
    if n < 10 or not np.isfinite(sr):
        return np.nan
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float,
                    n_trials: int, var_trials_sr: float) -> tuple[float, float]:
    """DSR: PSR contra un benchmark que ya incorpora el sesgo de selección.

    El umbral esperado del máximo de N Sharpes independientes de varianza
    var_trials_sr crece con log(N); comparar contra 0 sería tramposo.
    """
    if n_trials < 2 or not np.isfinite(var_trials_sr) or var_trials_sr <= 0:
        return probabilistic_sharpe(sr, n, skew, kurt, 0.0), 0.0
    e = np.euler_gamma
    N = float(n_trials)
    z1 = stats.norm.ppf(1 - 1 / N)
    z2 = stats.norm.ppf(1 - 1 / (N * np.e))
    sr0 = np.sqrt(var_trials_sr) * ((1 - e) * z1 + e * z2)
    return probabilistic_sharpe(sr, n, skew, kurt, sr0), float(sr0)


# ---------------------------------------------------------------------------
# 4. Test de permutación
# ---------------------------------------------------------------------------
def permutation_pvalue(observed: float, null_dist: np.ndarray,
                       higher_is_better: bool = True) -> float:
    null = np.asarray(null_dist, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return np.nan
    if higher_is_better:
        hits = int(np.sum(null >= observed))
    else:
        hits = int(np.sum(null <= observed))
    return (hits + 1) / (len(null) + 1)          # corrección de continuidad


def block_permute(y: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Permutación por bloques: rompe la relación X->y pero preserva la
    autocorrelación de y. Un shuffle simple destruiría la estructura serial y
    daría un nulo demasiado fácil de batir."""
    n = len(y)
    block = max(1, min(int(block), n // 4 if n >= 8 else 1))
    n_blocks = int(np.ceil(n / block))
    idx = rng.permutation(n_blocks)
    parts = [y[i * block:(i + 1) * block] for i in idx]
    return np.concatenate(parts)[:n]


# ---------------------------------------------------------------------------
# 5. Calidad de la calibración probabilística
# ---------------------------------------------------------------------------
def calibration_report(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Brier, log-loss y desviación de calibración (ECE)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return {"brier": np.nan, "log_loss": np.nan, "ece": np.nan,
                "brier_skill": np.nan, "n": int(len(y))}

    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean(y))
    brier_ref = base * (1 - base)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    curve = []
    for b in range(bins):
        m = which == b
        if m.sum() < 5:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
        curve.append({"bin": b, "p_pred": float(p[m].mean()),
                      "p_obs": float(y[m].mean()), "n": int(m.sum())})
    return {"brier": brier, "log_loss": ll, "ece": float(ece),
            "brier_skill": float(1 - brier / brier_ref) if brier_ref > 0 else np.nan,
            "base_rate": base, "n": int(len(y)), "curve": curve}


def stability_report(fold_metrics: list[dict], key: str = "sharpe") -> dict:
    """Consistencia entre folds: un sistema real no vive de un solo periodo."""
    v = np.array([m.get(key, np.nan) for m in fold_metrics], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"mean": np.nan, "std": np.nan, "frac_positive": np.nan,
                "worst": np.nan, "t_stat": np.nan}
    t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 1e-12 else np.nan
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "frac_positive": float((v > 0).mean()), "worst": float(v.min()),
            "best": float(v.max()), "t_stat": t, "n_folds": int(len(v))}
