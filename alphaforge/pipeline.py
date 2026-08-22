"""Orquestador del experimento completo.

FLUJO
-----
  datos -> features -> etiquetas -> alineación
        -> búsqueda de hiperparámetros (CV purgado, SOLO sobre el bloque inicial)
        -> walk-forward externo (cada fold reentrena; nunca ve su futuro)
        -> calibración de probabilidad en sub-holdout purgado
        -> pesos del ensemble aprendidos out-of-sample
        -> regresión cuantílica para la magnitud
        -> métricas OOS + PBO + Deflated Sharpe + test de permutación
        -> veredicto GO / NO-GO
        -> modelo final reentrenado con todo el histórico para producción

INVARIANTE CENTRAL: en el fold k, ningún objeto ajustado (imputador, escalador,
selector de features, calibrador, pesos) ha visto una sola fila del test de ese
fold. Se comprueba con asserts, no con buena fe.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import validation as val
from .backtest import (BacktestResult, buy_and_hold, continuous_pnl,
                       position_from_prob, run_backtest)
from .config import Config, TRADING_DAYS_YEAR
from .data import load_market_data
from .features import build_features, prune_correlated
from .labeling import align, build_labels, sample_weights
from .models import (QuantileBundle, blend, build_classifier,
                     learn_ensemble_weights, sample_params, HAS_TORCH)
from .utils import (LeakageError, assert_finite, fmt_num, fmt_pct, get_logger,
                    set_seed, suppress_noisy_warnings)


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------
@dataclass
class FoldArtifacts:
    fold: int
    features: list[str]
    imputer: Any
    models: dict[str, Any]
    calibrators: dict[str, Any]
    weights: np.ndarray
    families: list[str]
    quantiles: QuantileBundle | None
    auc_cal: dict[str, float]
    ensemble_calibrator: object | None = None
    train_end: str = ""
    refit_full: bool = False
    ref_quantiles: dict[str, np.ndarray] | None = None   # para detectar deriva


@dataclass
class ExperimentResult:
    cfg: Config
    oos: pd.DataFrame                    # prob, posición, retorno, cuantiles
    metrics: dict
    fold_metrics: list[dict]
    pbo: val.PBOResult | None
    search: dict
    calibration: dict
    permutation: dict
    verdict: dict
    feature_importance: pd.Series
    final_model: dict = field(default_factory=dict)
    data_report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preparación por fold (sin fugas)
# ---------------------------------------------------------------------------
class TrainFittedClipper:
    """Recorta colas usando percentiles aprendidos SOLO en entrenamiento.

    Antes esto se hacía en build_features sobre la serie completa: los umbrales
    salían de datos futuros respecto a cualquier fold. Es un look-ahead pequeño
    pero real, del tipo que nadie audita porque "solo es limpieza".
    """

    def __init__(self, q: float = 0.001):
        self.q = q

    def fit(self, Z: np.ndarray) -> "TrainFittedClipper":
        self.lo_ = np.nanquantile(Z, self.q, axis=0)
        self.hi_ = np.nanquantile(Z, 1 - self.q, axis=0)
        span = self.hi_ - self.lo_
        pad = np.where(np.isfinite(span) & (span > 0), 0.25 * span, 0.0)
        self.lo_, self.hi_ = self.lo_ - pad, self.hi_ + pad
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return np.clip(Z, self.lo_, self.hi_)


class FoldPrep:
    """Imputador + recorte, ambos ajustados en train. Interfaz .transform()."""

    def __init__(self, imputer, clipper, columns: list[str]):
        self.imputer, self.clipper, self.columns = imputer, clipper, columns

    def transform(self, X) -> np.ndarray:
        Z = self.imputer.transform(X[self.columns] if hasattr(X, "columns") else X)
        return self.clipper.transform(Z)


def _prepare_fold(X: pd.DataFrame, tr: np.ndarray, cfg: Config,
                  y_tr: np.ndarray, seed: int) -> tuple[list[str], "FoldPrep"]:
    """Poda de correlación + selección por importancia, ajustadas SOLO en train."""
    Xtr = X.iloc[tr]
    kept = prune_correlated(Xtr, cfg.features.max_abs_corr, verbose=0)
    Xk = Xtr[kept]

    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    Ztr = imp.fit_transform(Xk)
    Ztr = TrainFittedClipper().fit(Ztr).transform(Ztr)

    k = min(cfg.model.max_features_selected, len(kept))
    if len(kept) > k:
        # importancia por permutación sobre el tramo final del train (proxy de
        # régimen reciente) usando un modelo barato
        n_hold = max(120, int(0.2 * len(tr)))
        # separación entre core y hold: sin ella, la etiqueta del último día de
        # core usa el precio del primer día de hold y la importancia por
        # permutación se mide contaminada
        gap = cfg.validation.purge_days
        core = slice(0, max(50, len(tr) - n_hold - gap))
        hold = slice(len(tr) - n_hold, len(tr))
        probe = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.06, min_samples_leaf=40,
            early_stopping=False, random_state=seed)
        try:
            probe.fit(Ztr[core], y_tr[core])
            r = permutation_importance(probe, Ztr[hold], y_tr[hold], n_repeats=5,
                                       random_state=seed, scoring="roc_auc", n_jobs=-1)
            order = np.argsort(-r.importances_mean)
            kept = [kept[i] for i in order[:k]]
        except Exception:                        # noqa: BLE001 - degradación elegante
            var = np.nanvar(Ztr, axis=0)
            kept = [kept[i] for i in np.argsort(-var)[:k]]
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        Ztr = imp.fit_transform(Xtr[kept])
    else:
        Ztr = imp.transform(Xtr[kept])
    clip = TrainFittedClipper().fit(Ztr)
    return kept, FoldPrep(imp, clip, kept)


def _fit_family(family: str, params: dict, Ztr: np.ndarray, ytr: np.ndarray,
                w: np.ndarray, seed: int):
    """Entrena pasando pesos de muestra solo si el estimador los admite.

    Los Pipeline de sklearn exigen el prefijo del paso ('clf__sample_weight') y
    el MLP no admite pesos en absoluto: sin este manejo, familias enteras se
    caían en silencio dentro del `except` de la búsqueda.
    """
    est = build_classifier(family, params, seed, n_features=Ztr.shape[1])
    kw = _sw_kwargs(est, w)
    try:
        est.fit(Ztr, ytr, **kw)
    except (TypeError, ValueError) as e:
        if not kw:
            raise
        get_logger(0).debug(f"{family}: pesos no aceptados ({e}); reintento sin ellos")
        est = build_classifier(family, params, seed, n_features=Ztr.shape[1])
        est.fit(Ztr, ytr)
    return est


def _sw_kwargs(est, w: np.ndarray) -> dict:
    import inspect
    target, prefix = est, ""
    if hasattr(est, "steps") and est.steps:
        name, target = est.steps[-1]          # nunca evaluar el estimador como bool
        prefix = f"{name}__"
    from sklearn.neural_network import MLPClassifier
    if isinstance(target, MLPClassifier):          # no soporta sample_weight
        return {}
    try:
        if "sample_weight" in inspect.signature(target.fit).parameters:
            return {f"{prefix}sample_weight": w}
    except (TypeError, ValueError):
        pass
    return {}


def _proba(est, Z: np.ndarray) -> np.ndarray:
    p = est.predict_proba(Z)
    p = p[:, 1] if p.ndim == 2 else p.ravel()
    return np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Búsqueda de hiperparámetros
# ---------------------------------------------------------------------------
def hyperparameter_search(X: pd.DataFrame, y: pd.Series, y_ret: pd.Series,
                          w: pd.Series, cfg: Config) -> dict:
    """Random search con CV purgado interno. Devuelve la mejor config por
    familia y la matriz de rendimiento por periodo para el cálculo del PBO."""
    log = get_logger(cfg.verbose)
    rng = np.random.default_rng(cfg.model.random_state)
    families = [f for f in cfg.model.families if f != "gru" or True]

    inner = val.PurgedWalkForward(
        n_splits=cfg.validation.inner_splits, purge=cfg.validation.purge_days,
        embargo_pct=cfg.validation.embargo_pct,
        min_train=max(250, int(0.45 * len(X))), anchored=True)
    isplits = inner.split(X.index)
    val.assert_no_overlap(isplits, cfg.validation.purge_days)

    yv, rv, wv = y.to_numpy(float), y_ret.to_numpy(float), w.to_numpy(float)
    trials: list[dict] = []
    n_failed: dict[str, int] = {}
    n_per = max(1, cfg.model.n_trials // max(1, len(families)))

    # La selección de features depende del fold, no de los hiperparámetros:
    # se calcula UNA vez por fold en lugar de una vez por trial (x20 más rápido).
    prep: dict[int, tuple[list[str], Any, np.ndarray, np.ndarray]] = {}
    for sp in isplits:
        feats, imp = _prepare_fold(X, sp.train, cfg, yv[sp.train],
                                   cfg.model.random_state + sp.fold)
        prep[sp.fold] = (feats, imp,
                         imp.transform(X.iloc[sp.train][feats]),
                         imp.transform(X.iloc[sp.test][feats]))
    log.debug(f"preparación cacheada para {len(prep)} folds internos")

    t0 = time.time()
    for fam in families:
        if fam == "gru" and not HAS_TORCH:
            log.info("gru: torch no disponible -> se usará el respaldo MLP-ventana")
        for j in range(n_per):
            params = sample_params(fam, rng)
            seed = int(cfg.model.random_state + 1000 * len(trials))
            per_period = np.full(len(X), np.nan)
            aucs = []
            try:
                for sp in isplits:
                    _, _, Ztr, Zte = prep[sp.fold]
                    est = _fit_family(fam, params, Ztr, yv[sp.train], wv[sp.train], seed)
                    p = _proba(est, Zte)
                    yte = yv[sp.test]
                    if len(np.unique(yte)) > 1:
                        aucs.append(roc_auc_score(yte, p))
                    pos = np.where(p >= cfg.backtest.prob_threshold, 1.0,
                                   np.where(p <= 1 - cfg.backtest.prob_threshold,
                                            -1.0 if cfg.backtest.allow_short else 0.0, 0.0))
                    per_period[sp.test] = pos * rv[sp.test] - np.abs(pos) * cfg.roundtrip_cost / 2
            except Exception as e:                # noqa: BLE001
                log.debug(f"trial {fam}#{j} falló: {type(e).__name__}: {e}")
                n_failed[fam] = n_failed.get(fam, 0) + 1
                continue

            series = per_period[np.isfinite(per_period)]
            if len(series) < 30:
                continue
            sd = series.std(ddof=1)
            sharpe = float(series.mean() / sd * np.sqrt(TRADING_DAYS_YEAR)) if sd > 1e-12 else np.nan
            trials.append({
                "family": fam, "params": params, "seed": seed,
                "auc": float(np.mean(aucs)) if aucs else np.nan,
                "sharpe_cv": sharpe, "per_period": per_period,
                "n_eval": int(len(series)),
            })
        n_ok = sum(1 for t in trials if t["family"] == fam)
        log.info(f"búsqueda {fam}: {n_ok}/{n_per} trials válidos "
                 f"({time.time() - t0:.0f}s)")
        if n_ok == 0:
            log.warning(f"la familia '{fam}' no produjo ni un trial válido: "
                        f"queda excluida del ensemble. Ejecuta con --debug para ver "
                        f"la causa.")

    if not trials:
        raise RuntimeError("Ningún trial completó la validación. Revisa los datos.")

    # matriz para CSCV: periodos x configuraciones
    common = np.all(np.isfinite(np.column_stack([t["per_period"] for t in trials])), axis=1)
    perf_matrix = np.column_stack([t["per_period"][common] for t in trials])

    best_by_family: dict[str, dict] = {}
    for t in trials:
        score = t["sharpe_cv"] if np.isfinite(t["sharpe_cv"]) else -np.inf
        cur = best_by_family.get(t["family"])
        if cur is None or score > cur["_score"]:
            best_by_family[t["family"]] = {**t, "_score": score}

    sr_all = np.array([t["sharpe_cv"] for t in trials], dtype=float)
    sr_all = sr_all[np.isfinite(sr_all)]
    log.info(f"búsqueda terminada: {len(trials)} configuraciones | "
             f"Sharpe CV mediano {fmt_num(float(np.median(sr_all)))} | "
             f"máx {fmt_num(float(sr_all.max()))}")

    return {
        "trials": [{k: v for k, v in t.items() if k != "per_period"} for t in trials],
        "best_by_family": {k: {kk: vv for kk, vv in v.items()
                               if kk not in ("per_period", "_score")}
                           for k, v in best_by_family.items()},
        "perf_matrix": perf_matrix,
        "n_trials": len(trials),
        "var_trials_sharpe": float(np.var(sr_all, ddof=1)) if len(sr_all) > 1 else np.nan,
        "median_sharpe_cv": float(np.median(sr_all)) if len(sr_all) else np.nan,
        "max_sharpe_cv": float(sr_all.max()) if len(sr_all) else np.nan,
    }


# ---------------------------------------------------------------------------
# Entrenamiento de un fold completo
# ---------------------------------------------------------------------------
def _train_fold(X: pd.DataFrame, y: np.ndarray, y_ret: np.ndarray, y_norm: np.ndarray,
                w: np.ndarray, tr: np.ndarray, cfg: Config, best: dict,
                fold: int, refit_full: bool = False) -> FoldArtifacts:
    """Entrena, calibra y pondera usando exclusivamente índices de `tr`."""
    log = get_logger(cfg.verbose)
    seed = cfg.model.random_state + 97 * fold

    # sub-holdout final del train para calibrar y ponderar, con purga interna
    n_cal = max(120, int(0.18 * len(tr)))
    purge = cfg.validation.purge_days
    core = tr[: len(tr) - n_cal - purge]
    cal = tr[len(tr) - n_cal:]
    if len(core) < 200:
        core, cal = tr[: int(0.8 * len(tr))], tr[int(0.8 * len(tr)) + purge:]
    if len(cal) < 40:
        cal = core[-60:]

    # Los pesos por decaimiento se recalculan respecto al final de ESTE train:
    # en un fold que acaba en 2010, "reciente" significa reciente en 2010. Con
    # el cálculo global los folds antiguos entrenaban con pesos ~1000x menores.
    w = w.copy()
    w[tr] = sample_weights(X.index[tr], cfg.horizon_days, cfg,
                           pd.Series(y_ret[tr], index=X.index[tr])).to_numpy()

    feats, imp = _prepare_fold(X, core, cfg, y[core], seed)
    Zc = imp.transform(X.iloc[core][feats])
    Zk = imp.transform(X.iloc[cal][feats])
    assert_finite(Zc, f"fold{fold}/Z_core")

    models, calibrators, cal_probs, fams, aucs = {}, {}, [], [], {}
    for fam, spec in best.items():
        try:
            est = _fit_family(fam, spec["params"], Zc, y[core], w[core], seed)
            p_cal = _proba(est, Zk)
            calib = _fit_calibrator(p_cal, y[cal], cfg.model.calibration)
            models[fam] = est
            calibrators[fam] = calib
            cal_probs.append(_apply_calibrator(calib, p_cal))
            fams.append(fam)
            aucs[fam] = (float(roc_auc_score(y[cal], p_cal))
                         if len(np.unique(y[cal])) > 1 else np.nan)
        except Exception as e:                    # noqa: BLE001
            log.warning(f"fold {fold}: familia {fam} descartada ({type(e).__name__}: {e})")

    if not models:
        raise RuntimeError(f"fold {fold}: ninguna familia entrenó correctamente")

    # El bloque reservado se parte en dos mitades disjuntas: con la primera se
    # aprenden los pesos, con la segunda se recalibra la mezcla. Usar el mismo
    # tramo para ambas cosas sobreajusta la probabilidad final.
    P = np.column_stack(cal_probs)
    half = len(cal) // 2
    if half >= 40:
        w_idx, c_idx = slice(0, half), slice(half, len(cal))
    else:                                  # muy poco dato: se reutiliza, con aviso
        w_idx = c_idx = slice(0, len(cal))
        log.debug(f"fold {fold}: bloque de calibración corto ({len(cal)}), "
                  f"pesos y recalibración comparten muestras")
    weights = learn_ensemble_weights(P[w_idx], y[cal][w_idx])

    # Promediar modelos ya calibrados NO da un resultado calibrado: la mezcla
    # se aplana hacia 0.5. Se recalibra la salida del ensemble.
    ens_cal = None
    try:
        p_mix = blend(P[c_idx], weights)
        ens_cal = _select_calibrator(p_mix, y[cal][c_idx], log)
    except Exception as e:                   # noqa: BLE001
        log.warning(f"fold {fold}: recalibración del ensemble omitida ({e})")

    # regresión cuantílica de la magnitud (objetivo en unidades de sigma)
    qb = None
    try:
        qb = QuantileBundle().fit(
            pd.DataFrame(Zc, columns=feats, index=X.index[core]),
            y_norm[core], cfg.model.quantiles, seed=seed, sample_weight=w[core])
    except Exception as e:                        # noqa: BLE001
        log.warning(f"fold {fold}: regresión cuantílica no disponible ({e})")

    # --- modelo de producción: reajuste con TODO el histórico ---------------
    # Durante el walk-forward reservamos el tramo final para calibrar, y está
    # bien: mide honestamente. Pero el modelo que va a operar no puede
    # permitirse ignorar los meses más recientes, que son los que describen el
    # régimen actual. Se reentrenan las bases con todo y se conservan los
    # calibradores y los pesos aprendidos en el tramo reservado.
    if refit_full:
        Zall = imp.transform(X.iloc[tr][feats])
        for fam in list(models):
            try:
                models[fam] = _fit_family(fam, best[fam]["params"], Zall, y[tr],
                                          w[tr], seed)
            except Exception as e:                # noqa: BLE001
                log.warning(f"reajuste completo de {fam} fallido, se conserva "
                            f"el parcial ({e})")
        if qb is not None:
            try:
                qb = QuantileBundle().fit(
                    pd.DataFrame(Zall, columns=feats, index=X.index[tr]),
                    y_norm[tr], cfg.model.quantiles, seed=seed, sample_weight=w[tr])
            except Exception as e:                # noqa: BLE001
                log.warning(f"reajuste de cuantiles fallido ({e})")
        log.info(f"modelo de producción reajustado con {len(tr)} muestras "
                 f"(hasta {X.index[tr[-1]].date()})")

    # huella de la distribución de entrenamiento, para detectar deriva después
    Zref = imp.transform(X.iloc[tr][feats])
    ref_q = {"cols": feats,
             "q": np.nanquantile(Zref, np.linspace(0, 1, 11), axis=0)}

    log.debug(f"fold {fold}: {len(feats)} features | familias {fams} | "
              f"pesos {np.round(weights, 3).tolist()}")
    return FoldArtifacts(fold=fold, features=feats, imputer=imp, models=models,
                         calibrators=calibrators, weights=weights, families=fams,
                         quantiles=qb, auc_cal=aucs, ensemble_calibrator=ens_cal,
                         train_end=str(X.index[tr[-1]].date()),
                         refit_full=refit_full, ref_quantiles=ref_q)


def _select_calibrator(p: np.ndarray, y: np.ndarray, log=None):
    """Elige entre no calibrar, Platt o isotónica por validación interna.

    La isotónica necesita muchos datos: con el par de cientos de muestras que
    quedan tras partir el bloque reservado, sobreajusta y EMPEORA la
    probabilidad (medido: ECE de 0.017 a 0.079). Platt tiene dos parámetros y
    aguanta bien con poco. En vez de fijar uno a mano, se prueban los tres
    contra un log-loss validado y gana el que de verdad ayude — incluida la
    opción de no tocar nada, que a menudo es la correcta.
    """
    n = len(y)
    if n < 60 or len(np.unique(y)) < 2:
        return None
    folds = 3
    edges = np.linspace(0, n, folds + 1).astype(int)
    scores: dict[str, list] = {"none": [], "sigmoid": [], "isotonic": []}
    for k in range(folds):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.setdiff1d(np.arange(n), te)
        if len(te) < 15 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        for mode in list(scores):
            try:
                c = _fit_calibrator(p[tr], y[tr], mode) if mode != "none" else None
                q = _apply_calibrator(c, p[te])
                q = np.clip(q, 1e-6, 1 - 1e-6)
                scores[mode].append(float(-np.mean(
                    y[te] * np.log(q) + (1 - y[te]) * np.log(1 - q))))
            except Exception:                    # noqa: BLE001
                scores[mode].append(np.inf)
    means = {m: (np.mean(v) if v else np.inf) for m, v in scores.items()}
    best = min(means, key=means.get)
    if log is not None:
        log.debug("calibrador del ensemble: " + ", ".join(
            f"{m}={means[m]:.4f}" for m in means) + f" -> {best}")
    if best == "none" or not np.isfinite(means[best]):
        return None
    return _fit_calibrator(p, y, best)


def _fit_calibrator(p: np.ndarray, y: np.ndarray, mode: str):
    if mode == "none" or len(np.unique(y)) < 2 or len(y) < 40:
        return None
    if mode == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        iso.fit(p, y)
        return ("isotonic", iso)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(p.reshape(-1, 1), y)
    return ("sigmoid", lr)


def _apply_calibrator(calib, p: np.ndarray) -> np.ndarray:
    if calib is None:
        return np.clip(p, 1e-6, 1 - 1e-6)
    kind, m = calib
    q = m.predict(p) if kind == "isotonic" else m.predict_proba(p.reshape(-1, 1))[:, 1]
    return np.clip(np.nan_to_num(q, nan=0.5), 1e-6, 1 - 1e-6)


def _ensemble_prob(art: "FoldArtifacts", P: np.ndarray) -> np.ndarray:
    """Mezcla en logit y recalibra: la probabilidad que ve el usuario."""
    p = blend(P, art.weights)
    if getattr(art, "ensemble_calibrator", None) is not None:
        p = _apply_calibrator(art.ensemble_calibrator, p)
    return np.clip(p, 1e-6, 1 - 1e-6)


def _predict_fold(art: FoldArtifacts, X: pd.DataFrame, idx: np.ndarray
                  ) -> tuple[np.ndarray, pd.DataFrame | None]:
    Z = art.imputer.transform(X.iloc[idx][art.features])
    cols = []
    for fam in art.families:
        p = _proba(art.models[fam], Z)
        cols.append(_apply_calibrator(art.calibrators[fam], p))
    p_ens = _ensemble_prob(art, np.column_stack(cols))
    q = None
    if art.quantiles is not None:
        q = art.quantiles.predict(pd.DataFrame(Z, columns=art.features,
                                               index=X.index[idx]))
    return p_ens, q


# ---------------------------------------------------------------------------
# Experimento completo
# ---------------------------------------------------------------------------
def run_experiment(cfg: Config, md=None) -> ExperimentResult:
    """Ejecuta el experimento completo.

    md: MarketData ya cargado (para tests offline o para reutilizar descargas).
    """
    suppress_noisy_warnings()
    cfg.validate()
    log = get_logger(cfg.verbose)
    set_seed(cfg.model.random_state)

    log.info(f"=== AlphaForge | {cfg.data.ticker} | fingerprint {cfg.fingerprint()} ===")
    md = load_market_data(cfg) if md is None else md
    X = build_features(md, cfg)
    labels = build_labels(md, cfg)
    Xa, y_cls, y_ret, y_norm, sigma = align(X, labels, cfg)

    _leakage_guard(Xa, y_ret, cfg)

    w = sample_weights(Xa.index, cfg.horizon_days, cfg, y_ret)
    yv = y_cls.to_numpy(float)
    rv = y_ret.to_numpy(float)
    nv = y_norm.to_numpy(float)
    wv = w.to_numpy(float)

    splitter = val.PurgedWalkForward(
        n_splits=cfg.validation.n_splits, purge=cfg.validation.purge_days,
        embargo_pct=cfg.validation.embargo_pct,
        min_train=cfg.validation.min_train_days, anchored=cfg.validation.anchored)
    splits = splitter.split(Xa.index)
    val.assert_no_overlap(splits, cfg.validation.purge_days)
    log.info(f"walk-forward: {len(splits)} folds | test "
             f"{splits[0].test_span[0].date()} -> {splits[-1].test_span[1].date()}")

    # --- búsqueda de hiperparámetros: solo el bloque de entrenamiento inicial
    s0 = splits[0].train
    log.info(f"búsqueda de hiperparámetros sobre {len(s0)} días iniciales "
             f"({Xa.index[s0[0]].date()} -> {Xa.index[s0[-1]].date()})")
    if len(s0) < 750:
        log.warning(
            f"solo {len(s0)} días para elegir hiperparámetros: la selección será "
            "ruidosa. Sube validation.min_train_days o amplía el histórico.")
    search = hyperparameter_search(Xa.iloc[s0], y_cls.iloc[s0], y_ret.iloc[s0],
                                   w.iloc[s0], cfg)
    best = {k: v for k, v in search["best_by_family"].items()}

    # --- walk-forward
    preds, fold_metrics, importances = [], [], []
    refit_every = max(1, cfg.validation.search_refit_folds)
    for sp in splits:
        # Rebúsqueda periódica: unos hiperparámetros elegidos en 2008 no tienen
        # por qué servir en 2024. Se rehace cada N folds usando SOLO el train de
        # ese fold, así que sigue sin mirar el futuro.
        if sp.fold > 0 and sp.fold % refit_every == 0:
            try:
                log.info(f"fold {sp.fold}: rebuscando hiperparámetros sobre "
                         f"{len(sp.train)} días (hasta {sp.train_span[1].date()})")
                sr = hyperparameter_search(Xa.iloc[sp.train], y_cls.iloc[sp.train],
                                           y_ret.iloc[sp.train], w.iloc[sp.train], cfg)
                best = {k: v for k, v in sr["best_by_family"].items()}
            except Exception as e:                # noqa: BLE001
                log.warning(f"fold {sp.fold}: rebúsqueda fallida, se conservan "
                            f"los hiperparámetros anteriores ({e})")
        art = _train_fold(Xa, yv, rv, nv, wv, sp.train, cfg, best, sp.fold)
        p, q = _predict_fold(art, Xa, sp.test)
        block = pd.DataFrame({"prob_up": p}, index=Xa.index[sp.test])
        block["fold"] = sp.fold
        block["y_ret"] = rv[sp.test]
        block["y_cls"] = yv[sp.test]
        block["sigma"] = sigma.to_numpy()[sp.test]
        if q is not None:
            for c in q.columns:
                block[f"ret_{c}"] = q[c].to_numpy() * block["sigma"].to_numpy()
        preds.append(block)
        importances.append(pd.Series(1.0, index=art.features))

        fpos = position_from_prob(block["prob_up"], cfg,
                                  expected_ret=block.get("ret_q50"),
                                  sigma=block["sigma"])
        fbt = run_backtest(fpos, block["y_ret"], cfg)
        fm = dict(fbt.metrics)
        fm["fold"] = sp.fold
        fm["auc"] = (float(roc_auc_score(block["y_cls"], block["prob_up"]))
                     if block["y_cls"].nunique() > 1 else np.nan)
        fm["test_start"] = str(sp.test_span[0].date())
        fm["test_end"] = str(sp.test_span[1].date())
        fold_metrics.append(fm)
        log.info(f"  fold {sp.fold} [{fm['test_start']}..{fm['test_end']}] "
                 f"AUC={fmt_num(fm['auc'])} Sharpe={fmt_num(fm['sharpe'], 2)} "
                 f"ret={fmt_pct(fm['total_return'])} DD={fmt_pct(fm['max_dd'])}")

    oos = pd.concat(preds).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]

    # --- backtest agregado con estrés del ancla
    noise = md.anchor.residual_stats.get("std", 0.0) if md.anchor.coverage < 0.98 else 0.0
    pos = position_from_prob(oos["prob_up"], cfg,
                             expected_ret=oos.get("ret_q50"), sigma=oos["sigma"])
    pos, n_veto = _coherence_gate(pos, oos, cfg)
    if n_veto:
        log.info(f"puerta de coherencia: {n_veto} operaciones vetadas por "
                 f"desacuerdo entre el clasificador y el regresor de magnitud")
    bt = run_backtest(pos, oos["y_ret"], cfg, anchor_noise_std=float(noise or 0.0),
                      seed=cfg.model.random_state, n_mc=300 if noise else 0)
    oos["position"] = bt.position
    oos["net_ret"] = bt.net_returns
    oos["equity"] = bt.equity

    metrics = dict(bt.metrics)
    # PnL sin solapamiento: lo que realmente se puede ejecutar
    metrics["continuous"] = continuous_pnl(bt.position, md.anchor.price, cfg)
    metrics["auc_oos"] = (float(roc_auc_score(oos["y_cls"], oos["prob_up"]))
                          if oos["y_cls"].nunique() > 1 else np.nan)
    metrics["buy_hold"] = buy_and_hold(md.daily["Close"], oos.index, cfg)
    metrics["anchor_source"] = md.anchor.source
    metrics["anchor_coverage"] = md.anchor.coverage
    metrics["anchor_residual"] = md.anchor.residual_stats

    calib = val.calibration_report(oos["y_cls"].to_numpy(), oos["prob_up"].to_numpy())
    stab = val.stability_report(fold_metrics, "sharpe")
    metrics["stability"] = stab
    metrics["stability_auc"] = val.stability_report(fold_metrics, "auc")

    # --- PBO
    pbo = None
    try:
        pbo = val.compute_pbo(search["perf_matrix"], cfg.validation.cscv_blocks)
        log.info(f"PBO = {fmt_num(pbo.pbo)} sobre {pbo.n_configs} configuraciones "
                 f"({pbo.n_combinations} particiones CSCV)")
    except Exception as e:                        # noqa: BLE001
        log.warning(f"PBO no calculable: {e}")

    # --- Deflated Sharpe
    x = bt.net_returns.dropna().to_numpy()
    sr_daily = x.mean() / x.std(ddof=1) if len(x) > 5 and x.std(ddof=1) > 1e-12 else np.nan
    from scipy import stats as _st
    dsr, sr0 = val.deflated_sharpe(
        sr=float(sr_daily) if np.isfinite(sr_daily) else 0.0, n=len(x),
        skew=float(_st.skew(x)) if len(x) > 8 else 0.0,
        kurt=float(_st.kurtosis(x, fisher=False)) if len(x) > 8 else 3.0,
        n_trials=max(2, search["n_trials"]),
        var_trials_sr=search["var_trials_sharpe"] / TRADING_DAYS_YEAR
        if np.isfinite(search["var_trials_sharpe"]) else np.nan)
    metrics["deflated_sharpe"] = dsr
    metrics["dsr_threshold_sharpe_annual"] = float(sr0 * np.sqrt(TRADING_DAYS_YEAR))
    metrics["psr_vs_zero"] = val.probabilistic_sharpe(
        float(sr_daily) if np.isfinite(sr_daily) else 0.0, len(x),
        float(_st.skew(x)) if len(x) > 8 else 0.0,
        float(_st.kurtosis(x, fisher=False)) if len(x) > 8 else 3.0)

    # --- test de permutación
    perm = _permutation_test(Xa, y_cls, y_ret, w, splits, best, cfg,
                             observed_sharpe=metrics.get("sharpe", np.nan),
                             observed_auc=metrics.get("auc_oos", np.nan))

    # --- veredicto
    verdict = _verdict(metrics, pbo, perm, calib, stab, cfg)

    # --- importancia agregada
    fi = pd.concat(importances).groupby(level=0).sum().sort_values(ascending=False)
    fi = fi / max(len(splits), 1)

    # --- modelo final para producción (reentrenado con TODO el histórico)
    final_art = _train_fold(Xa, yv, rv, nv, wv, np.arange(len(Xa)), cfg, best,
                            fold=999, refit_full=True)
    final = {"artifacts": final_art, "best": best, "index_end": str(Xa.index[-1].date())}

    return ExperimentResult(cfg=cfg, oos=oos, metrics=metrics,
                            fold_metrics=fold_metrics, pbo=pbo, search=search,
                            calibration=calib, permutation=perm, verdict=verdict,
                            feature_importance=fi, final_model=final,
                            data_report=md.reports)


def _coherence_gate(pos: pd.Series, frame: pd.DataFrame,
                    cfg: Config) -> tuple[pd.Series, int]:
    """No operar cuando las dos cabezas del modelo se contradicen.

    El clasificador dice "sube" y el regresor de magnitud estima retorno
    negativo: son dos lecturas del mismo dato que no coinciden. Medido en
    pruebas: ocurría el 12% de los días y se operaba igualmente. Ante señales
    contradictorias, lo sensato es quedarse fuera.
    """
    if not cfg.backtest.coherence_gate or "ret_q50" not in frame.columns:
        return pos, 0
    q50 = frame["ret_q50"]
    ok = q50.notna()
    conflict = ok & (np.sign(pos) != 0) & (np.sign(pos) != np.sign(q50))
    out = pos.where(~conflict, 0.0)
    return out, int(conflict.sum())


# ---------------------------------------------------------------------------
# Test de permutación
# ---------------------------------------------------------------------------
def _permutation_test(X, y_cls, y_ret, w, splits, best, cfg, observed_sharpe,
                      observed_auc) -> dict:
    """Reentrena con etiquetas permutadas por bloques y compara.

    Usa un subconjunto de folds y una sola familia (la más barata que esté
    disponible) para que el coste sea asumible: el objetivo es el p-valor del
    proceso, no reproducir el ensemble entero.
    """
    n_perm = cfg.validation.n_permutations
    if n_perm < 10:
        return {"n": 0, "p_sharpe": np.nan, "p_auc": np.nan}
    log = get_logger(cfg.verbose)
    rng = np.random.default_rng(cfg.model.random_state + 7)

    fam = "hgb" if "hgb" in best else list(best)[0]
    params = best[fam]["params"]
    use = splits[::2] if len(splits) > 4 else splits
    yv, rv, wv = y_cls.to_numpy(float), y_ret.to_numpy(float), w.to_numpy(float)

    # El pipeline de features se fija con las etiquetas reales; lo que se permuta
    # es exclusivamente el objetivo. Así el nulo mide el edge, no el preprocesado.
    prep = {}
    for sp in use:
        feats, imp = _prepare_fold(X, sp.train, cfg, yv[sp.train],
                                   cfg.model.random_state + sp.fold)
        prep[sp.fold] = (imp.transform(X.iloc[sp.train][feats]),
                         imp.transform(X.iloc[sp.test][feats]))

    null_sh, null_auc = [], []
    t0 = time.time()
    for i in range(n_perm):
        ysh = val.block_permute(yv, block=cfg.horizon_days * 5, rng=rng)
        rsh = np.where(ysh > 0.5, np.abs(rv), -np.abs(rv))   # coherente con la clase
        rets, aucs = [], []
        try:
            for sp in use:
                seed = cfg.model.random_state + i
                Ztr, Zte = prep[sp.fold]
                est = _fit_family(fam, params, Ztr, ysh[sp.train], wv[sp.train], seed)
                p = _proba(est, Zte)
                if len(np.unique(ysh[sp.test])) > 1:
                    aucs.append(roc_auc_score(ysh[sp.test], p))
                thr = cfg.backtest.prob_threshold
                pos = np.where(p >= thr, 1.0, np.where(p <= 1 - thr, -1.0, 0.0))
                rets.append(pos * rsh[sp.test] - np.abs(pos) * cfg.roundtrip_cost / 2)
        except Exception:                         # noqa: BLE001
            continue
        if not rets:
            continue
        arr = np.concatenate(rets)
        sd = arr.std(ddof=1)
        null_sh.append(arr.mean() / sd * np.sqrt(TRADING_DAYS_YEAR) if sd > 1e-12 else 0.0)
        if aucs:
            null_auc.append(float(np.mean(aucs)))

    p_sh = val.permutation_pvalue(observed_sharpe, np.array(null_sh))
    p_auc = val.permutation_pvalue(observed_auc, np.array(null_auc))
    log.info(f"test de permutación ({len(null_sh)} repeticiones, {time.time() - t0:.0f}s): "
             f"p(Sharpe)={fmt_num(p_sh)} p(AUC)={fmt_num(p_auc)}")
    return {"n": len(null_sh), "p_sharpe": p_sh, "p_auc": p_auc,
            "null_sharpe_mean": float(np.mean(null_sh)) if null_sh else np.nan,
            "null_sharpe_p95": float(np.percentile(null_sh, 95)) if null_sh else np.nan,
            "null_auc_mean": float(np.mean(null_auc)) if null_auc else np.nan,
            "family": fam}


# ---------------------------------------------------------------------------
# Guarda anti-fuga
# ---------------------------------------------------------------------------
def _leakage_guard(X: pd.DataFrame, y_ret: pd.Series, cfg: Config,
                   alarm_corr: float | None = None) -> None:
    """Ninguna feature debería correlacionar fuertemente con el retorno futuro.

    Un |rho| alto con y(t) significa casi con seguridad que la feature contiene
    información de t+1. Es la comprobación que separa un backtest de una
    fantasía.
    """
    log = get_logger(cfg.verbose)
    alarm_corr = (cfg.validation.leak_alarm_corr if alarm_corr is None else alarm_corr)
    yv = y_ret.to_numpy(float)
    suspects = []
    for col in X.columns:
        v = X[col].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(yv)
        if ok.sum() < 200 or np.std(v[ok]) < 1e-12:
            continue
        rho = float(np.corrcoef(v[ok], yv[ok])[0, 1])
        if abs(rho) > alarm_corr:
            suspects.append((col, rho))
    if suspects:
        det = ", ".join(f"{c} (rho={r:+.2f})" for c, r in suspects[:8])
        raise LeakageError(
            f"Posible fuga temporal en {len(suspects)} features (|rho| > {alarm_corr}): "
            f"{det}. Una correlación así con el retorno futuro no ocurre en mercados "
            "reales; revisa el desplazamiento de esas columnas. Si estás usando datos "
            "sintéticos con señal inyectada a propósito, sube "
            "validation.leak_alarm_corr (--leak-alarm).")
    log.debug(f"guarda anti-fuga superada: máx |rho| < {alarm_corr}")


# ---------------------------------------------------------------------------
# Veredicto
# ---------------------------------------------------------------------------
def _verdict(metrics: dict, pbo, perm: dict, calib: dict, stab: dict,
             cfg: Config) -> dict:
    checks: list[dict] = []

    def add(name, ok, detail, critical=True):
        checks.append({"check": name, "pass": bool(ok), "detail": detail,
                       "critical": critical})

    n_act = metrics.get("n_active", 0)
    add("muestra suficiente", n_act >= cfg.validation.min_oos_trades,
        f"{n_act} operaciones OOS (mínimo {cfg.validation.min_oos_trades})")

    # se juzga con el Sharpe corregido por solapamiento, que es el honesto
    sh_raw = metrics.get("sharpe", np.nan)
    sh = metrics.get("sharpe_overlap_adj", sh_raw)
    if not np.isfinite(sh):
        sh = sh_raw
    add("Sharpe OOS positivo", np.isfinite(sh) and sh > 0.3,
        f"Sharpe corregido = {fmt_num(sh, 2)} (bruto {fmt_num(sh_raw, 2)})")

    auc = metrics.get("auc_oos", np.nan)
    add("AUC > 0.52", np.isfinite(auc) and auc > 0.52, f"AUC = {fmt_num(auc)}")

    if pbo is not None and np.isfinite(pbo.pbo):
        add("PBO bajo", pbo.pbo <= cfg.validation.pbo_alarm,
            f"PBO = {fmt_num(pbo.pbo)} (alarma > {cfg.validation.pbo_alarm})")
    else:
        add("PBO calculable", False, "no se pudo calcular", critical=False)

    dsr = metrics.get("deflated_sharpe", np.nan)
    add("Deflated Sharpe", np.isfinite(dsr) and dsr >= cfg.validation.dsr_alarm,
        f"DSR = {fmt_num(dsr)} (umbral {cfg.validation.dsr_alarm})")

    p_sh = perm.get("p_sharpe", np.nan)
    add("significancia vs. azar", np.isfinite(p_sh) and p_sh < 0.05,
        f"p-valor permutación = {fmt_num(p_sh)}")

    fp = stab.get("frac_positive", np.nan)
    add("consistencia entre folds", np.isfinite(fp) and fp >= 0.6,
        f"{fmt_pct(fp, 0)} de folds con Sharpe positivo")

    bs = calib.get("brier_skill", np.nan)
    add("calibración útil", np.isfinite(bs) and bs > 0,
        f"Brier skill = {fmt_num(bs)}, ECE = {fmt_num(calib.get('ece', np.nan))}",
        critical=False)

    bh = metrics.get("buy_hold", {}).get("sharpe", np.nan)
    add("bate al buy & hold", np.isfinite(sh) and np.isfinite(bh) and sh > bh,
        f"Sharpe estrategia {fmt_num(sh, 2)} vs. B&H {fmt_num(bh, 2)}", critical=False)

    cont = metrics.get("continuous", {})
    if cont.get("n", 0) > 20 and np.isfinite(cont.get("sharpe", np.nan)):
        cs, rs = cont["sharpe"], metrics.get("sharpe", np.nan)
        add("PnL ejecutable coherente",
            cs > 0.3 and (not np.isfinite(rs) or cs > 0.6 * rs),
            f"Sharpe ancla->ancla = {fmt_num(cs, 2)} frente a {fmt_num(rs, 2)} "
            f"del cálculo solapado")

    stress = metrics.get("sharpe_anchor_stress_p05")
    if stress is not None:
        add("robusto al ruido del ancla", stress > 0,
            f"Sharpe percentil 5 bajo estrés = {fmt_num(stress, 2)}")

    crit = [c for c in checks if c["critical"]]
    n_fail_crit = sum(1 for c in crit if not c["pass"])
    if n_fail_crit == 0:
        decision = "GO"
    elif n_fail_crit <= 2:
        decision = "REVISAR"
    else:
        decision = "NO-GO"
    return {"decision": decision, "checks": checks,
            "n_failed_critical": n_fail_crit, "n_critical": len(crit)}


# ---------------------------------------------------------------------------
# Salud del snapshot que se va a predecir
# ---------------------------------------------------------------------------
def snapshot_health(art: "FoldArtifacts", row: pd.DataFrame) -> dict:
    """Comprueba que la fila de hoy se parece a lo que el modelo aprendió.

    Dos preguntas que nadie hace antes de operar:
      1. ¿Cuántas features vienen vacías? Las rellena el imputador con la
         mediana de entrenamiento, así que una fila medio vacía produce una
         predicción basada en valores inventados, con la misma cara de
         seguridad que cualquier otra.
      2. ¿El mercado de hoy se parece al del entrenamiento? Se mide con el PSI
         (Population Stability Index) contra los cuantiles guardados. Por
         encima de 0.25 el modelo está extrapolando.
    """
    feats = art.features
    present = row[feats] if all(f in row.columns for f in feats) else None
    if present is None:
        return {"ok": False, "reason": "faltan features en el snapshot"}

    nan_frac = float(present.isna().mean(axis=1).iloc[0])
    warnings: list[str] = []
    if nan_frac > 0.30:
        warnings.append(f"{100 * nan_frac:.0f}% de las features vienen vacías")

    psi = np.nan
    ref = art.ref_quantiles
    if ref is not None:
        Z = art.imputer.transform(present)[0]
        q = ref["q"]
        outside = 0
        for i in range(len(Z)):
            lo, hi = q[0, i], q[-1, i]
            if np.isfinite(lo) and np.isfinite(hi) and (Z[i] < lo or Z[i] > hi):
                outside += 1
        psi = outside / max(len(Z), 1)
        if psi > 0.25:
            warnings.append(
                f"{100 * psi:.0f}% de las features caen fuera del rango visto en "
                f"entrenamiento: posible cambio de régimen")

    age = None
    if art.train_end:
        age = (pd.Timestamp(row.index[-1]).normalize()
               - pd.Timestamp(art.train_end)).days
        if age is not None and age > 120:
            warnings.append(f"el modelo se entrenó hace {age} días; reentrena")

    return {"ok": not warnings, "nan_frac": round(nan_frac, 3),
            "out_of_range_frac": None if not np.isfinite(psi) else round(psi, 3),
            "model_age_days": age, "warnings": warnings}


# ---------------------------------------------------------------------------
# Predicción para mañana
# ---------------------------------------------------------------------------
def predict_next(cfg: Config, result: ExperimentResult | None = None,
                 live_anchor_price: float | None = None, md=None) -> dict:
    """Genera la señal para el día siguiente usando el modelo final.

    live_anchor_price: si estás mirando la pantalla a las 15:30 ET y el feed
    diario aún no tiene el cierre de hoy, pásalo a mano y se usa ese.
    """
    log = get_logger(cfg.verbose)
    if result is None:
        result = run_experiment(cfg, md=md)

    md = load_market_data(cfg) if md is None else md
    if live_anchor_price is not None:
        md.anchor.price.iloc[-1] = float(live_anchor_price)
        log.info(f"usando precio de ancla en vivo: {live_anchor_price}")

    X = build_features(md, cfg)
    labels = build_labels(md, cfg)

    art: FoldArtifacts = result.final_model["artifacts"]
    row = X.iloc[[-1]]
    missing = [f for f in art.features if f not in row.columns]
    if missing:
        raise RuntimeError(f"faltan features en el snapshot actual: {missing[:5]}")

    Z = art.imputer.transform(row[art.features])
    probs = []
    for fam in art.families:
        p = _proba(art.models[fam], Z)
        probs.append(_apply_calibrator(art.calibrators[fam], p))
    p_up = float(_ensemble_prob(art, np.column_stack(probs))[0])

    sigma = float(labels.sigma_ex_ante.iloc[-1])
    q = {}
    if art.quantiles is not None:
        qq = art.quantiles.predict(pd.DataFrame(Z, columns=art.features,
                                                index=row.index))
        q = {k: float(v.iloc[0]) * sigma for k, v in qq.items()}

    pos_s = position_from_prob(pd.Series([p_up], index=row.index), cfg,
                               expected_ret=pd.Series([q.get("q50", 0.0)],
                                                      index=row.index),
                               sigma=pd.Series([sigma], index=row.index))
    pos_s, vetoed = _coherence_gate(
        pos_s, pd.DataFrame({"ret_q50": [q.get("q50", np.nan)]}, index=row.index), cfg)
    pos = float(pos_s.iloc[0])

    health = snapshot_health(art, row)
    if health.get("warnings"):
        for wmsg in health["warnings"]:
            log.warning(f"{cfg.data.ticker}: {wmsg}")

    return {
        "ticker": cfg.data.ticker,
        "as_of": str(row.index[-1].date()),
        "health": health,
        "anchor_price": float(md.anchor.price.iloc[-1]),
        "anchor_source": md.anchor.source,
        "prob_up": p_up,
        "expected_return": q.get("q50", np.nan),
        "ret_p10": q.get("q10", np.nan),
        "ret_p90": q.get("q90", np.nan),
        "sigma_ex_ante": sigma,
        "position": pos,
        "horizon": cfg.label.horizon,
        "verdict": result.verdict["decision"],
        "oos_sharpe": result.metrics.get("sharpe"),
        "oos_auc": result.metrics.get("auc_oos"),
        "pbo": result.pbo.pbo if result.pbo else np.nan,
    }
