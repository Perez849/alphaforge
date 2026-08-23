"""Predicción de VOLATILIDAD.

Por qué esto y no dirección: la dirección diaria de una megacap es la variable
menos predecible que existe con datos públicos —medido en este mismo sistema,
ventaja sobre la tasa base de +0.03 pp con z≈0.1—. La volatilidad es lo
contrario: es persistente, agrupada y se predice con R² de 0.4-0.5. No es una
diferencia de grado, es de naturaleza.

El listón aquí NO es acertar. Es **batir a HAR-RV** (Corsi, 2009), tres
regresores —volatilidad de ayer, de la última semana y del último mes— que
llevan quince años siendo dificilísimos de superar. Cualquier modelo de
volatilidad que no se compare con HAR está contando la mitad de la historia:
un R² de 0.45 suena estupendo hasta que descubres que HAR da 0.47.

ESTIMADORES
-----------
Con OHLCV diario se puede estimar la volatilidad de UN día mucho mejor que con
el retorno cierre-a-cierre, que desperdicia el recorrido intradía:

  * Parkinson     - usa el rango high-low. ~5x más eficiente que close-to-close
  * Garman-Klass  - añade open y close. ~7x más eficiente
  * Rogers-Satchell - tolera deriva distinta de cero
  * Yang-Zhang    - combina salto nocturno + apertura-cierre + Rogers-Satchell.
                    Es el mejor con datos OHLC y el que se usa por defecto

MÉTRICAS
--------
  * R² fuera de muestra contra la media histórica (el mínimo exigible)
  * R² fuera de muestra contra HAR-RV (el listón de verdad)
  * QLIKE - función de pérdida robusta para volatilidad (Patton, 2011). El MSE
    castiga desproporcionadamente los errores en picos de volatilidad; QLIKE no
  * Diebold-Mariano - contrasta si la diferencia frente a HAR es significativa
    o es ruido
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Estimadores de volatilidad de una barra
# ---------------------------------------------------------------------------
def parkinson(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["High"] / df["Low"].where(df["Low"] > 0))
    return np.sqrt((hl ** 2) / (4 * np.log(2)))


def garman_klass(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["High"] / df["Low"].where(df["Low"] > 0)) ** 2
    co = np.log(df["Close"] / df["Open"].where(df["Open"] > 0)) ** 2
    v = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(v.clip(lower=0))


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    ho, lo = np.log(h / o.where(o > 0)), np.log(l / o.where(o > 0))
    hc, lc = np.log(h / c.where(c > 0)), np.log(l / c.where(c > 0))
    return np.sqrt((ho * hc + lo * lc).clip(lower=0))


def yang_zhang(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Yang-Zhang: el mejor estimador con OHLC.

    Con window=1 devuelve la estimación de cada barra suelta; con window>1
    aplica la ponderación original entre las tres componentes.
    """
    o, c = df["Open"], df["Close"]
    prev_c = c.shift(1)
    on = np.log(o / prev_c.where(prev_c > 0))          # salto nocturno
    oc = np.log(c / o.where(o > 0))                    # apertura a cierre
    rs = rogers_satchell(df) ** 2

    if window <= 1:
        return np.sqrt((on ** 2 + oc ** 2 + rs).clip(lower=0) / 3.0)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v_on = on.rolling(n, min_periods=max(2, n // 2)).var(ddof=1)
    v_oc = oc.rolling(n, min_periods=max(2, n // 2)).var(ddof=1)
    v_rs = rs.rolling(n, min_periods=max(2, n // 2)).mean()
    return np.sqrt((v_on + k * v_oc + (1 - k) * v_rs).clip(lower=0))


ESTIMATORS = {"parkinson": parkinson, "garman_klass": garman_klass,
              "rogers_satchell": rogers_satchell,
              "yang_zhang": lambda df: yang_zhang(df, 1)}


def realized_vol(df: pd.DataFrame, estimator: str = "yang_zhang",
                 annualize: bool = True, floor: float = 1e-4) -> pd.Series:
    """Volatilidad realizada de cada barra, con suelo para poder tomar logs."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimador desconocido: {estimator}. "
                         f"Opciones: {list(ESTIMATORS)}")
    v = ESTIMATORS[estimator](df)
    v = v.replace([np.inf, -np.inf], np.nan)
    if annualize:
        v = v * np.sqrt(TRADING_DAYS)
    return v.clip(lower=floor)


# ---------------------------------------------------------------------------
# HAR-RV: el modelo a batir
# ---------------------------------------------------------------------------
@dataclass
class HARModel:
    """Heterogeneous AutoRegressive (Corsi, 2009).

        log RV(t+h) = b0 + b1 log RV_d(t) + b2 log RV_w(t) + b3 log RV_m(t)

    La idea es que en el mercado conviven agentes con horizontes distintos
    —día, semana, mes— y cada uno deja su huella en la volatilidad. Tres
    regresores, ningún hiperparámetro, y una barbaridad de trabajos publicados
    que no consiguen mejorarlo de forma consistente.
    """
    coef: np.ndarray | None = None
    cols: tuple[str, ...] = ("d", "w", "m")

    @staticmethod
    def design(log_rv: pd.Series) -> pd.DataFrame:
        """Los tres regresores, todos con información hasta t inclusive."""
        return pd.DataFrame({
            "d": log_rv,
            "w": log_rv.rolling(5, min_periods=3).mean(),
            "m": log_rv.rolling(22, min_periods=10).mean(),
        }, index=log_rv.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in self.cols])
        ok = np.isfinite(A).all(axis=1) & np.isfinite(y.to_numpy())
        if ok.sum() < 30:
            raise ValueError(f"HAR: solo {ok.sum()} filas utilizables")
        self.coef = np.linalg.lstsq(A[ok], y.to_numpy()[ok], rcond=None)[0]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("HAR sin ajustar")
        A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in self.cols])
        A = np.nan_to_num(A, nan=np.nanmedian(A))
        return A @ self.coef


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def r2_oos(y: np.ndarray, pred: np.ndarray, bench: np.ndarray) -> float:
    """R² fuera de muestra frente a un benchmark (Campbell-Thompson).

    Positivo = el modelo mejora al benchmark. Negativo = lo empeora, que es el
    resultado habitual y el que casi nadie reporta.
    """
    y, pred, bench = map(lambda a: np.asarray(a, float), (y, pred, bench))
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(bench)
    if ok.sum() < 20:
        return np.nan
    sse = np.sum((y[ok] - pred[ok]) ** 2)
    sst = np.sum((y[ok] - bench[ok]) ** 2)
    return float(1 - sse / sst) if sst > _EPS else np.nan


def qlike(y_var: np.ndarray, pred_var: np.ndarray) -> float:
    """QLIKE (Patton, 2011), en varianza, no en logs.

    El MSE penaliza sobre todo los errores en los picos, que es donde el ruido
    manda; QLIKE es robusto a que la medida de volatilidad sea imperfecta, que
    siempre lo es. Menor es mejor.
    """
    y, p = np.asarray(y_var, float), np.asarray(pred_var, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > _EPS) & (p > _EPS)
    if ok.sum() < 20:
        return np.nan
    r = y[ok] / p[ok]
    return float(np.mean(r - np.log(r) - 1))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """¿La diferencia entre dos modelos es real o es ruido?

    e1, e2: series de pérdidas de cada modelo. Devuelve (estadístico, p-valor).
    Negativo = el primero es mejor. Usa varianza de largo plazo (Newey-West)
    porque los errores de predicción están autocorrelacionados.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    d = e1[ok] - e2[ok]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / n
    lag = max(1, int(np.floor(1.5 * n ** (1 / 3))))
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        g = np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
        var += 2 * (1 - k / (lag + 1)) * g
    var = max(var, _EPS)
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def mincer_zarnowitz(y: np.ndarray, pred: np.ndarray) -> dict:
    """Regresión y = a + b·pred. Un buen modelo tiene a≈0 y b≈1.

    b < 1 indica predicciones demasiado extremas; b > 1, demasiado tibias.
    Es la prueba de calibración de una predicción continua.
    """
    y, p = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 30:
        return {"alpha": np.nan, "beta": np.nan, "r2": np.nan}
    A = np.column_stack([np.ones(ok.sum()), p[ok]])
    coef = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    resid = y[ok] - A @ coef
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    return {"alpha": float(coef[0]), "beta": float(coef[1]),
            "r2": float(1 - np.sum(resid ** 2) / sst) if sst > _EPS else np.nan}


@dataclass
class VolTarget:
    """Objetivo de volatilidad, en logaritmos y sin fuga temporal."""
    log_rv_next: pd.Series          # objetivo: log RV del periodo siguiente
    log_rv_now: pd.Series           # RV conocida hasta hoy (regresor HAR)
    rv_next: pd.Series              # el mismo objetivo en nivel
    har_design: pd.DataFrame
    horizon: int
    estimator: str
    mask: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))


def build_vol_target(daily: pd.DataFrame, horizon: int = 1,
                     estimator: str = "yang_zhang") -> VolTarget:
    """Construye el objetivo de volatilidad.

    CONTRATO TEMPORAL: `log_rv_now` usa barras hasta t inclusive y `log_rv_next`
    es la volatilidad de t+1..t+h, futuro puro. Los regresores HAR se calculan
    solo con `log_rv_now`, así que el benchmark juega con las mismas cartas que
    el modelo: comparar contra un HAR que viera el futuro no valdría nada.
    """
    rv = realized_vol(daily, estimator)
    log_rv = np.log(rv)

    if horizon == 1:
        fut = rv.shift(-1)
    else:                            # media de varianza en la ventana futura
        fut = np.sqrt((rv ** 2).shift(-horizon).rolling(
            horizon, min_periods=max(1, horizon // 2)).mean().shift(horizon - 1))
        fut = fut.shift(-(horizon - 1))
    fut = fut.clip(lower=1e-4)

    design = HARModel.design(log_rv)
    mask = (np.isfinite(fut) & np.isfinite(log_rv)
            & design.notna().all(axis=1))
    return VolTarget(log_rv_next=np.log(fut), log_rv_now=log_rv, rv_next=fut,
                     har_design=design, horizon=horizon, estimator=estimator,
                     mask=mask)


# ---------------------------------------------------------------------------
# Experimento walk-forward de volatilidad
# ---------------------------------------------------------------------------
@dataclass
class VolResult:
    ticker: str
    oos: pd.DataFrame
    metrics: dict
    fold_metrics: list
    verdict: dict
    horizon: int
    estimator: str
    final_model: dict = field(default_factory=dict)


def run_vol_experiment(md, cfg, horizon: int = 1,
                       estimator: str = "yang_zhang") -> VolResult:
    """Predice volatilidad y la enfrenta a HAR-RV con el mismo walk-forward.

    Reutiliza el resto del sistema —features, splits purgados, guarda anti-fuga—
    y cambia lo que tiene que cambiar: objetivo continuo, modelos de regresión y
    métricas de volatilidad. El veredicto NO premia acertar: premia acertar
    MEJOR QUE HAR, que es el único listón que significa algo aquí.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.impute import SimpleImputer

    from . import validation as val
    from .features import build_features
    from .utils import fmt_num, get_logger, suppress_noisy_warnings

    suppress_noisy_warnings()
    log = get_logger(cfg.verbose)
    log.info(f"=== Volatilidad | {md.ticker} | horizonte {horizon}d | "
             f"estimador {estimator} ===")

    X = build_features(md, cfg)
    tgt = build_vol_target(md.daily, horizon, estimator)

    # los regresores HAR entran también como features: si no, el modelo partiría
    # en desventaja artificial frente al benchmark
    X = X.join(tgt.har_design.add_prefix("har_"), how="left")

    idx = X.index.intersection(tgt.log_rv_next.index)
    m = tgt.mask.reindex(idx).fillna(False) & X.loc[idx].notna().mean(axis=1).ge(0.7)
    idx = idx[m.to_numpy()]
    X, y = X.loc[idx], tgt.log_rv_next.loc[idx]
    har_X = tgt.har_design.loc[idx]
    log.info(f"muestras: {len(X)} x {X.shape[1]} features | "
             f"{idx.min().date()} -> {idx.max().date()}")

    splitter = val.PurgedWalkForward(
        n_splits=cfg.validation.n_splits, purge=max(horizon + 1, 3),
        embargo_pct=cfg.validation.embargo_pct,
        min_train=cfg.validation.min_train_days, anchored=True)
    splits = splitter.split(X.index)
    val.assert_no_overlap(splits, max(horizon + 1, 3))

    rows, folds = [], []
    for sp in splits:
        tr, te = sp.train, sp.test
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        Ztr, Zte = imp.fit_transform(X.iloc[tr]), imp.transform(X.iloc[te])
        ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()

        # ── Bloque para calibrar el peso, SIN entrenar en él ────────────────
        # El peso del residuo se estimaba sobre el tramo final del train, que
        # es tramo que los modelos ya habían visto: el residuo parecía perfecto
        # y el peso salía 1.00 siempre. Con horizonte 1 apenas molestaba; con
        # horizonte 5, donde los objetivos de días consecutivos comparten 4 de 5
        # días, la memorización es masiva y aplicaba residuo basura sobre un HAR
        # que funcionaba (medido: -0.40 de R2 frente a HAR).
        gap = max(horizon + 1, 3)
        n_val = max(150, int(0.2 * len(tr)))
        core = tr[: max(100, len(tr) - n_val - gap)]
        val = tr[len(tr) - n_val:]
        if len(core) < 200:                       # train corto: sin calibración
            core, val = tr, np.array([], dtype=int)
        Zc = imp.transform(X.iloc[core])
        yc = y.iloc[core].to_numpy()

        # benchmark: HAR con exactamente la misma información
        har = HARModel().fit(har_X.iloc[tr], y.iloc[tr])
        p_har = har.predict(har_X.iloc[te])

        # ── El ML predice el RESIDUO de HAR, no el nivel ────────────────────
        # Pedirle el nivel es pedirle que redescubra por su cuenta lo que HAR
        # resuelve con tres regresores, y con 158 features acaba diluyendo esa
        # señal: medido, peor que HAR en los seis folds. Modelando el residuo
        # parte DESDE HAR y solo puede añadir lo que HAR no captura.
        har_in = har.predict(har_X.iloc[tr])
        resid_tr = ytr - har_in
        # los modelos que estiman el peso se ajustan solo con `core`
        har_core = HARModel().fit(har_X.iloc[core], y.iloc[core])
        resid_core = yc - har_core.predict(har_X.iloc[core])

        preds, fitted = {}, []
        try:
            r = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Ztr, resid_tr)
            preds["ridge"] = r.predict(Zte)
            fitted.append(r)
        except Exception as e:                       # noqa: BLE001
            log.warning(f"fold {sp.fold}: ridge falló ({e})")
        try:
            g = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.04, max_iter=250,
                min_samples_leaf=60, l2_regularization=3.0,
                early_stopping=True, validation_fraction=0.15,
                random_state=cfg.model.random_state + sp.fold)
            g.fit(Ztr, resid_tr)
            preds["hgb"] = g.predict(Zte)
            fitted.append(g)
        except Exception as e:                       # noqa: BLE001
            log.warning(f"fold {sp.fold}: hgb falló ({e})")
        if not preds:
            continue
        r_ml = np.mean(np.column_stack(list(preds.values())), axis=1)

        # ── Cuánto fiarse del residuo: se decide con datos, no a ojo ────────
        # Se estima en el tramo final del train (no visto por los modelos) qué
        # fracción del residuo predicho conviene aplicar. Si el ML no aporta,
        # el coeficiente se va a cero y el resultado es HAR intacto: por
        # construcción, no se puede quedar peor.
        shrink = 0.0
        if len(val) >= 60:
            try:
                # Modelos gemelos entrenados SOLO con `core`: para ellos, `val`
                # es territorio no visto, que es la única forma de que el peso
                # signifique algo.
                probe = []
                pr = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Zc, resid_core)
                probe.append(pr)
                pg = HistGradientBoostingRegressor(
                    max_depth=3, learning_rate=0.04, max_iter=250,
                    min_samples_leaf=60, l2_regularization=3.0,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=cfg.model.random_state + sp.fold)
                pg.fit(Zc, resid_core)
                probe.append(pg)

                Zv = imp.transform(X.iloc[val])
                rv_hat = np.mean(np.column_stack([mm.predict(Zv) for mm in probe]),
                                 axis=1)
                rv_true = y.iloc[val].to_numpy() - har_core.predict(har_X.iloc[val])
                den = float(np.sum(rv_hat ** 2))
                if den > _EPS:
                    shrink = float(np.clip(np.sum(rv_hat * rv_true) / den, 0.0, 1.0))
            except Exception as e:                   # noqa: BLE001
                log.warning(f"fold {sp.fold}: peso del residuo no estimable ({e})")
        else:
            log.debug(f"fold {sp.fold}: bloque de calibración corto, peso 0")

        p_ml = p_har + shrink * r_ml
        # tope de desviación: ni con shrinkage alto se permite delirar
        p_ml = np.clip(p_ml, p_har - 0.5, p_har + 0.5)

        blk = pd.DataFrame({"y": yte, "pred": p_ml, "har": p_har,
                            "media": ytr.mean(), "shrink": shrink},
                           index=X.index[te])
        blk["fold"] = sp.fold
        rows.append(blk)
        folds.append({
            "fold": sp.fold, "test_start": str(sp.test_span[0].date()),
            "test_end": str(sp.test_span[1].date()),
            "r2_vs_media": r2_oos(yte, p_ml, np.full(len(yte), ytr.mean())),
            "r2_vs_har": r2_oos(yte, p_ml, p_har),
            "har_r2_vs_media": r2_oos(yte, p_har, np.full(len(yte), ytr.mean())),
            "shrink": round(shrink, 3),
        })
        log.info(f"  fold {sp.fold} [{folds[-1]['test_start']}..{folds[-1]['test_end']}] "
                 f"R2 vs media {fmt_num(folds[-1]['r2_vs_media'])} | "
                 f"HAR {fmt_num(folds[-1]['har_r2_vs_media'])} | "
                 f"ML vs HAR {fmt_num(folds[-1]['r2_vs_har'])} "
                 f"(peso del residuo {shrink:.2f})")

    oos = pd.concat(rows).sort_index()
    yv, pm, ph = (oos["y"].to_numpy(), oos["pred"].to_numpy(),
                  oos["har"].to_numpy())
    med = oos["media"].to_numpy()

    l_ml = (yv - pm) ** 2
    l_har = (yv - ph) ** 2
    dm_stat, dm_p = diebold_mariano(l_ml, l_har)
    q_ml = qlike(np.exp(yv) ** 2, np.exp(pm) ** 2)
    q_har = qlike(np.exp(yv) ** 2, np.exp(ph) ** 2)

    metrics = {
        "n": int(len(oos)),
        "r2_vs_media": r2_oos(yv, pm, med),
        "har_r2_vs_media": r2_oos(yv, ph, med),
        "r2_vs_har": r2_oos(yv, pm, ph),
        "qlike": q_ml, "qlike_har": q_har,
        "qlike_mejora": (q_har - q_ml) / q_har if q_har and np.isfinite(q_har) else np.nan,
        "dm_stat": dm_stat, "dm_pvalue": dm_p,
        "mz": mincer_zarnowitz(yv, pm),
        "mz_har": mincer_zarnowitz(yv, ph),
        "corr": float(np.corrcoef(yv, pm)[0, 1]) if len(yv) > 10 else np.nan,
        "folds_mejor_que_har": float(np.mean([f["r2_vs_har"] > 0 for f in folds])),
        "shrink_medio": float(np.mean([f["shrink"] for f in folds])),
    }

    checks = []

    def add(name, ok, detail, critical=True):
        checks.append({"check": name, "pass": bool(ok), "detail": detail,
                       "critical": critical})

    add("predice algo", metrics["r2_vs_media"] > 0.05,
        f"R2 frente a la media = {fmt_num(metrics['r2_vs_media'])}")
    add("bate a HAR-RV", metrics["r2_vs_har"] > 0,
        f"R2 frente a HAR = {fmt_num(metrics['r2_vs_har'])} "
        f"(HAR da {fmt_num(metrics['har_r2_vs_media'])} sobre la media)")
    add("la mejora no es ruido", np.isfinite(dm_p) and dm_p < 0.05 and dm_stat < 0,
        f"Diebold-Mariano t={fmt_num(dm_stat, 2)}, p={fmt_num(dm_p, 4)}")
    add("mejora también en QLIKE", np.isfinite(q_ml) and np.isfinite(q_har) and q_ml < q_har,
        f"QLIKE {fmt_num(q_ml, 4)} frente a {fmt_num(q_har, 4)} de HAR")
    b = metrics["mz"]["beta"]
    add("predicción bien escalada", np.isfinite(b) and 0.75 <= b <= 1.25,
        f"Mincer-Zarnowitz beta = {fmt_num(b, 3)} (ideal 1.0)", critical=False)
    add("consistente entre folds", metrics["folds_mejor_que_har"] >= 0.6,
        f"{100 * metrics['folds_mejor_que_har']:.0f}% de folds por encima de HAR")

    crit = [c for c in checks if c["critical"]]
    nf = sum(1 for c in crit if not c["pass"])
    verdict = {"decision": "GO" if nf == 0 else ("REVISAR" if nf <= 1 else "NO-GO"),
               "checks": checks, "n_failed_critical": nf, "n_critical": len(crit)}

    log.info(f"R2 vs media {fmt_num(metrics['r2_vs_media'])} | "
             f"HAR {fmt_num(metrics['har_r2_vs_media'])} | "
             f"ML vs HAR {fmt_num(metrics['r2_vs_har'])} | "
             f"DM p={fmt_num(dm_p, 4)} -> {verdict['decision']}")

    return VolResult(ticker=md.ticker, oos=oos, metrics=metrics,
                     fold_metrics=folds, verdict=verdict, horizon=horizon,
                     estimator=estimator)
