"""Autodiagnóstico del sistema. `python -m alphaforge selftest`.

No son tests de juguete: cada uno cubre una forma concreta y conocida de que
un backtest de trading mienta. Si alguno falla, NO uses el sistema para operar.
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import validation as val
from .backtest import compute_metrics, run_backtest
from .config import Config
from .data import _extract_anchor_from_intraday, validate_ohlcv
from .features import build_features
from .labeling import align, build_labels
from .models import QuantileBundle
from .synthetic import make_market_data, make_ohlcv
from .utils import DataIntegrityError, LeakageError, suppress_noisy_warnings


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    seconds: float


_REGISTRY: list = []


def check(name: str):
    def deco(fn):
        _REGISTRY.append((name, fn))
        return fn
    return deco


def _fast_cfg(ticker: str) -> Config:
    """Configuración ligera para los checks que ejecutan el pipeline entero."""
    cfg = Config()
    cfg.data.ticker = ticker
    cfg.verbose = 0
    cfg.model.families = ("logit", "hgb")
    cfg.model.n_trials = 8
    cfg.model.max_features_selected = 20
    cfg.validation.n_splits = 4
    cfg.validation.inner_splits = 3
    cfg.validation.min_train_days = 700
    cfg.validation.n_permutations = 0
    cfg.features.cross_asset = False
    cfg.features.monthly = False
    return cfg


# ---------------------------------------------------------------------------
# 1. Contrato temporal de las features
# ---------------------------------------------------------------------------
@check("features: el bloque BASE está desplazado un día")
def _t_shift():
    md = make_market_data(n=600, seed=1, with_context=False)
    cfg = Config()
    cfg.features.cross_asset = False
    X = build_features(md, cfg)
    c = md.daily["Close"]
    expected = (c / c.shift(1) - 1).shift(1)          # retorno de AYER
    got = X["b_roc_1"]
    ok = expected.align(got, join="inner")
    d = (ok[0] - ok[1]).abs().dropna()
    assert d.max() < 1e-10, f"b_roc_1 no coincide con el retorno de t-1 (err={d.max():.2e})"
    return f"máx. desviación {d.max():.2e} sobre {len(d)} filas"


@check("features: el bloque TODAY usa el ancla, no el cierre")
def _t_today():
    md = make_market_data(n=400, seed=2, with_context=False)
    cfg = Config()
    cfg.features.cross_asset = False
    X = build_features(md, cfg)
    a, c = md.anchor.price, md.daily["Close"]
    expected = a / c.shift(1) - 1
    got = X["tdy_anchor_ret"]
    j = pd.concat([expected, got], axis=1).dropna()
    d = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
    assert d.max() < 1e-10, "tdy_anchor_ret mal construido"
    wrong = (c / c.shift(1) - 1)
    corr_close = j.iloc[:, 1].corr(wrong.reindex(j.index))
    assert corr_close < 0.999, "la feature de hoy es indistinguible del cierre"
    return f"exacta (err={d.max():.2e}); corr con retorno-a-cierre = {corr_close:.3f}"


@check("features: ninguna columna correlaciona con el futuro")
def _t_leak_scan():
    md = make_market_data(n=1600, seed=3)
    cfg = Config()
    X = build_features(md, cfg)
    lab = build_labels(md, cfg)
    from .pipeline import _leakage_guard
    Xa, *_ = align(X, lab, cfg)
    _leakage_guard(Xa, lab.y_ret.reindex(Xa.index), cfg)
    yv = lab.y_ret.reindex(Xa.index).to_numpy(float)
    best = 0.0
    for col in Xa.columns:
        v = Xa[col].to_numpy(float)
        m = np.isfinite(v) & np.isfinite(yv)
        if m.sum() > 200 and np.std(v[m]) > 1e-12:
            best = max(best, abs(float(np.corrcoef(v[m], yv[m])[0, 1])))
    return f"máx |rho| con el retorno futuro = {best:.3f}"


@check("centinela: se detecta una fuga inyectada a propósito")
def _t_sentinel():
    md = make_market_data(n=1500, seed=4, with_context=False)
    cfg = Config()
    cfg.features.cross_asset = False
    X = build_features(md, cfg)
    lab = build_labels(md, cfg)
    Xa, *_ = align(X, lab, cfg)
    y = lab.y_ret.reindex(Xa.index)
    Xa = Xa.copy()
    Xa["TRAMPA_futuro"] = y * 0.9 + np.random.default_rng(0).normal(0, 1e-4, len(y))
    from .pipeline import _leakage_guard
    try:
        _leakage_guard(Xa, y, cfg)
    except LeakageError as e:
        assert "TRAMPA_futuro" in str(e)
        return "la guarda detecta y nombra la feature contaminada"
    raise AssertionError("¡la guarda anti-fuga NO detectó una fuga evidente!")


# ---------------------------------------------------------------------------
# 2. Etiquetado y ancla
# ---------------------------------------------------------------------------
@check("etiquetas: y(t) = Close(t+1)/Anchor(t) - 1")
def _t_labels():
    md = make_market_data(n=500, seed=5, with_context=False)
    cfg = Config()
    lab = build_labels(md, cfg)
    exp = md.daily["Close"].shift(-1) / md.anchor.price - 1
    d = (exp - lab.y_ret).abs().dropna()
    assert d.max() < 1e-12, f"etiqueta mal alineada ({d.max():.2e})"
    # y la última fila NO puede tener etiqueta (no existe el mañana)
    assert not np.isfinite(lab.y_ret.iloc[-1]), "la última fila tiene etiqueta: fuga"
    return f"exacto en {len(d)} filas; última fila sin etiqueta"


@check("ancla: se extrae la barra correcta de datos intradía")
def _t_anchor():
    # sesiones completas en horario de Nueva York (no en UTC: es el error clásico)
    ts = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"2026-03-{d:02d} 09:30", f"2026-03-{d:02d} 16:00",
                      freq="30min", tz="America/New_York").to_numpy()
        for d in (2, 3, 4)]), tz="UTC").tz_convert("America/New_York")
    df = pd.DataFrame({"Open": np.arange(len(ts), dtype=float),
                       "High": np.arange(len(ts)) + 1.0,
                       "Low": np.arange(len(ts)) - 1.0,
                       "Close": np.arange(len(ts)) + 0.5,
                       "Volume": 100.0}, index=ts)
    price, vol, _ = _extract_anchor_from_intraday(df, offset_min=30, interval="30m")
    # Con offset 30' el corte está en 15:30 ET; la barra elegida empieza ahí,
    # luego frac = 0 y el precio es su Open.
    et = df.copy()
    et.index = et.index.tz_convert("America/New_York")
    for d, p in price.items():
        bar = et[(et.index.date == d.date()) &
                 (et.index.hour == 15) & (et.index.minute == 30)]
        assert len(bar) == 1, f"no hay barra 15:30 para {d.date()}"
        assert abs(p - bar["Open"].iloc[0]) < 1e-9, f"ancla incorrecta en {d.date()}"
    assert (vol > 0).all()
    return f"{len(price)} días verificados contra la barra de las 15:30 ET"


@check("ancla: el volumen acumulado excluye los últimos 30 minutos")
def _t_anchor_vol():
    ts = pd.date_range("2026-03-02 14:30", "2026-03-02 21:00", freq="30min", tz="UTC")
    df = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5,
                       "Volume": 1.0}, index=ts)
    _, vol, _ = _extract_anchor_from_intraday(df, offset_min=30, interval="30m")
    total = df.between_time("14:30", "21:00")["Volume"].sum()
    assert vol.iloc[0] < total, "el volumen del ancla incluye la barra final"
    return f"volumen ancla {vol.iloc[0]:.0f} < total sesión {total:.0f}"


@check("ancla: media sesión (cierre a las 13:00) no contamina el precio")
def _t_half_day():
    def sesion(day, close_h):
        ts = pd.date_range(f"{day} 09:30", f"{day} {close_h}", freq="30min",
                           tz="America/New_York")
        n = len(ts)
        return pd.DataFrame({"Open": np.arange(n, dtype=float),
                             "High": np.arange(n) + 1.0, "Low": np.arange(n) - 1.0,
                             "Close": np.arange(n) + 0.5, "Volume": 10.0}, index=ts)

    df = pd.concat([sesion("2026-11-25", "16:00"),      # normal
                    sesion("2026-11-27", "13:00")])     # Black Friday
    price, vol, _ = _extract_anchor_from_intraday(df, 30, "30m")
    assert len(price) == 2, f"se esperaban 2 días, hay {len(price)}"
    for d, pv in price.items():
        day = df[df.index.date == d.date()]
        cierre = day["Close"].iloc[-1]
        assert abs(pv - cierre) > 1e-9, (
            f"{d.date()}: el ancla coincide con el cierre de la sesión: "
            "30 minutos de futuro colados en la feature principal")
        assert vol[d] < day["Volume"].sum() - 1e-9, \
            f"{d.date()}: el volumen del ancla incluye la barra de cierre"
    return "ancla y volumen medidos contra el cierre REAL de cada sesión"


@check("etiquetas: los huecos de cotización se descartan")
def _t_gap_labels():
    md = make_market_data(n=600, seed=21, with_context=False)
    keep = list(range(300)) + list(range(315, 600))       # 15 días suspendido
    md.daily = md.daily.iloc[keep]
    md.anchor.price = md.anchor.price.iloc[keep]
    cfg = Config()
    lab = build_labels(md, cfg)
    gaps = md.daily.index.to_series().diff().shift(-1).dt.days
    bad = int(((gaps > cfg.label.max_calendar_gap_days) & lab.mask_valid).sum())
    assert bad == 0, f"{bad} etiquetas válidas abarcan huecos de cotización"
    idx = md.daily.index[299]
    assert not bool(lab.mask_valid.loc[idx]), "la etiqueta del hueco sigue activa"
    return f"etiqueta a través de un hueco de {int(gaps.loc[idx])} días invalidada"


@check("caché: rangos de fechas distintos no comparten fichero")
def _t_cache_key():
    from alphaforge.data import _cache_path
    dc = Config().data
    a = _cache_path(dc, "AAPL", "1d", "2006-01-01", None)
    b = _cache_path(dc, "AAPL", "1d", "2020-01-01", None)
    c = _cache_path(dc, "AAPL", "1d", "2006-01-01", None)
    assert a != b, "misma clave de caché para rangos distintos"
    assert a == c, "la clave no es estable para el mismo rango"
    return "la clave incluye el rango solicitado"


@check("preprocesado: los umbrales de recorte no ven el futuro")
def _t_causal_clip():
    from alphaforge.pipeline import _prepare_fold
    md = make_market_data(n=1400, seed=22, with_context=False)
    cfg = Config()
    cfg.features.cross_asset = False
    cfg.validation.leak_alarm_corr = 0.95
    X = build_features(md, cfg)
    lab = build_labels(md, cfg)
    Xa, y, *_ = align(X, lab, cfg)
    tr = np.arange(800)
    yv = y.to_numpy(float)
    _, prep = _prepare_fold(Xa, tr, cfg, yv[tr], 0)
    Xs = Xa.copy()
    Xs.iloc[1000:] = Xs.iloc[1000:] * 500          # shock brutal en el futuro
    _, prep2 = _prepare_fold(Xs, tr, cfg, yv[tr], 0)
    assert np.allclose(prep.clipper.hi_, prep2.clipper.hi_), \
        "un shock posterior al train mueve los umbrales: look-ahead"
    Z = prep.transform(Xa.iloc[900:])
    assert np.all(np.isfinite(Z)), "el preprocesado deja valores no finitos"
    return "recorte e imputación ajustados solo con datos de entrenamiento"


@check("cotización: un precio que no es de hoy se rechaza")
def _t_stale_quote():
    from alphaforge.live import LiveQuote, StaleQuoteError
    viejo = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=3)
    q = LiveQuote("X", 100.0, viejo, "daily_close", 60.0, True, 99.0, 1e6)
    try:
        q.check_fresh()
    except StaleQuoteError:
        ahora = LiveQuote("X", 100.0, pd.Timestamp.now(tz="America/New_York"),
                          "1m", 60.0, False, 99.0, 1e6)
        ahora.check_fresh()
        return "rechaza precios de días anteriores y acepta los de hoy"
    raise AssertionError("se aceptó una cotización de hace 3 días como actual")


@check("solapamiento: se mide y el Sharpe se corrige")
def _t_overlap():
    rng = np.random.default_rng(31)
    idx = pd.bdate_range("2020-01-01", periods=900)
    base = rng.normal(0.0004, 0.01, 900)
    r = pd.Series(base + 0.6 * np.r_[0, base[:-1]], index=idx)   # autocorrelada
    pos = pd.Series(1.0, index=idx)
    m = run_backtest(pos, r, Config()).metrics
    assert np.isfinite(m["autocorr_1"]) and m["autocorr_1"] > 0.2, \
        "no se detecta la autocorrelación inducida"
    assert m["sharpe_overlap_adj"] < m["sharpe"], \
        "el Sharpe corregido no penaliza los retornos solapados"
    # la primera barra no tiene día previo, de ahí que no llegue a 1.0 exacto
    assert m["overlap_frac"] > 0.99, "no se cuentan los días encadenados"
    return (f"AC1={m['autocorr_1']:.2f}, Sharpe {m['sharpe']:.2f} -> "
            f"{m['sharpe_overlap_adj']:.2f} (factor {m['newey_west_factor']:.2f})")


@check("splits: el embargo separa de verdad train y test")
def _t_embargo():
    idx = pd.bdate_range("2010-01-01", periods=3000)
    emb = 0.02
    wf = val.PurgedWalkForward(n_splits=6, purge=3, embargo_pct=emb, min_train=1000)
    sp = wf.split(idx)
    val.assert_no_overlap(sp, 3)
    need = 3 + int(np.ceil(emb * len(idx))) - 1
    for s in sp:
        sep = int(s.test.min() - s.train.max() - 1)
        assert sep >= need, f"fold {s.fold}: separación {sep} < {need}"
    return f"{len(sp)} folds con al menos {need} barras entre train y test"


@check("PRODUCCIÓN: predecir hoy no depende de datos de mañana")
def _t_production_path():
    """El test más importante del conjunto.

    Se predice un día concreto dos veces: con el histórico completo (que
    contiene su futuro) y con el histórico cortado en ese mismo día. Si las dos
    probabilidades no son idénticas, el camino que usa el robot en vivo está
    mirando el futuro y todo lo demás sobra.
    """
    from alphaforge.pipeline import _apply_calibrator, _proba, run_experiment
    cfg = _fast_cfg("SYN_PATH")
    cfg.model.families = ("hgb",)
    cfg.validation.leak_alarm_corr = 0.95
    md = make_market_data(n=2000, seed=9, signal="mean_revert",
                          signal_strength=0.4, with_context=False)
    res = run_experiment(cfg, md=md)
    art = res.final_model["artifacts"]

    X_full = build_features(md, cfg)
    target = X_full.index[-40]                      # deja 39 días de futuro

    md_t = make_market_data(n=2000, seed=9, signal="mean_revert",
                            signal_strength=0.4, with_context=False)
    cut = md_t.daily.index.get_loc(target) + 1
    md_t.daily = md_t.daily.iloc[:cut]
    md_t.anchor.price = md_t.anchor.price.iloc[:cut]
    X_cut = build_features(md_t, cfg)

    a, b = X_full.loc[[target]], X_cut.loc[[target]]
    cols = [f for f in art.features if f in a.columns and f in b.columns]
    assert len(cols) >= 10, "no hay features comparables"
    diff = (a[cols].iloc[0] - b[cols].iloc[0]).abs()
    worst = diff.max()
    assert worst < 1e-9, (
        f"{int((diff > 1e-9).sum())} features cambian al añadir futuro "
        f"(peor: {diff.idxmax()}, {worst:.2e})")

    def prob(row):
        Z = art.imputer.transform(row[art.features])
        P = [_apply_calibrator(art.calibrators[f], _proba(art.models[f], Z))
             for f in art.families]
        return float(np.clip(np.column_stack(P) @ art.weights, 1e-6, 1 - 1e-6)[0])

    pa, pb = prob(a), prob(b)
    assert abs(pa - pb) < 1e-9, f"P(sube) difiere: {pa:.8f} vs {pb:.8f}"
    return f"P(sube)={pa:.6f} idéntica con y sin 39 días de futuro"


@check("PnL: el cálculo ancla->ancla no depende del solapamiento")
def _t_continuous():
    from alphaforge.backtest import continuous_pnl
    md = make_market_data(n=800, seed=41, with_context=False)
    pos = pd.Series(1.0, index=md.daily.index)
    r = continuous_pnl(pos, md.anchor.price, Config())
    a = md.anchor.price
    manual = float((a.iloc[-1] / a.iloc[0]) - 1)
    assert r["n"] > 700, f"solo {r['n']} días evaluados"
    # sin costes debe coincidir con comprar en la primera ancla y vender en la última
    cfg0 = Config()
    cfg0.backtest.commission_bps = cfg0.backtest.spread_bps = 0
    cfg0.backtest.slippage_bps = 0
    r0 = continuous_pnl(pos, md.anchor.price, cfg0)
    err = abs(r0["total_return"] - manual) / max(abs(manual), 1e-9)
    assert err < 0.02, f"PnL continuo {r0['total_return']:.4f} vs real {manual:.4f}"
    return f"retorno ancla->ancla {100 * r0['total_return']:+.1f}% coincide con el real"


@check("cotización: precios imposibles se rechazan")
def _t_bad_price():
    from alphaforge.live import LiveQuote, StaleQuoteError, sanity_check_quote
    now = pd.Timestamp.now(tz="America/New_York")
    ok = LiveQuote("X", 100.0, now, "1m", 60.0, False, 99.0, 1e6)
    sanity_check_quote(ok, 99.0)
    for px, label in [(0.0, "cero"), (-5.0, "negativo"), (250.0, "split x2.5"),
                      (1.0, "tick erróneo")]:
        try:
            sanity_check_quote(LiveQuote("X", px, now, "1m", 60.0, False, 99.0, 1e6),
                               100.0)
        except StaleQuoteError:
            continue
        raise AssertionError(f"se aceptó un precio {label} ({px})")
    return "rechaza cero, negativos, splits sin ajustar y ticks erróneos"


@check("modelo de producción: se entrena con el histórico completo")
def _t_refit_full():
    from alphaforge.pipeline import run_experiment
    cfg = _fast_cfg("SYN_REFIT")
    cfg.model.families = ("hgb",)
    cfg.validation.leak_alarm_corr = 0.95
    md = make_market_data(n=1900, seed=44, signal="mean_revert",
                          signal_strength=0.4, with_context=False)
    res = run_experiment(cfg, md=md)
    art = res.final_model["artifacts"]
    assert art.refit_full, "el modelo final no se reajustó con todo"
    last = str(res.oos.index[-1].date())
    assert art.train_end >= last, (
        f"el modelo de producción solo llega a {art.train_end} y hay datos "
        f"hasta {last}: se descartan los días más recientes")
    assert art.ref_quantiles is not None, "sin huella de distribución para la deriva"
    return f"entrenado hasta {art.train_end}, con huella para detectar deriva"


@check("snapshot: se avisa cuando la fila de hoy no es fiable")
def _t_snapshot_health():
    from alphaforge.pipeline import run_experiment, snapshot_health
    cfg = _fast_cfg("SYN_HEALTH")
    cfg.model.families = ("hgb",)
    cfg.validation.leak_alarm_corr = 0.95
    md = make_market_data(n=1900, seed=45, signal="mean_revert",
                          signal_strength=0.4, with_context=False)
    res = run_experiment(cfg, md=md)
    art = res.final_model["artifacts"]
    X = build_features(md, cfg)

    good = snapshot_health(art, X.iloc[[-1]])
    assert good["ok"], f"una fila normal se marca como problemática: {good}"

    empty = X.iloc[[-1]].copy()
    empty[art.features[: int(0.8 * len(art.features))]] = np.nan
    bad = snapshot_health(art, empty)
    assert not bad["ok"] and bad["warnings"], "una fila casi vacía pasa sin aviso"

    shifted = X.iloc[[-1]].copy()
    shifted[art.features] = shifted[art.features] * 1e4          # régimen imposible
    drift = snapshot_health(art, shifted)
    assert not drift["ok"], "un régimen fuera de rango pasa sin aviso"
    return "detecta filas vacías y valores fuera del rango de entrenamiento"


@check("ensemble: la mezcla se hace en el mismo espacio en que se pesó")
def _t_blend_space():
    from alphaforge.models import blend, learn_ensemble_weights
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.random(n) > 0.5).astype(float)
    P = np.column_stack([
        np.clip(0.5 + 0.22 * (y - 0.5) * 2 + rng.normal(0, 0.13, n), 0.02, 0.98)
        for _ in range(3)])
    w = learn_ensemble_weights(P, y)
    p_logit = blend(P, w)
    p_arit = P @ w
    ll_logit, ll_arit = log_loss(y, p_logit), log_loss(y, p_arit)
    assert ll_logit <= ll_arit + 1e-9, \
        f"la mezcla en logit ({ll_logit:.5f}) no mejora la aritmética ({ll_arit:.5f})"
    assert p_logit.std() >= p_arit.std() - 1e-9, \
        "la mezcla en logit aplana más que la aritmética"
    return (f"log-loss {ll_arit:.4f} -> {ll_logit:.4f}; dispersión "
            f"{p_arit.std():.4f} -> {p_logit.std():.4f}")


@check("calibrador: se elige por validación, no a mano")
def _t_calibrator_choice():
    from alphaforge.pipeline import _select_calibrator, _apply_calibrator
    rng = np.random.default_rng(4)
    # caso 1: probabilidades ya calibradas -> no debería tocarlas
    n = 800
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p).astype(float)
    c = _select_calibrator(p, y)
    q = _apply_calibrator(c, p)
    err_before = float(np.mean(np.abs(p - y)))
    err_after = float(np.mean(np.abs(q - y)))
    assert err_after <= err_before + 0.02, "empeora una probabilidad ya calibrada"

    # caso 2: probabilidades sesgadas -> debe corregirlas
    p_bad = np.clip(p * 0.5 + 0.25, 1e-6, 1 - 1e-6)
    c2 = _select_calibrator(p_bad, y)
    q2 = _apply_calibrator(c2, p_bad)
    assert c2 is not None, "no corrige una probabilidad claramente sesgada"
    assert np.std(q2) > np.std(p_bad) * 1.1, "no recupera la dispersión perdida"

    # con muy pocos datos no debe inventarse un calibrador frágil
    assert _select_calibrator(p[:30], y[:30]) is None, \
        "ajusta un calibrador con 30 muestras"
    return "respeta lo calibrado, corrige lo sesgado y se abstiene con poca muestra"


@check("pesos de muestra: el decaimiento es relativo a cada fold")
def _t_fold_weights():
    from alphaforge.labeling import sample_weights
    cfg = Config()
    idx = pd.bdate_range("2006-01-01", periods=5000)
    w_global = sample_weights(idx, 1, cfg)
    w_fold = sample_weights(idx[:1000], 1, cfg)
    # con el cálculo global, el primer fold entrena con pesos ínfimos
    assert w_global.iloc[:1000].mean() < 0.1, "el caso problemático ya no se da"
    assert 0.5 < w_fold.mean() < 2.0, \
        f"los pesos del fold no están normalizados ({w_fold.mean():.3f})"
    assert w_fold.iloc[-1] > w_fold.iloc[0], "el decaimiento no favorece lo reciente"
    return (f"global: media {w_global.iloc[:1000].mean():.4f} en el primer fold; "
            f"por fold: {w_fold.mean():.3f}")


@check("selección de features: hay purga entre core y holdout")
def _t_selection_purge():
    import inspect
    from alphaforge.pipeline import _prepare_fold
    src = inspect.getsource(_prepare_fold)
    assert "purge_days" in src, "la selección de features no aplica purga"
    cfg = Config()
    n_tr, n_hold = 1000, 200
    gap = cfg.validation.purge_days
    core_end = max(50, n_tr - n_hold - gap)
    hold_start = n_tr - n_hold
    assert hold_start - core_end >= gap, "la separación no llega a la purga"
    return f"{hold_start - core_end} barras entre core y holdout (purga {gap})"


@check("reloj: se detecta el ancla prematura, no solo la tardía")
def _t_anchor_timing():
    from alphaforge.live import market_clock
    offset = 60
    casos = {}
    for utc_h, etiqueta in [(18, "pronto"), (19, "en hora"), (20, "tarde")]:
        c = market_clock(pd.Timestamp(f"2026-08-21 {utc_h}:00", tz="UTC"))
        m = c["minutes_to_close"] - offset
        casos[etiqueta] = {"mins": c["minutes_to_close"],
                           "early": m > 10, "late": m < -10}
    assert casos["pronto"]["early"] and not casos["pronto"]["late"], \
        "una ejecución 60 min antes de tiempo no se marca como prematura"
    assert not casos["en hora"]["early"] and not casos["en hora"]["late"], \
        "la ejecución correcta se marca como problemática"
    assert casos["tarde"]["late"], "una ejecución al cierre no se marca como tardía"
    # el desajuste importa: el modelo aprendió con T-60
    assert abs(casos["pronto"]["mins"] - offset) > 30, "el desfase no se cuantifica"
    return (f"T−{casos['pronto']['mins']:.0f}′ = prematura, "
            f"T−{casos['en hora']['mins']:.0f}′ = correcta, "
            f"T−{casos['tarde']['mins']:.0f}′ = tardía")


@check("ancla: se reescala a la escala de la serie diaria")
def _t_anchor_rescale():
    """yfinance sirve el diario ajustado por dividendos y el intradía con otro
    criterio. Mezclarlos mete un sesgo del tamaño del dividendo acumulado
    (medido en SPY real: 161 bps) y fabrica correlación falsa con el futuro."""
    from alphaforge.data import build_anchor
    from alphaforge.config import Config as C
    import alphaforge.data as D

    n = 400
    idx = pd.bdate_range("2026-01-02", periods=n)
    px = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0, 0.01, n))
    daily = pd.DataFrame({"Open": px * 0.999, "High": px * 1.01, "Low": px * 0.99,
                          "Close": px, "Volume": 1e6}, index=idx)

    DIV = 0.984                       # el intradía viene 1.6% por encima
    anchor_raw = pd.Series(px * 0.998 / DIV, index=idx)
    sess_close = pd.Series(px / DIV, index=idx)
    vol = pd.Series(1e5, index=idx)

    orig = D.download
    D.download = lambda *a, **k: pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=pd.DatetimeIndex(["2026-01-02 15:00"], tz="America/New_York"))
    orig_ex = D._extract_anchor_from_intraday
    D._extract_anchor_from_intraday = lambda *a, **k: (anchor_raw, vol, sess_close)
    try:
        cfg = C()
        cfg.data.anchor_intervals = ("30m",)
        res = build_anchor(daily, "TEST", cfg, verbose=0)
    finally:
        D.download, D._extract_anchor_from_intraday = orig, orig_ex

    sesgo = float((daily["Close"] / res.price - 1).mean())
    sin_reescalar = float((daily["Close"] / anchor_raw - 1).mean())
    assert abs(sin_reescalar) > 0.010, "el caso de prueba no reproduce el sesgo"
    assert abs(sesgo) < 0.004, (
        f"tras reescalar queda un sesgo de {1e4 * sesgo:.0f} bps "
        f"(sin reescalar: {1e4 * sin_reescalar:.0f} bps)")
    return (f"sesgo {1e4 * sin_reescalar:+.0f} bps -> {1e4 * sesgo:+.0f} bps "
            f"tras llevar el ancla a la escala diaria")


@check("sizing: el modo continuo cosecha el centro y topa los extremos")
def _t_continuous_sizing():
    """Con AUC ~0.52 la información vive en el centro de la distribución, no en
    las colas. Medido sobre datos reales: el 87-89% de las predicciones caen
    entre 0.40 y 0.60. Un umbral de 0.55 descartaba esa masa entera."""
    from alphaforge.backtest import position_from_prob
    cfg = Config()
    p = pd.Series([0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.90])
    pos = position_from_prob(p, cfg)
    centro = pos[[1, 2, 4, 5]]
    assert (centro.abs() > 0).all(), "el modo continuo sigue descartando el centro"
    assert pos.abs().max() <= cfg.backtest.max_position + 1e-9, \
        f"posición {pos.abs().max():.3f} por encima del tope"
    assert abs(pos.iloc[3]) < 1e-9, "p=0.50 debería dar posición nula"
    assert pos.iloc[6] == pytest_approx(pos.iloc[7]), \
        "no satura: una probabilidad extrema produce una apuesta desmedida"
    # monótono y con el signo correcto
    assert (pos.diff().dropna() >= -1e-9).all(), "la posición no es monótona en p"
    assert pos.iloc[0] < 0 < pos.iloc[5], "signo invertido"

    viejo = Config()
    viejo.backtest.sizing = "confidence"
    pv = position_from_prob(p, viejo)
    n_new = int((pos.abs() > 0).sum())
    n_old = int((pv.abs() > 0).sum())
    assert n_new > n_old, "el modo continuo no aumenta las ocasiones operadas"
    return (f"opera en {n_new}/8 niveles de probabilidad frente a {n_old}/8 del "
            f"modo anterior; tope {cfg.backtest.max_position}")


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()


@check("cartera: la diversificación se calcula bien")
def _t_portfolio_math():
    """Sharpe de N señales independientes = Sharpe medio x sqrt(N)."""
    rng = np.random.default_rng(11)
    k, n = 8, 3000
    R = rng.normal(0.0004, 0.01, size=(n, k))          # independientes
    ind = [float(R[:, j].mean() / R[:, j].std(ddof=1) * np.sqrt(252)) for j in range(k)]
    port = R.mean(axis=1)
    sp = float(port.mean() / port.std(ddof=1) * np.sqrt(252))
    teo = float(np.mean(ind) * np.sqrt(k))
    assert abs(sp - teo) / max(abs(teo), 1e-9) < 0.15, \
        f"cartera {sp:.2f} frente a teórico {teo:.2f}"
    # con señales idénticas NO debe haber ganancia por diversificar
    Rc = np.repeat(R[:, [0]], k, axis=1)
    pc = Rc.mean(axis=1)
    spc = float(pc.mean() / pc.std(ddof=1) * np.sqrt(252))
    assert abs(spc - ind[0]) < 0.05, "inventa diversificación donde no la hay"
    return (f"8 independientes: {np.mean(ind):.2f} -> {sp:.2f} (teórico {teo:.2f}); "
            f"8 idénticas: sin ganancia")


# ---------------------------------------------------------------------------
# 3. Validación temporal
# ---------------------------------------------------------------------------
@check("splits: purga y embargo se respetan")
def _t_splits():
    idx = pd.bdate_range("2010-01-01", periods=2500)
    wf = val.PurgedWalkForward(n_splits=8, purge=5, embargo_pct=0.02, min_train=800)
    sp = wf.split(idx)
    val.assert_no_overlap(sp, 5)
    for s in sp:
        assert s.train.max() < s.test.min() - 4, "purga insuficiente"
        assert len(np.intersect1d(s.train, s.test)) == 0
    assert len(sp) >= 6, f"solo {len(sp)} folds"
    # el test siempre avanza en el tiempo
    starts = [s.test.min() for s in sp]
    assert all(b > a for a, b in zip(starts, starts[1:])), "folds no monótonos"
    return f"{len(sp)} folds, purga verificada, orden temporal correcto"


@check("splits: configuración incoherente aborta")
def _t_cfg():
    cfg = Config()
    cfg.validation.purge_days = 1          # < horizonte + 1
    try:
        cfg.validate()
    except ValueError as e:
        assert "purge_days" in str(e)
        return "purge_days insuficiente se rechaza en config.validate()"
    raise AssertionError("config.validate() aceptó una purga insuficiente")


# ---------------------------------------------------------------------------
# 4. Diagnóstico de overfitting
# ---------------------------------------------------------------------------
@check("PBO: configuraciones puramente aleatorias dan PBO ~ 0.5")
def _t_pbo_null():
    # El PBO de una sola matriz tiene un error de muestreo grande (sigma ~ 0.17):
    # todos los combos CSCV comparten el mismo dataset. Se promedian realizaciones.
    v = [val.compute_pbo(np.random.default_rng(s).normal(0, 0.01, (600, 40)), 10).pbo
         for s in range(15)]
    mean = float(np.mean(v))
    assert 0.40 < mean < 0.60, f"PBO medio={mean:.3f} en ruido puro (esperado ~0.50)"
    return (f"PBO medio={mean:.3f} sobre 15 realizaciones "
            f"(sd={np.std(v):.2f} — un PBO aislado es muy ruidoso)")


@check("PBO: una configuración genuinamente buena da PBO bajo")
def _t_pbo_signal():
    rng = np.random.default_rng(1)
    M = rng.normal(0, 0.01, size=(600, 20))
    M[:, 0] += 0.0035                       # esta sí tiene edge real y persistente
    r = val.compute_pbo(M, n_blocks=10)
    assert r.pbo < 0.25, f"PBO={r.pbo:.3f} pese a existir una config buena"
    return f"PBO={r.pbo:.3f} (detecta correctamente el edge real)"


@check("Deflated Sharpe: penaliza el número de pruebas")
def _t_dsr():
    d1, _ = val.deflated_sharpe(0.08, 1000, 0.0, 3.0, n_trials=5, var_trials_sr=0.001)
    d2, _ = val.deflated_sharpe(0.08, 1000, 0.0, 3.0, n_trials=5000, var_trials_sr=0.001)
    assert d2 < d1, "el DSR no penaliza más pruebas"
    p = val.probabilistic_sharpe(0.08, 1000, 0.0, 3.0)
    assert p > d2, "el PSR debería ser más laxo que el DSR"
    return f"DSR 5 pruebas={d1:.3f} > DSR 5000 pruebas={d2:.3f}; PSR={p:.3f}"


@check("permutación: el p-valor de una métrica nula ronda 0.5")
def _t_perm():
    rng = np.random.default_rng(2)
    null = rng.normal(0, 1, 500)
    p_mid = val.permutation_pvalue(0.0, null)
    p_hi = val.permutation_pvalue(4.0, null)
    assert 0.35 < p_mid < 0.65, f"p={p_mid:.3f} para el centro de la distribución nula"
    assert p_hi < 0.02, f"p={p_hi:.3f} para un valor extremo"
    return f"p(centro)={p_mid:.3f}, p(cola)={p_hi:.4f}"


@check("permutación por bloques: conserva la autocorrelación")
def _t_block_perm():
    rng = np.random.default_rng(3)
    y = pd.Series(rng.normal(size=2000)).rolling(20).mean().bfill().to_numpy()
    ac_orig = pd.Series(y).autocorr(1)
    ac_block = pd.Series(val.block_permute(y, 50, rng)).autocorr(1)
    ac_shuf = pd.Series(rng.permutation(y)).autocorr(1)
    assert abs(ac_block - ac_orig) < abs(ac_shuf - ac_orig), \
        "la permutación por bloques no preserva la estructura serial"
    return f"AC1 original={ac_orig:.3f}, bloques={ac_block:.3f}, shuffle={ac_shuf:.3f}"


# ---------------------------------------------------------------------------
# 5. Backtest y métricas
# ---------------------------------------------------------------------------
@check("métricas: Sharpe reproduce un valor analítico conocido")
def _t_sharpe():
    n = 2520
    r = pd.Series(np.full(n, 0.0004))       # sin varianza -> Sharpe infinito
    r.iloc[::2] += 0.005
    r.iloc[1::2] -= 0.005
    pos = pd.Series(1.0, index=r.index)
    m = compute_metrics(r, pos, r, Config())
    manual = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(m["sharpe"] - manual) < 1e-9, "Sharpe mal calculado"
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    assert abs(m["max_dd"] - dd) < 1e-9, "max drawdown mal calculado"
    return f"Sharpe={m['sharpe']:.4f} coincide con el cálculo directo"


@check("backtest: los costes reducen el retorno de forma monótona")
def _t_costs():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2020-01-01", periods=500)
    r = pd.Series(rng.normal(0.0006, 0.01, 500), index=idx)
    pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], 500), index=idx)
    prev = None
    for bps in (0.0, 2.0, 10.0):
        c = Config()
        c.backtest.commission_bps = bps
        c.backtest.spread_bps = 0
        c.backtest.slippage_bps = 0
        tr = run_backtest(pos, r, c).metrics["total_return"]
        if prev is not None:
            assert tr < prev, f"coste {bps}bps no redujo el retorno"
        prev = tr
    return "retorno estrictamente decreciente con el coste"


@check("backtest: sin posición no hay PnL ni comisiones")
def _t_flat():
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = pd.Series(np.random.default_rng(5).normal(0, 0.02, 200), index=idx)
    pos = pd.Series(0.0, index=idx)
    res = run_backtest(pos, r, Config())
    assert abs(res.metrics["total_return"]) < 1e-12, "PnL sin posición"
    assert res.metrics["n_active"] == 0
    return "posición plana => PnL exactamente cero"


@check("cuantiles: q10 <= q50 <= q90 siempre")
def _t_quantiles():
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(600, 6)),
                     columns=[f"f{i}" for i in range(6)])
    y = X["f0"] * 0.5 + rng.normal(0, 1, 600)
    qb = QuantileBundle().fit(X, y.to_numpy(), (0.1, 0.5, 0.9), seed=0)
    q = qb.predict(X)
    assert (q["q10"] <= q["q50"] + 1e-9).all() and (q["q50"] <= q["q90"] + 1e-9).all(), \
        "cuantiles cruzados"
    cov = float(((y >= q["q10"]) & (y <= q["q90"])).mean())
    assert cov > 0.60, f"cobertura del intervalo 10-90 solo {cov:.2f}"
    return f"monotonía garantizada; cobertura P10-P90 = {cov:.2f}"


# ---------------------------------------------------------------------------
# 6. Integridad de datos
# ---------------------------------------------------------------------------
@check("datos: OHLCV corrupto se rechaza")
def _t_validate():
    df, _ = make_ohlcv(n=300, seed=7)
    validate_ohlcv(df, "ok", min_rows=100, verbose=0)
    for name, mut in [
        ("High<Low", lambda d: d.assign(High=d["Low"] - 1)),
        ("precio<=0", lambda d: d.assign(Close=d["Close"] * 0 - 1)),
        ("desordenado", lambda d: d.iloc[::-1]),
    ]:
        try:
            validate_ohlcv(mut(df.copy()), name, min_rows=100, verbose=0)
        except DataIntegrityError:
            continue
        raise AssertionError(f"validate_ohlcv aceptó datos corruptos: {name}")
    return "rechaza High<Low, precios no positivos e índices desordenados"


# ---------------------------------------------------------------------------
# 7. Las dos pruebas que de verdad importan
# ---------------------------------------------------------------------------
@check("SEÑAL: el sistema recupera una relación inyectada (AUC > 0.55)")
def _t_recover():
    from .pipeline import run_experiment
    md = make_market_data(n=2200, seed=11, signal="mean_revert",
                          signal_strength=0.45, with_context=False)
    res = run_experiment(_fast_cfg("SYNTH_SIGNAL"), md=md)
    auc = res.metrics.get("auc_oos", np.nan)
    sh = res.metrics.get("sharpe", np.nan)
    assert np.isfinite(auc) and auc > 0.55, \
        f"AUC OOS={auc:.3f}: no recupera una señal fuerte y conocida"
    return f"AUC OOS={auc:.3f}, Sharpe OOS={sh:.2f}, veredicto={res.verdict['decision']}"


@check("RUIDO: el sistema NO encuentra alpha en un paseo aleatorio")
def _t_noise():
    from .pipeline import run_experiment
    md = make_market_data(n=2200, seed=12, signal="none", with_context=False)
    res = run_experiment(_fast_cfg("SYNTH_NOISE"), md=md)
    auc = res.metrics.get("auc_oos", np.nan)
    sh = res.metrics.get("sharpe", np.nan)
    assert not (np.isfinite(auc) and auc > 0.60), \
        f"AUC OOS={auc:.3f} sobre ruido puro: hay fuga en alguna parte"
    assert res.verdict["decision"] != "GO", \
        f"veredicto GO sobre ruido puro (Sharpe={sh:.2f}): el filtro no funciona"
    return f"AUC OOS={auc:.3f}, Sharpe={sh:.2f}, veredicto={res.verdict['decision']} (correcto)"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(verbose: bool = True, stop_on_fail: bool = False) -> list[CheckResult]:
    suppress_noisy_warnings()
    results: list[CheckResult] = []
    width = 68
    if verbose:
        print("=" * width)
        print("ALPHAFORGE — AUTODIAGNÓSTICO")
        print("=" * width)
    for name, fn in _REGISTRY:
        t0 = time.time()
        try:
            detail = fn() or "ok"
            r = CheckResult(name, True, str(detail), time.time() - t0)
        except Exception as e:                     # noqa: BLE001
            tb = traceback.format_exc(limit=3) if verbose else ""
            r = CheckResult(name, False, f"{type(e).__name__}: {e}\n{tb}",
                            time.time() - t0)
        results.append(r)
        if verbose:
            mark = "PASA" if r.passed else "FALLA"
            print(f"[{mark:5s}] {name}  ({r.seconds:.1f}s)")
            print(f"         {r.detail.splitlines()[0] if r.detail else ''}")
            if not r.passed and len(r.detail.splitlines()) > 1:
                for line in r.detail.splitlines()[1:]:
                    print(f"         {line}")
        if not r.passed and stop_on_fail:
            break
    n_ok = sum(r.passed for r in results)
    if verbose:
        print("-" * width)
        print(f"RESULTADO: {n_ok}/{len(results)} comprobaciones superadas "
              f"({sum(r.seconds for r in results):.0f}s)")
        if n_ok < len(results):
            print("\n>>> HAY FALLOS. No operes con este sistema hasta resolverlos.")
        print("=" * width)
    return results


def main() -> int:
    res = run_all()
    return 0 if all(r.passed for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
