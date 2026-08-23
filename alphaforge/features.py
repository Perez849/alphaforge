"""Construcción de la matriz de features.

CONTRATO TEMPORAL (lo más importante del fichero)
--------------------------------------------------
La decisión se toma en el instante A(t) = 15:30 ET del día t. En ese momento
conocemos:

  (1) TODAS las barras diarias completas hasta t-1        -> bloque BASE
  (2) el Open(t) de hoy                                   -> bloque TODAY
  (3) el precio del ancla P_a(t) y el volumen acumulado   -> bloque TODAY

NO conocemos High(t), Low(t), Close(t) ni Volume(t) completos: incluyen los
últimos 30 minutos. Usarlos sería fuga.

Implementación: se calculan todos los indicadores sobre el diario, se aplica un
`.shift(1)` GLOBAL al bloque BASE (deja todo en información de t-1) y solo
después se añade el bloque TODAY, construido explícitamente con Open(t) y P_a(t)
contra referencias de t-1. Cualquier feature nueva debe entrar por uno de los
dos bloques; no hay tercera vía.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import Config
from .data import MarketData
from .utils import get_logger, pct_change_safe, sanitize, zscore

TODAY_PREFIX = "tdy_"


# ---------------------------------------------------------------------------
# Bloque BASE: barras diarias completas (se desplaza 1 día)
# ---------------------------------------------------------------------------
def _base_block(daily: pd.DataFrame, fc) -> pd.DataFrame:
    o, h, l, c, v = (daily["Open"], daily["High"], daily["Low"],
                     daily["Close"], daily["Volume"])
    r = pct_change_safe(c)
    f: dict[str, pd.Series] = {}

    # --- retornos y momentum
    for n in fc.roc_periods:
        f[f"roc_{n}"] = pct_change_safe(c, n)
        if n > 1:
            f[f"roc_{n}_z"] = zscore(pct_change_safe(c, n), 252)
    for k in range(1, 6):
        f[f"ret_lag_{k}"] = r.shift(k - 1)          # lag 0 = retorno de ayer
    f["ret_skew_21"] = r.rolling(21, min_periods=10).skew()
    f["ret_kurt_63"] = r.rolling(63, min_periods=30).kurt()
    f["up_days_21"] = (r > 0).rolling(21, min_periods=10).mean()
    f["streak"] = _streak(r)

    # --- medias móviles y distancias (normalizadas: sin escala de precio)
    for n in fc.ma_periods:
        s, e = ind.sma(c, n), ind.ema(c, n)
        f[f"dist_sma_{n}"] = c / s.where(s.abs() > 1e-12) - 1
        f[f"dist_ema_{n}"] = c / e.where(e.abs() > 1e-12) - 1
        f[f"sma_slope_{n}"] = pct_change_safe(s, 5)
    f["sma_cross_20_50"] = (ind.sma(c, 20) / ind.sma(c, 50).where(
        ind.sma(c, 50).abs() > 1e-12) - 1)
    f["sma_cross_50_200"] = (ind.sma(c, 50) / ind.sma(c, 200).where(
        ind.sma(c, 200).abs() > 1e-12) - 1)

    # --- osciladores
    for n in fc.rsi_periods:
        f[f"rsi_{n}"] = ind.rsi(c, n)
        f[f"rsi_{n}_d"] = ind.rsi(c, n).diff()
    m = ind.macd(c)
    norm = c.where(c.abs() > 1e-12)
    for col in m.columns:
        f[col] = m[col] / norm                     # normalizado por precio
    for n in fc.stoch_periods:
        st = ind.stochastic(h, l, c, n)
        for col in st.columns:
            f[col] = st[col]
    for n in fc.adx_periods:
        ax = ind.adx(h, l, c, n)
        for col in ax.columns:
            f[col] = ax[col]
    f["cci_20"] = ind.cci(h, l, c, 20)
    f["willr_14"] = ind.williams_r(h, l, c, 14)
    f["mfi_14"] = ind.mfi(h, l, c, v, 14)

    # --- volatilidad y régimen
    for n in fc.vol_windows:
        f[f"rvol_{n}"] = ind.realized_vol(r, n)
        f[f"dvol_{n}"] = ind.downside_vol(r, n)
    f["pvol_21"] = ind.parkinson_vol(h, l, 21)
    f["vol_ratio_5_63"] = (ind.realized_vol(r, 5) /
                           ind.realized_vol(r, 63).where(ind.realized_vol(r, 63) > 1e-8))
    f["vol_z_252"] = zscore(ind.realized_vol(r, 21), 252)
    f["hurst_63"] = ind.hurst_proxy(r, 63)
    for n in fc.atr_periods:
        a = ind.atr(h, l, c, n)
        f[f"atr_{n}_norm"] = a / c.where(c.abs() > 1e-12)
    for n in fc.bb_periods:
        bb = ind.bollinger(c, n)
        for col in bb.columns:
            f[col] = bb[col]

    # --- estructura de rango
    for n in fc.range_windows:
        f[f"rangepos_{n}"] = ind.range_position(c, n)
        hh = c.rolling(n, min_periods=n // 2).max()
        f[f"drawdown_{n}"] = c / hh.where(hh.abs() > 1e-12) - 1

    # --- volumen
    vol_ma = v.rolling(21, min_periods=10).mean()
    f["vol_rel_21"] = v / vol_ma.where(vol_ma > 1e-9)
    f["vol_z_63"] = zscore(v.astype(float), 63)
    f["obv_slope_21"] = pct_change_safe(ind.obv(c, v).abs() + 1, 21)
    f["dollar_vol_z"] = zscore((c * v).astype(float), 63)

    # --- microestructura de la vela
    rng = (h - l).where((h - l).abs() > 1e-12)
    f["candle_body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(c, o)) / rng
    f["lower_wick"] = (np.minimum(c, o) - l) / rng
    f["gap_open"] = o / c.shift(1).where(c.shift(1).abs() > 1e-12) - 1
    f["intraday_ret"] = c / o.where(o.abs() > 1e-12) - 1
    f["overnight_vs_intraday_21"] = (
        f["gap_open"].rolling(21, min_periods=10).mean()
        - f["intraday_ret"].rolling(21, min_periods=10).mean()
    )

    out = pd.DataFrame(f, index=daily.index)
    return out


def _streak(r: pd.Series) -> pd.Series:
    """Racha consecutiva de días al alza (+) o a la baja (-)."""
    sign = np.sign(r.fillna(0.0))
    out = np.zeros(len(sign))
    prev = 0.0
    for i, s in enumerate(sign.to_numpy()):
        if s == 0:
            out[i] = 0.0
        elif np.sign(prev) == s:
            out[i] = prev + s
        else:
            out[i] = s
        prev = out[i]
    return pd.Series(np.clip(out, -10, 10), index=r.index)


# ---------------------------------------------------------------------------
# Multi-timeframe: semanal y mensual
# ---------------------------------------------------------------------------
def _resampled_block(daily: pd.DataFrame, rule: str, tag: str) -> pd.DataFrame:
    agg = daily.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    if len(agg) < 40:
        return pd.DataFrame(index=daily.index)
    c, h, l = agg["Close"], agg["High"], agg["Low"]
    r = pct_change_safe(c)
    f = {
        f"{tag}_ret_1": r,
        f"{tag}_ret_4": pct_change_safe(c, 4),
        f"{tag}_rsi_14": ind.rsi(c, 14),
        f"{tag}_dist_sma_10": c / ind.sma(c, 10).where(ind.sma(c, 10).abs() > 1e-12) - 1,
        f"{tag}_dist_sma_30": c / ind.sma(c, 30).where(ind.sma(c, 30).abs() > 1e-12) - 1,
        f"{tag}_rvol_12": ind.realized_vol(r, 12, annualize=False),
        f"{tag}_rangepos_26": ind.range_position(c, 26),
        f"{tag}_adx_14": ind.adx(h, l, c, 14)[f"adx_14"],
        f"{tag}_macd_hist": ind.macd(c)["macd_hist"] / c.where(c.abs() > 1e-12),
    }
    blk = pd.DataFrame(f, index=agg.index)
    # La barra semanal/mensual solo está CERRADA al final del periodo: shift(1)
    # antes de reindexar evita usar la vela en curso.
    blk = blk.shift(1)
    return blk.reindex(daily.index, method="ffill")


# ---------------------------------------------------------------------------
# Cross-asset
# ---------------------------------------------------------------------------
def _cross_block(daily: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not context:
        return pd.DataFrame(index=daily.index)
    r = pct_change_safe(daily["Close"])
    f: dict[str, pd.Series] = {}
    for name, df in context.items():
        tag = name.replace("^", "").lower()
        cc = df["Close"].reindex(daily.index).ffill(limit=5)
        rc = pct_change_safe(cc)
        f[f"x_{tag}_ret_1"] = rc
        f[f"x_{tag}_ret_5"] = pct_change_safe(cc, 5)
        f[f"x_{tag}_ret_21"] = pct_change_safe(cc, 21)
        f[f"x_{tag}_dist_sma50"] = (cc / ind.sma(cc, 50).where(
            ind.sma(cc, 50).abs() > 1e-12) - 1)
        f[f"x_{tag}_beta_63"] = ind.rolling_beta(r, rc, 63)
        f[f"x_{tag}_corr_63"] = ind.rolling_corr(r, rc, 63)
        f[f"x_{tag}_rs_21"] = (pct_change_safe(daily["Close"], 21)
                               - pct_change_safe(cc, 21))
        if "vix" in tag:
            f["x_vix_level"] = cc
            f["x_vix_z_252"] = zscore(cc, 252)
            f["x_vix_term"] = cc / ind.sma(cc, 21).where(ind.sma(cc, 21).abs() > 1e-9) - 1
    return pd.DataFrame(f, index=daily.index)


# ---------------------------------------------------------------------------
# Estructura de volatilidad y posicionamiento
# ---------------------------------------------------------------------------
def _vol_structure_block(daily: pd.DataFrame,
                         context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Lo que dice el mercado de opciones sobre el riesgo que viene.

    El nivel del VIX a secas es la versión pobre de esta información. Lo que
    tiene contenido es la FORMA de la curva:

      * VIX9D/VIX y VIX/VIX3M: pendiente de la estructura temporal. En contango
        normal el mercado no espera sobresaltos; en backwardation hay estrés
        inmediato y el comportamiento de los retornos cambia de régimen.
      * VVIX: lo que cuesta la volatilidad de la volatilidad. Sube cuando se
        compra convexidad, es decir, cuando alguien se está cubriendo en serio.
      * SKEW: precio relativo de las puts lejanas frente a las calls. Mide
        cuánto se está pagando por protegerse de una caída brusca.

    Todo esto son series públicas de CBOE con histórico largo, y ninguna se
    deriva del precio del valor: aportan información que el OHLCV no contiene.
    """
    have = {k.replace("^", "").upper(): v for k, v in context.items()}
    f: dict[str, pd.Series] = {}

    def get(name: str) -> pd.Series | None:
        d = have.get(name)
        return None if d is None else d["Close"].reindex(daily.index).ffill(limit=5)

    vix, v9d, v3m = get("VIX"), get("VIX9D"), get("VIX3M")
    vvix, skew = get("VVIX"), get("SKEW")

    if vix is not None:
        if v9d is not None:                      # pendiente corta
            r = v9d / vix.where(vix > 1e-9)
            f["vs_term_9d_1m"] = r
            f["vs_term_9d_1m_z"] = zscore(r, 252)
            f["vs_backwardation_corto"] = (r > 1.0).astype(float)
        if v3m is not None:                      # pendiente larga
            r = vix / v3m.where(v3m > 1e-9)
            f["vs_term_1m_3m"] = r
            f["vs_term_1m_3m_z"] = zscore(r, 252)
            f["vs_backwardation"] = (r > 1.0).astype(float)
            f["vs_term_slope_chg"] = r.diff(5)
        f["vs_vix_chg_1"] = vix.diff()
        f["vs_vix_chg_5"] = vix.diff(5)
        f["vs_vix_rel_ma21"] = vix / ind.sma(vix, 21).where(ind.sma(vix, 21) > 1e-9) - 1

    if vvix is not None:
        f["vs_vvix_z"] = zscore(vvix, 252)
        f["vs_vvix_chg5"] = vvix.diff(5)
        if vix is not None:
            f["vs_vvix_vix_ratio"] = vvix / vix.where(vix > 1e-9)

    if skew is not None:
        f["vs_skew_z"] = zscore(skew, 252)
        f["vs_skew_chg5"] = skew.diff(5)

    return pd.DataFrame(f, index=daily.index)


def _overnight_block(daily: pd.DataFrame) -> pd.DataFrame:
    """Separa el retorno nocturno del intradía.

    Está documentado que ambos componentes tienen dinámicas propias y a menudo
    opuestas: quien compra en la apertura no es quien compra en el cierre. El
    objetivo del sistema (ancla -> cierre siguiente) los mezcla, y mezclados se
    cancelan. Estas features los mantienen separados para que el modelo pueda
    usar cada uno por su lado.
    """
    o, c = daily["Open"], daily["Close"]
    prev = c.shift(1)
    on = o / prev.where(prev.abs() > 1e-12) - 1          # nocturno
    id_ = c / o.where(o.abs() > 1e-12) - 1               # sesión
    f: dict[str, pd.Series] = {"on_ret": on, "id_ret": id_}
    for n in (5, 21, 63):
        f[f"on_mean_{n}"] = on.rolling(n, min_periods=n // 2).mean()
        f[f"id_mean_{n}"] = id_.rolling(n, min_periods=n // 2).mean()
        f[f"on_id_spread_{n}"] = f[f"on_mean_{n}"] - f[f"id_mean_{n}"]
        f[f"on_id_corr_{n}"] = on.rolling(n, min_periods=n // 2).corr(id_)
    f["on_vol_21"] = on.rolling(21, min_periods=10).std(ddof=0)
    f["id_vol_21"] = id_.rolling(21, min_periods=10).std(ddof=0)
    v = f["id_vol_21"]
    f["on_id_vol_ratio"] = f["on_vol_21"] / v.where(v > 1e-9)
    f["on_streak"] = np.sign(on).rolling(5, min_periods=3).sum()
    f["id_streak"] = np.sign(id_).rolling(5, min_periods=3).sum()
    # reversión: ¿la sesión deshace lo que hizo la noche?
    f["on_then_id"] = np.sign(on) * np.sign(id_)
    f["on_then_id_21"] = f["on_then_id"].rolling(21, min_periods=10).mean()
    return pd.DataFrame(f, index=daily.index)


# ---------------------------------------------------------------------------
# Calendario (no necesita shift: es determinista y conocido de antemano)
# ---------------------------------------------------------------------------
def _calendar_block(idx: pd.DatetimeIndex) -> pd.DataFrame:
    dow = idx.dayofweek
    mon = idx.month
    dom = idx.day
    f = {
        "cal_dow_sin": np.sin(2 * np.pi * dow / 5),
        "cal_dow_cos": np.cos(2 * np.pi * dow / 5),
        "cal_month_sin": np.sin(2 * np.pi * mon / 12),
        "cal_month_cos": np.cos(2 * np.pi * mon / 12),
        "cal_turn_of_month": ((dom >= 28) | (dom <= 3)).astype(float),
        "cal_dom_norm": dom / 31.0,
        "cal_quarter_end": idx.is_quarter_end.astype(float),
        "cal_week_of_year": idx.isocalendar().week.to_numpy() / 53.0,
    }
    return pd.DataFrame(f, index=idx)


# ---------------------------------------------------------------------------
# Bloque TODAY: única información del día t admisible
# ---------------------------------------------------------------------------
def _today_block(daily: pd.DataFrame, anchor: pd.Series,
                 intraday_vol: pd.Series | None) -> pd.DataFrame:
    c_prev = daily["Close"].shift(1)
    o = daily["Open"]
    a = anchor.reindex(daily.index)
    denom_prev = c_prev.where(c_prev.abs() > 1e-12)
    f = {
        f"{TODAY_PREFIX}gap": o / denom_prev - 1,
        f"{TODAY_PREFIX}anchor_ret": a / denom_prev - 1,          # retorno de hoy hasta 15:30
        f"{TODAY_PREFIX}anchor_vs_open": a / o.where(o.abs() > 1e-12) - 1,
    }
    ar = pd.Series(f[f"{TODAY_PREFIX}anchor_ret"], index=daily.index)

    # posición del ancla respecto a estructura de AYER (todo shift(1) explícito)
    for n in (20, 50, 200):
        s_prev = ind.sma(daily["Close"], n).shift(1)
        f[f"{TODAY_PREFIX}anchor_dist_sma{n}"] = a / s_prev.where(s_prev.abs() > 1e-12) - 1
    hh_prev = daily["High"].rolling(21, min_periods=10).max().shift(1)
    ll_prev = daily["Low"].rolling(21, min_periods=10).min().shift(1)
    rng = (hh_prev - ll_prev).where((hh_prev - ll_prev).abs() > 1e-12)
    f[f"{TODAY_PREFIX}anchor_rangepos21"] = ((a - ll_prev) / rng).clip(-0.5, 1.5)

    # magnitud del movimiento de hoy en unidades de volatilidad de ayer
    atr_prev = (ind.atr(daily["High"], daily["Low"], daily["Close"], 14)
                / daily["Close"]).shift(1)
    f[f"{TODAY_PREFIX}anchor_ret_atr"] = ar / atr_prev.where(atr_prev > 1e-8)
    rvol_prev = ind.realized_vol(pct_change_safe(daily["Close"]), 21,
                                 annualize=False).shift(1)
    f[f"{TODAY_PREFIX}anchor_ret_sigma"] = ar / rvol_prev.where(rvol_prev > 1e-8)
    f[f"{TODAY_PREFIX}anchor_ret_z252"] = zscore(ar, 252)

    if intraday_vol is not None:
        iv = intraday_vol.reindex(daily.index).astype(float)
        ivm = iv.rolling(21, min_periods=5).mean().shift(1)
        f[f"{TODAY_PREFIX}vol_partial_rel"] = iv / ivm.where(ivm > 1e-9)

    return pd.DataFrame(f, index=daily.index)


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------
def build_features(md: MarketData, cfg: Config) -> pd.DataFrame:
    log = get_logger(cfg.verbose)
    fc = cfg.features
    daily = md.daily

    base = _base_block(daily, fc)
    if fc.weekly:
        base = base.join(_resampled_block(daily, "W-FRI", "wk"), how="left")
    if fc.monthly:
        base = base.join(_resampled_block(daily, "ME", "mo"), how="left")
    if fc.cross_asset:
        base = base.join(_cross_block(daily, md.context), how="left")
    if fc.vol_structure:
        base = base.join(_vol_structure_block(daily, md.context), how="left")
    if fc.overnight:
        base = base.join(_overnight_block(daily), how="left")

    # ---- DESPLAZAMIENTO GLOBAL: todo el bloque BASE pasa a ser info de t-1 ----
    base = base.shift(1)
    base.columns = [f"b_{c}" for c in base.columns]

    today = _today_block(daily, md.anchor.price, md.anchor.intraday_volume)
    X = base.join(today, how="left")

    if fc.calendar:
        X = X.join(_calendar_block(daily.index), how="left")

    # Solo se limpian infinitos aquí. El recorte de colas NO se hace sobre la
    # serie completa: sus umbrales saldrían de datos futuros respecto a
    # cualquier fold. Se aplica más tarde, ajustado únicamente en train
    # (pipeline._prepare_fold -> TrainFittedClipper).
    X = sanitize(X, "features", clip_sigma=0.0)

    # Poda: varianza nula y columnas casi vacías.
    # La cobertura se evalúa sobre el PRIMER 60% de la muestra, no sobre el
    # total: decidir qué columnas existen mirando el dataset entero es una
    # decisión de modelado tomada con datos futuros. Es leve, pero es del mismo
    # tipo que el recorte de colas que ya se movió a train.
    head = max(250, int(0.6 * len(X)))
    keep = []
    for col in X.columns:
        s = X[col].iloc[:head]
        if s.notna().mean() < 0.60:
            continue
        if float(np.nanvar(s.to_numpy(dtype=float))) < fc.min_variance:
            continue
        keep.append(col)
    dropped = len(X.columns) - len(keep)
    X = X[keep]
    log.info(f"features construidas: {X.shape[1]} columnas "
             f"({dropped} descartadas por varianza/cobertura)")
    return X


def prune_correlated(X: pd.DataFrame, max_abs_corr: float = 0.95,
                     verbose: int = 1) -> list[str]:
    """Elimina redundancia lineal quedándose con la de mayor varianza explicada.

    Se ejecuta SOLO sobre datos de entrenamiento para no filtrar información.
    """
    if X.shape[1] < 2:
        return list(X.columns)
    corr_df = X.corr(numeric_only=True).abs()
    C = np.asarray(corr_df, dtype=float).copy()      # pandas 3 devuelve solo lectura
    np.fill_diagonal(C, 0.0)
    C = np.nan_to_num(C, nan=0.0)
    cols = list(corr_df.columns)
    pos = {c: i for i, c in enumerate(cols)}

    order = [c for c in X.std(ddof=0).sort_values(ascending=False).index if c in pos]
    kept: list[str] = []
    for col in order:
        i = pos[col]
        if all(C[i, pos[k]] < max_abs_corr for k in kept):
            kept.append(col)
    if not kept:                                     # red de seguridad
        kept = cols[: min(20, len(cols))]
    get_logger(verbose).debug(
        f"poda de correlación: {X.shape[1]} -> {len(kept)} features")
    return kept
