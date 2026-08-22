"""Generador de mercados sintéticos.

Sirve para dos pruebas que ningún backtest serio debería saltarse:

  A) RUIDO PURO: si le das al sistema un paseo aleatorio, tiene que decir
     "aquí no hay nada". Si encuentra alpha en ruido, el sistema está roto.

  B) SEÑAL CONOCIDA: si inyectas una relación real y medible, tiene que
     encontrarla. Si no la encuentra, el sistema tampoco vale.

Un pipeline que falle cualquiera de las dos es papel mojado, por muy bonito
que sea el equity curve que dibuje sobre datos reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AnchorResult, MarketData, _residual_stats


def make_ohlcv(n: int = 3000, seed: int = 0, start: str = "2012-01-03",
               annual_vol: float = 0.22, drift: float = 0.06,
               signal: str = "none", signal_strength: float = 0.0,
               ) -> tuple[pd.DataFrame, pd.Series]:
    """Genera OHLCV diario + precio del ancla (T-30').

    signal:
      'none'         -> paseo aleatorio con volatilidad estocástica
      'mean_revert'  -> el retorno de mañana depende (-) del movimiento de hoy
                        hasta el ancla: reversión intradía->overnight
      'momentum'     -> depende (+) del momentum de 5 días
      'regime'       -> la señal solo existe cuando la volatilidad es alta
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)

    # volatilidad estocástica (proceso log-AR1) para que no sea gaussiano plano
    logv = np.zeros(n)
    logv[0] = np.log(annual_vol / np.sqrt(252))
    for t in range(1, n):
        logv[t] = 0.97 * logv[t - 1] + 0.03 * np.log(annual_vol / np.sqrt(252)) \
            + 0.12 * rng.normal()
    sig = np.exp(logv)

    mu = drift / 252
    eps = rng.standard_normal(n) * sig
    close = np.zeros(n)
    close[0] = 100.0

    anchor = np.zeros(n)
    open_ = np.zeros(n)
    intraday_frac = 0.60          # cuánto del movimiento diario ocurre en sesión

    prev_state = 0.0
    mom5 = 0.0
    for t in range(1, n):
        # componente inducido por la señal (depende del estado en t-1)
        extra = 0.0
        if signal == "mean_revert":
            extra = -signal_strength * prev_state * sig[t]
        elif signal == "momentum":
            extra = signal_strength * np.tanh(mom5 / max(sig[t] * np.sqrt(5), 1e-9)) * sig[t]
        elif signal == "regime":
            hot = 1.0 if sig[t] > np.median(sig[:max(t, 30)]) else 0.0
            extra = -signal_strength * hot * prev_state * sig[t]

        r_total = mu + eps[t] + extra
        close[t] = close[t - 1] * (1 + r_total)

        # descomposición: gap de apertura + recorrido intradía
        gap = r_total * (1 - intraday_frac) + 0.15 * sig[t] * rng.normal()
        open_[t] = close[t - 1] * (1 + gap)
        # el ancla está al 85% del camino entre apertura y cierre
        anchor[t] = open_[t] + 0.85 * (close[t] - open_[t]) \
            + 0.05 * sig[t] * close[t] * rng.normal()

        # estado que genera la señal de mañana: movimiento de hoy hasta el ancla
        prev_state = (anchor[t] / close[t - 1] - 1) / max(sig[t], 1e-9)
        if t >= 6:
            mom5 = close[t] / close[t - 5] - 1

    open_[0], anchor[0] = close[0], close[0]

    noise = np.abs(rng.standard_normal(n)) * sig * close * 0.6
    high = np.maximum.reduce([open_, close, anchor]) + noise
    low = np.minimum.reduce([open_, close, anchor]) - noise
    low = np.maximum(low, 0.01)
    volume = np.abs(rng.lognormal(15, 0.4, n)) * (1 + 3 * np.abs(eps) / sig.mean())

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": volume}, index=idx)
    anchor_s = pd.Series(anchor, index=idx, name="anchor")
    return df, anchor_s


def make_market_data(ticker: str = "SYNTH", n: int = 3000, seed: int = 0,
                     signal: str = "none", signal_strength: float = 0.0,
                     with_context: bool = True) -> MarketData:
    daily, anchor = make_ohlcv(n=n, seed=seed, signal=signal,
                               signal_strength=signal_strength)
    resid = daily["Close"] / anchor - 1
    ar = AnchorResult(price=anchor, source="synthetic_30m", coverage=1.0,
                      residual_stats=_residual_stats(resid.dropna()),
                      intraday_volume=daily["Volume"] * 0.9)
    ctx = {}
    if with_context:
        for i, name in enumerate(("SPY", "^VIX")):
            d, _ = make_ohlcv(n=n, seed=seed + 100 + i,
                              annual_vol=0.16 if i == 0 else 0.9)
            ctx[name] = d
    return MarketData(ticker=ticker, daily=daily, anchor=ar, context=ctx, reports={})
