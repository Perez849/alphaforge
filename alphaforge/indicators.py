"""Indicadores y osciladores en pandas/numpy puro (sin TA-Lib).

Todas las funciones son causales: el valor en t usa exclusivamente datos
hasta t inclusive. El desplazamiento temporal se aplica después, en features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Suavizado de Wilder (equivale a EMA con alpha = 1/n)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = _wilder(d.clip(lower=0), n)
    dn = _wilder((-d).clip(lower=0), n)
    rs = up / dn.where(dn > 1e-12)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0).clip(0, 100)


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()],
                     axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = sma(close, n)
    sd = close.rolling(n, min_periods=max(2, n // 2)).std(ddof=0)
    upper, lower = m + k * sd, m - k * sd
    width = (upper - lower) / m.where(m.abs() > 1e-12)
    pctb = (close - lower) / (upper - lower).where((upper - lower).abs() > 1e-12)
    return pd.DataFrame({f"bb_width_{n}": width, f"bb_pctb_{n}": pctb.clip(-1, 2)})


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 14, d: int = 3) -> pd.DataFrame:
    hh = high.rolling(n, min_periods=max(2, n // 2)).max()
    ll = low.rolling(n, min_periods=max(2, n // 2)).min()
    rng = (hh - ll).where((hh - ll).abs() > 1e-12)
    k = 100 * (close - ll) / rng
    return pd.DataFrame({f"stoch_k_{n}": k.clip(0, 100),
                         f"stoch_d_{n}": k.rolling(d, min_periods=1).mean().clip(0, 100)})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=max(2, n // 2)).max()
    ll = low.rolling(n, min_periods=max(2, n // 2)).min()
    rng = (hh - ll).where((hh - ll).abs() > 1e-12)
    return (-100 * (hh - close) / rng).clip(-100, 0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr_n = _wilder(true_range(high, low, close), n)
    tr_n = tr_n.where(tr_n.abs() > 1e-12)
    pdi = 100 * _wilder(pd.Series(plus_dm, index=high.index), n) / tr_n
    mdi = 100 * _wilder(pd.Series(minus_dm, index=high.index), n) / tr_n
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where((pdi + mdi).abs() > 1e-12)
    return pd.DataFrame({f"adx_{n}": _wilder(dx, n).clip(0, 100),
                         f"pdi_{n}": pdi.clip(0, 100),
                         f"mdi_{n}": mdi.clip(0, 100)})


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    m = tp.rolling(n, min_periods=max(2, n // 2)).mean()
    md = (tp - m).abs().rolling(n, min_periods=max(2, n // 2)).mean()
    return ((tp - m) / (0.015 * md.where(md.abs() > 1e-12))).clip(-500, 500)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume.fillna(0)).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume.fillna(0)
    d = tp.diff()
    pos = rmf.where(d > 0, 0.0).rolling(n, min_periods=max(2, n // 2)).sum()
    neg = rmf.where(d < 0, 0.0).rolling(n, min_periods=max(2, n // 2)).sum()
    ratio = pos / neg.where(neg.abs() > 1e-12)
    return (100 - 100 / (1 + ratio)).fillna(50.0).clip(0, 100)


def realized_vol(ret: pd.Series, n: int, annualize: bool = True) -> pd.Series:
    v = ret.rolling(n, min_periods=max(3, n // 2)).std(ddof=0)
    return v * np.sqrt(252) if annualize else v


def downside_vol(ret: pd.Series, n: int) -> pd.Series:
    neg = ret.where(ret < 0, 0.0)
    return neg.rolling(n, min_periods=max(3, n // 2)).std(ddof=0) * np.sqrt(252)


def parkinson_vol(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    hl = np.log(high / low.where(low > 0)) ** 2
    return np.sqrt(hl.rolling(n, min_periods=max(3, n // 2)).mean()
                   / (4 * np.log(2))) * np.sqrt(252)


def range_position(close: pd.Series, n: int) -> pd.Series:
    hh = close.rolling(n, min_periods=max(3, n // 2)).max()
    ll = close.rolling(n, min_periods=max(3, n // 2)).min()
    rng = (hh - ll).where((hh - ll).abs() > 1e-12)
    return ((close - ll) / rng).clip(0, 1)


def hurst_proxy(ret: pd.Series, n: int = 63) -> pd.Series:
    """Proxy barato del exponente de Hurst: ratio de varianzas escaladas.

    >0.5 sugiere tendencia; <0.5 reversión. Implementado como
    log(Var(r_2) / (2 Var(r_1))) / (2 log 2) + 0.5
    """
    v1 = ret.rolling(n, min_periods=n // 2).var(ddof=0)
    r2 = ret.rolling(2).sum()
    v2 = r2.rolling(n, min_periods=n // 2).var(ddof=0)
    ratio = v2 / (2 * v1).where((2 * v1).abs() > 1e-16)
    return (np.log(ratio.where(ratio > 1e-12)) / (2 * np.log(2)) + 0.5).clip(0, 1)


def rolling_beta(ret: pd.Series, bench: pd.Series, n: int = 63) -> pd.Series:
    cov = ret.rolling(n, min_periods=n // 2).cov(bench)
    var = bench.rolling(n, min_periods=n // 2).var(ddof=0)
    return cov / var.where(var.abs() > 1e-16)


def rolling_corr(a: pd.Series, b: pd.Series, n: int = 63) -> pd.Series:
    return a.rolling(n, min_periods=n // 2).corr(b).clip(-1, 1)
