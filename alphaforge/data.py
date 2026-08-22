"""Capa de datos: descarga, caché, validación y construcción del ANCLA intradía.

EL PROBLEMA DEL ANCLA
---------------------
Queremos decidir a las 15:30 ET (30' antes del cierre US), no a cierre, para no
regalar el gap de apertura al día siguiente. Eso obliga a que:

  * el precio de entrada sea  P_anchor(t) = precio a las 15:30 ET del día t
  * el objetivo sea           y(t) = Close(t+1) / P_anchor(t) - 1

Pero los proveedores gratuitos (yfinance) solo sirven intradía reciente:
60 días para 30m, ~730 días para 1h. No hay 20 años de barras de 30 minutos.

SOLUCIÓN: ancla en dos capas + calibración explícita del error.

  Capa A (histórico largo, proxy):  P_anchor(t) ≈ Close(t)
  Capa B (histórico reciente, real): P_anchor(t) desde barras 30m/1h

Con la capa B medimos la distribución del residuo del proxy
        eps(t) = Close(t)/P_anchor_real(t) - 1
y la reportamos: media, sigma, autocorrelación y, sobre todo, cuánto se come
del edge. Ese eps se inyecta como ruido en el backtest largo (Monte Carlo) para
que el Sharpe reportado NO sea optimista. Si eps no es despreciable frente al
retorno esperado, el sistema lo dice a gritos en el informe.

En producción (`predict`) se usa SIEMPRE el ancla real.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, DataConfig
from .utils import DataIntegrityError, get_logger, pct_change_safe

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Descarga con caché
# ---------------------------------------------------------------------------
def _cache_path(cfg: DataConfig, ticker: str, interval: str,
                start: str | None = None, end: str | None = None) -> str:
    """La clave incluye el rango: si no, pedir 2006 devolvía la caché de 2020."""
    d = os.path.expanduser(cfg.cache_dir)
    os.makedirs(d, exist_ok=True)
    safe = ticker.replace("^", "IDX_").replace("/", "_").replace("=", "_")
    rng = f"{start or 'max'}_{end or 'now'}".replace("-", "")
    return os.path.join(d, f"{safe}__{interval}__{rng}.parquet")


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance devuelve MultiIndex según versión/parámetros. Normalizamos."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        if {"Open", "Close"} & lvl0:
            df = df.droplevel(-1, axis=1)
        else:
            df = df.droplevel(0, axis=1)
    df.columns = [str(c).title().replace(" ", "") for c in df.columns]
    if "Adjclose" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adjclose": "Close"})
    return df


def download(ticker: str, interval: str, cfg: DataConfig, start: str | None = None,
             end: str | None = None, use_cache: bool = True,
             verbose: int = 1) -> pd.DataFrame:
    """Descarga OHLCV con caché en disco y reintentos."""
    log = get_logger(verbose)
    path = _cache_path(cfg, ticker, interval, start, end)

    if use_cache and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < cfg.max_cache_age_hours:
            try:
                df = pd.read_parquet(path)
                log.debug(f"caché HIT {ticker}/{interval} ({len(df)} filas, {age_h:.1f}h)")
                return df
            except Exception as e:                       # caché corrupta -> se ignora
                log.warning(f"caché ilegible para {ticker}/{interval}: {e}")

    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Falta yfinance. Instala: pip install yfinance") from e

    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(
                ticker, interval=interval, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
                period=None if start else _default_period(interval),
            )
            if df is None or len(df) == 0:
                raise DataIntegrityError(f"yfinance devolvió 0 filas para {ticker}/{interval}")
            df = _flatten_columns(df, ticker)
            missing = [c for c in OHLCV if c not in df.columns]
            if missing:
                raise DataIntegrityError(f"{ticker}/{interval}: faltan columnas {missing}")
            df = df[OHLCV].copy()
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            try:
                df.to_parquet(path)
            except Exception:
                pass
            log.debug(f"descargado {ticker}/{interval}: {len(df)} filas")
            return df
        except Exception as e:                            # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise DataIntegrityError(f"No se pudo descargar {ticker}/{interval}: {last_err}")


def _default_period(interval: str) -> str:
    return {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
            "30m": "60d", "60m": "730d", "1h": "730d"}.get(interval, "max")


# ---------------------------------------------------------------------------
# Validación de integridad
# ---------------------------------------------------------------------------
def validate_ohlcv(df: pd.DataFrame, name: str, min_rows: int = 100,
                   verbose: int = 1) -> dict:
    """Invariantes duras. Devuelve informe; lanza si algo es irrecuperable."""
    log = get_logger(verbose)
    report: dict = {"name": name, "rows": len(df), "warnings": []}

    if len(df) < min_rows:
        raise DataIntegrityError(f"{name}: solo {len(df)} filas, se requieren {min_rows}.")
    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError(f"{name}: índice temporal no ordenado.")
    if df.index.duplicated().any():
        raise DataIntegrityError(f"{name}: hay marcas temporales duplicadas.")

    px = df[["Open", "High", "Low", "Close"]]
    if (px <= 0).any().any():
        n = int((px <= 0).sum().sum())
        raise DataIntegrityError(f"{name}: {n} precios <= 0.")

    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl:
        raise DataIntegrityError(f"{name}: {bad_hl} barras con High < Low.")

    bad_env = ((df["Close"] > df["High"] * 1.0001) |
               (df["Close"] < df["Low"] * 0.9999) |
               (df["Open"] > df["High"] * 1.0001) |
               (df["Open"] < df["Low"] * 0.9999)).sum()
    if bad_env:
        report["warnings"].append(f"{bad_env} barras con O/C fuera del rango H/L")

    r = pct_change_safe(df["Close"]).abs()
    n_jump = int((r > 0.5).sum())
    if n_jump:
        report["warnings"].append(
            f"{n_jump} saltos > 50% en Close (¿split/dividendo no ajustado?)")
    report["max_abs_return"] = float(r.max()) if len(r.dropna()) else np.nan

    nan_frac = df.isna().mean().max()
    report["max_nan_frac"] = float(nan_frac)
    if nan_frac > 0.05:
        report["warnings"].append(f"columna con {100 * nan_frac:.1f}% de NaN")

    zero_vol = float((df["Volume"] <= 0).mean())
    report["zero_volume_frac"] = zero_vol
    if zero_vol > 0.10:
        report["warnings"].append(f"{100 * zero_vol:.1f}% de barras sin volumen")

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 20:
        gaps = df.index.to_series().diff().dt.days.dropna()
        big = int((gaps > 7).sum())
        if big:
            report["warnings"].append(f"{big} huecos > 7 días naturales en el calendario")

    for w in report["warnings"]:
        log.warning(f"[{name}] {w}")
    return report


# ---------------------------------------------------------------------------
# Ancla intradía
# ---------------------------------------------------------------------------
@dataclass
class AnchorResult:
    price: pd.Series             # indexado por fecha (naive, normalizado a día)
    source: str                  # '30m' | '1h' | 'close_proxy'
    coverage: float              # fracción de días diarios con ancla real
    residual_stats: dict         # estadísticas de Close(t)/anchor(t)-1
    intraday_volume: pd.Series | None = None


def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convierte a hora de Nueva York; asume UTC si viene naive."""
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("America/New_York")


def build_anchor(daily: pd.DataFrame, ticker: str, cfg: Config,
                 verbose: int = 1) -> AnchorResult:
    """Construye la serie de precios a T-offset del cierre.

    Estrategia: intenta 30m, luego 1h; el resto de la historia se rellena con
    Close (proxy) y se calibra el residuo sobre el solape.
    """
    log = get_logger(verbose)
    off = cfg.data.anchor_offset_min
    best: tuple[pd.Series, pd.Series, str] | None = None

    for interval in cfg.data.anchor_intervals:
        try:
            intr = download(ticker, interval, cfg.data, verbose=verbose)
        except Exception as e:                            # noqa: BLE001
            log.warning(f"ancla: sin datos {interval} para {ticker} ({e})")
            continue
        try:
            s, vol, sess_close = _extract_anchor_from_intraday(intr, off, interval)
        except Exception as e:                            # noqa: BLE001
            log.warning(f"ancla: fallo extrayendo {interval}: {e}")
            continue
        if len(s) < 15:
            log.warning(f"ancla: {interval} aporta solo {len(s)} días, se descarta")
            continue
        if best is None or len(s) > len(best[0]):
            best = (s, vol, interval, sess_close)

    close = daily["Close"]
    if best is None:
        log.warning(
            "Sin datos intradía: el ancla usará Close(t) como proxy. "
            "El resultado del backtest será OPTIMISTA respecto a operar a T-30'."
        )
        return AnchorResult(price=close.copy(), source="close_proxy", coverage=0.0,
                            residual_stats=_residual_stats(pd.Series(dtype=float)))

    anchor_real, vol_real, source, sess_close = best
    anchor_real = anchor_real[~anchor_real.index.duplicated(keep="last")]
    sess_close = sess_close[~sess_close.index.duplicated(keep="last")]

    # ── REESCALADO A LA ESCALA DE LA SERIE DIARIA ────────────────────────────
    # yfinance sirve el histórico diario ajustado por dividendos y splits, y el
    # intradía con otro criterio: son escalas de precio DISTINTAS. Mezclarlas en
    # y(t) = Close(t+1)/Ancla(t) - 1 mete un sesgo sistemático del tamaño del
    # dividendo acumulado (medido en SPY: 161 bps) y fabrica una correlación
    # falsa con el retorno futuro.
    #
    # El puente es el cierre de la sesión, que ambas series conocen: el factor
    # Close_diario / Close_intradía lleva el ancla a la escala correcta y anula
    # de paso cualquier otra discrepancia de ajuste.
    common0 = close.index.intersection(sess_close.index)
    factor = pd.Series(1.0, index=anchor_real.index)
    if len(common0) >= 10:
        f = (close.loc[common0] / sess_close.loc[common0].where(
            sess_close.loc[common0].abs() > 1e-12))
        f = f.replace([np.inf, -np.inf], np.nan).dropna()
        # un factor sano vive cerca de 1; lejos de ahí hay algo roto
        f = f[(f > 0.5) & (f < 2.0)]
        if len(f) >= 10:
            drift = float((f - 1.0).abs().mean())
            factor = f.reindex(anchor_real.index).ffill().bfill().fillna(1.0)
            if drift > 0.0005:
                log.info(f"ancla reescalada a la serie diaria: desajuste medio "
                         f"{1e4 * drift:.0f} bps (ajuste por dividendos/splits)")
        else:
            log.warning("no se pudo calcular el factor de escala del ancla; "
                        "se usa sin reescalar")
    anchor_real = anchor_real * factor.reindex(anchor_real.index).fillna(1.0)

    # Solape para calibrar
    common = close.index.intersection(anchor_real.index)
    resid = (close.loc[common] / anchor_real.loc[common] - 1.0).dropna()
    stats = _residual_stats(resid)

    # Serie final: real donde exista, proxy en el resto
    merged = close.copy().astype(float)
    merged.loc[common] = anchor_real.loc[common].astype(float)
    coverage = len(common) / max(len(close), 1)

    log.info(
        f"ancla T-{off}': fuente={source}, cobertura={100 * coverage:.1f}% "
        f"({len(common)}/{len(close)} días), "
        f"residuo Close/ancla: mu={1e4 * stats['mean']:.1f}bps "
        f"sigma={1e4 * stats['std']:.1f}bps"
    )

    vol_series = None
    if vol_real is not None:
        vol_series = vol_real.reindex(merged.index)

    return AnchorResult(price=merged, source=source, coverage=coverage,
                        residual_stats=stats, intraday_volume=vol_series)


def _extract_anchor_from_intraday(intr: pd.DataFrame, offset_min: int,
                                  interval: str) -> tuple[pd.Series, pd.Series]:
    """Precio de la sesión regular más cercano a (cierre - offset).

    Para cada día toma la última barra cuyo *inicio* sea <= 16:00 - offset.
    Con barras de 30m eso es la barra 15:30 (su Open ≈ precio a las 15:30).
    Con barras de 1h es la barra 15:00; se usa su Close (=16:00) solo si no
    hay nada mejor, así que preferimos aproximar por interpolación del Open
    de la última barra + fracción, documentado como aproximación.
    """
    df = intr.copy()
    df.index = _to_et(pd.DatetimeIndex(df.index))
    # sesión regular
    df = df.between_time("09:30", "16:00")
    if df.empty:
        raise ValueError("sin barras en sesión regular")

    bar_min = {"30m": 30, "1h": 60, "60m": 60, "15m": 15, "5m": 5}.get(interval, 30)

    minutes = df.index.hour * 60 + df.index.minute
    df = df.assign(_min=minutes, _date=pd.Index(df.index.date))

    # CIERRE REAL DE CADA DÍA, no las 16:00 fijas.
    # En media sesión (Black Friday, Nochebuena) el mercado cierra a las 13:00.
    # Usar 16:00 como referencia hacía que el "ancla" fuese literalmente el
    # cierre de esos días: 30 minutos de información futura colados en la
    # feature más importante del sistema, unas 6 veces al año.
    day_close = df.groupby("_date")["_min"].max()
    df = df.assign(_close_min=df["_date"].map(day_close))
    df = df.assign(_cutoff=df["_close_min"] - offset_min)

    elig = df[df["_min"] <= df["_cutoff"]]
    if elig.empty:
        raise ValueError("ninguna barra dentro del margen previo al cierre")

    last = elig.groupby("_date").tail(1).set_index("_date")

    # Si la barra elegida empieza justo en el corte, su Open ES el precio
    # buscado. Si empieza antes (caso 1h), se interpola dentro de la barra.
    frac = ((last["_cutoff"] - last["_min"]) / bar_min).clip(0.0, 1.0)
    price = last["Open"] * (1 - frac) + last["Close"] * frac

    vol = elig.groupby("_date")["Volume"].sum()   # volumen SOLO hasta el corte
    # cierre de la sesión SEGÚN LAS BARRAS INTRADÍA: sirve de puente para
    # llevar el ancla a la misma escala de precios que la serie diaria
    sess_close = df.groupby("_date")["Close"].last()

    # Días con tan pocas barras que el margen no es fiable (sesiones truncadas
    # por incidencias, o resolución insuficiente): se descartan en vez de
    # devolver un precio contaminado.
    n_bars = elig.groupby("_date").size()
    ok = n_bars[n_bars >= 2].index
    price, vol = price.loc[ok], vol.loc[ok]
    sess_close = sess_close.reindex(ok)

    for x in (price, vol, sess_close):
        x.index = pd.to_datetime(x.index)
    return (price.astype(float).sort_index(), vol.astype(float).sort_index(),
            sess_close.astype(float).sort_index())


def _residual_stats(resid: pd.Series) -> dict:
    if len(resid) < 5:
        return {"n": int(len(resid)), "mean": 0.0, "std": 0.0, "ac1": 0.0,
                "q05": 0.0, "q95": 0.0, "reliable": False}
    return {
        "n": int(len(resid)),
        "mean": float(resid.mean()),
        "std": float(resid.std(ddof=1)),
        "ac1": float(resid.autocorr(1)) if len(resid) > 3 else 0.0,
        "q05": float(resid.quantile(0.05)),
        "q95": float(resid.quantile(0.95)),
        "reliable": len(resid) >= 30,
    }


# ---------------------------------------------------------------------------
# Carga completa
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    ticker: str
    daily: pd.DataFrame
    anchor: AnchorResult
    context: dict[str, pd.DataFrame]
    reports: dict[str, dict]


def load_market_data(cfg: Config, verbose: int | None = None) -> MarketData:
    verbose = cfg.verbose if verbose is None else verbose
    log = get_logger(verbose)
    t = cfg.data.ticker

    daily = download(t, "1d", cfg.data, start=cfg.data.start, end=cfg.data.end,
                     verbose=verbose)
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    rep = {t: validate_ohlcv(daily, t, cfg.data.min_rows, verbose)}

    anchor = build_anchor(daily, t, cfg, verbose)

    ctx: dict[str, pd.DataFrame] = {}
    if cfg.features.cross_asset:
        for c in cfg.data.context_tickers:
            if c.upper() == t.upper():
                continue
            try:
                d = download(c, "1d", cfg.data, start=cfg.data.start,
                             end=cfg.data.end, verbose=verbose)
                d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
                d = d[~d.index.duplicated(keep="last")].sort_index()
                rep[c] = validate_ohlcv(d, c, 100, verbose=0)
                ctx[c] = d
            except Exception as e:                        # noqa: BLE001
                log.warning(f"contexto {c} no disponible: {e}")

    return MarketData(ticker=t, daily=daily, anchor=anchor, context=ctx, reports=rep)
