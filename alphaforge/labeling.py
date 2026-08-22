"""Etiquetado del objetivo y pesos de muestra.

El objetivo se mide SIEMPRE desde el precio del ancla, que es el precio al que
realmente podríamos ejecutar:

    y(t) = P_salida(t+h) / P_anchor(t) - 1

Con horizon='close_next' la operación es: compro a las 15:30 ET del día t,
vendo al cierre del día t+1. Ese retorno incluye los últimos 30' de hoy, la
sesión nocturna (el gap) y toda la sesión de mañana. Es exactamente lo que
queremos capturar y lo que se pierde si esperas al cierre para decidir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketData
from .utils import get_logger, pct_change_safe


@dataclass
class Labels:
    y_ret: pd.Series          # retorno bruto desde el ancla
    y_ret_norm: pd.Series     # retorno normalizado por volatilidad ex-ante
    y_cls: pd.Series          # 1 = sube (por encima del umbral), 0 = no
    mask_valid: pd.Series     # True si la etiqueta es utilizable
    sigma_ex_ante: pd.Series  # volatilidad esperada del horizonte, conocida en t
    threshold: pd.Series      # umbral efectivo aplicado
    exit_price: pd.Series
    entry_price: pd.Series


def build_labels(md: MarketData, cfg: Config) -> Labels:
    log = get_logger(cfg.verbose)
    daily, lc = md.daily, cfg.label
    entry = md.anchor.price.reindex(daily.index).astype(float)

    if lc.horizon == "close_next":
        exit_px = daily["Close"].shift(-1)
    elif lc.horizon == "open_next":
        exit_px = daily["Open"].shift(-1)
    elif lc.horizon == "close_next2":
        exit_px = daily["Close"].shift(-2)
    else:
        raise ValueError(f"horizonte no soportado: {lc.horizon}")

    y_ret = exit_px / entry.where(entry.abs() > 1e-12) - 1.0
    y_ret = y_ret.replace([np.inf, -np.inf], np.nan)

    # --- volatilidad ex-ante: conocida en t (usa datos hasta t-1) -------------
    r_daily = pct_change_safe(daily["Close"])
    sigma = r_daily.rolling(21, min_periods=10).std(ddof=0).shift(1)
    sigma = sigma.clip(lower=1e-4) * np.sqrt(max(1, cfg.horizon_days))
    sigma = sigma.ffill()

    # --- umbral de clasificación ---------------------------------------------
    if lc.threshold_mode == "zero":
        thr = pd.Series(0.0, index=daily.index)
    elif lc.threshold_mode == "cost":
        thr = pd.Series(cfg.roundtrip_cost, index=daily.index)
    elif lc.threshold_mode == "vol":
        thr = lc.neutral_band_k * sigma
    else:
        raise ValueError(f"threshold_mode no soportado: {lc.threshold_mode}")

    y_cls = (y_ret > thr).astype(float)
    y_norm = (y_ret / sigma).clip(-8, 8) if lc.vol_normalize_target else y_ret

    valid = y_ret.notna() & entry.notna() & sigma.notna()

    # --- huecos de calendario ------------------------------------------------
    # y(t) se mide contra la SIGUIENTE FILA del índice. Si el valor estuvo
    # suspendido, o el proveedor tiene agujeros, esa "siguiente fila" puede
    # estar a semanas: la etiqueta abarcaría un horizonte muy distinto del que
    # el modelo cree estar aprendiendo. Se invalidan.
    h = max(1, cfg.horizon_days)
    gap = daily.index.to_series().diff().shift(-h).dt.days
    max_gap = lc.max_calendar_gap_days
    stale = gap > max_gap
    n_stale = int((stale & valid).sum())
    if n_stale:
        log.warning(f"{n_stale} etiquetas descartadas: el precio de salida está a "
                    f"más de {max_gap} días naturales (huecos de cotización)")
        valid &= ~stale.fillna(True)
    # Filtro de outliers absurdos (errores de datos, no mercado)
    extreme = y_ret.abs() > 0.60
    if extreme.sum():
        log.warning(f"{int(extreme.sum())} etiquetas con |y| > 60% marcadas como inválidas")
        valid &= ~extreme

    bal = float(y_cls[valid].mean()) if valid.any() else np.nan
    log.info(f"etiquetas: {int(valid.sum())} válidas | clase positiva "
             f"{100 * bal:.1f}% | umbral medio {1e4 * float(thr.mean()):.1f}bps")

    return Labels(y_ret=y_ret, y_ret_norm=y_norm, y_cls=y_cls, mask_valid=valid,
                  sigma_ex_ante=sigma, threshold=thr,
                  exit_price=exit_px, entry_price=entry)


def sample_weights(index: pd.DatetimeIndex, horizon: int, cfg: Config,
                   y_ret: pd.Series | None = None) -> pd.Series:
    """Pesos = unicidad temporal x decaimiento exponencial x magnitud.

    - Unicidad: con horizonte h, h etiquetas consecutivas comparten información
      del mismo tramo de mercado. Reducimos su peso a 1/h (López de Prado).
    - Decaimiento: el pasado lejano informa menos sobre el régimen actual.
    - Magnitud: los movimientos grandes son la señal que importa; el ruido
      alrededor de cero pesa menos.
    """
    n = len(index)
    uniqueness = np.full(n, 1.0 / max(1, horizon))

    age_days = (index[-1] - index).days.to_numpy().astype(float)
    hl = max(1.0, cfg.label.time_decay_halflife_days)
    decay = np.exp(-np.log(2.0) * age_days / hl)

    w = uniqueness * decay
    if y_ret is not None:
        mag = y_ret.abs().to_numpy()
        med = np.nanmedian(mag[np.isfinite(mag)]) if np.isfinite(mag).any() else 1.0
        mag_w = np.clip(mag / max(med, 1e-8), 0.25, 4.0)
        mag_w = np.where(np.isfinite(mag_w), mag_w, 1.0)
        w = w * mag_w

    w = w / max(w.mean(), 1e-12)
    return pd.Series(w, index=index)


def align(X: pd.DataFrame, labels: Labels, cfg: Config
          ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Interseca features y etiquetas, descarta filas inservibles."""
    log = get_logger(cfg.verbose)
    idx = X.index.intersection(labels.y_ret.index)
    mask = labels.mask_valid.reindex(idx).fillna(False).to_numpy()

    Xa = X.loc[idx][mask]
    # Requerimos que al menos el 70% de las features estén presentes
    row_cov = Xa.notna().mean(axis=1)
    ok = row_cov >= 0.70
    n_drop = int((~ok).sum())
    Xa = Xa[ok]

    keep_idx = Xa.index
    y_ret = labels.y_ret.reindex(keep_idx)
    y_cls = labels.y_cls.reindex(keep_idx)
    y_norm = labels.y_ret_norm.reindex(keep_idx)
    sigma = labels.sigma_ex_ante.reindex(keep_idx)

    log.info(f"alineado: {len(Xa)} muestras x {Xa.shape[1]} features "
             f"({n_drop} filas descartadas por cobertura) | "
             f"{keep_idx.min().date()} -> {keep_idx.max().date()}")
    if len(Xa) < cfg.validation.min_train_days + 100:
        raise ValueError(
            f"Solo {len(Xa)} muestras utilizables; se necesitan al menos "
            f"{cfg.validation.min_train_days + 100}. Amplía el rango de fechas.")
    return Xa, y_cls, y_ret, y_norm, sigma
