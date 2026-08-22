"""Captura del precio EN VIVO en el instante del ancla.

Durante la sesión, el histórico diario de yfinance ya trae una fila parcial de
hoy cuyo Close es el último precio negociado. Eso vale como ancla, pero es
frágil: a veces la fila no existe todavía. Aquí se resuelve pidiendo barras de
1 minuto y se deja el diario coherente pase lo que pase.

IMPORTANTE sobre el retardo: yfinance sirve datos con un retardo que depende del
mercado (típicamente inmediato en índices y valores US, hasta 15 minutos en
algunas plazas). Un desfase de minutos mueve el precio de entrada y por tanto la
señal. Si vas a operar de verdad, engancha el feed de tu bróker y pasa el precio
con `live_anchor_price`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import Config, DataConfig
from .data import MarketData, download
from .utils import get_logger

ET = "America/New_York"


class StaleQuoteError(RuntimeError):
    """El precio disponible no es de la sesión de hoy."""


@dataclass
class LiveQuote:
    ticker: str
    price: float
    timestamp: pd.Timestamp          # en hora de Nueva York
    source: str                      # '1m' | '5m' | '15m' | 'daily_close'
    minutes_to_close: float
    degraded: bool                   # True si no es un precio realmente intradía
    session_open: float | None = None
    session_volume: float | None = None

    @property
    def age_minutes(self) -> float:
        return (pd.Timestamp.now(tz=ET) - self.timestamp).total_seconds() / 60.0

    def check_fresh(self, max_age_min: float = 90.0) -> None:
        """Aborta si el precio no es de hoy o llega demasiado viejo.

        Sin esto, un fallo del proveedor hacía que se publicase la predicción de
        ayer con la fecha de hoy: el error más caro posible, porque no se nota.
        """
        today = pd.Timestamp.now(tz=ET).date()
        if self.timestamp.date() != today:
            raise StaleQuoteError(
                f"{self.ticker}: el último precio disponible es del "
                f"{self.timestamp.date()}, no de hoy ({today}). No se publica "
                f"una señal con datos rancios.")
        if self.age_minutes > max_age_min:
            raise StaleQuoteError(
                f"{self.ticker}: precio de hace {self.age_minutes:.0f} min "
                f"({self.timestamp:%H:%M} ET); el límite son {max_age_min:.0f}.")


def market_clock(now: datetime | None = None) -> dict:
    """Reloj del mercado americano en hora local de Nueva York.

    Resuelve solo el horario de verano; los festivos se detectan por ausencia de
    datos, no por calendario codificado a mano (que envejece mal).
    """
    now_et = (pd.Timestamp.now(tz=ET) if now is None
              else pd.Timestamp(now).tz_convert(ET))
    close = now_et.normalize() + pd.Timedelta(hours=16)
    open_ = now_et.normalize() + pd.Timedelta(hours=9, minutes=30)
    return {
        "now_et": now_et,
        "open_et": open_,
        "close_et": close,
        "minutes_to_close": (close - now_et).total_seconds() / 60.0,
        "is_weekday": now_et.dayofweek < 5,
        "in_session": open_ <= now_et <= close and now_et.dayofweek < 5,
        "utc_offset_hours": -now_et.utcoffset().total_seconds() / 3600.0,
    }


def get_live_quote(ticker: str, cfg: DataConfig, verbose: int = 1) -> LiveQuote:
    """Último precio negociado, probando resoluciones de más a menos fina."""
    log = get_logger(verbose)
    clock = market_clock()
    now_et = clock["now_et"]

    for interval in ("1m", "5m", "15m"):
        try:
            df = download(ticker, interval, cfg, use_cache=False, verbose=0)
        except Exception as e:                       # noqa: BLE001
            log.debug(f"{ticker}: sin barras {interval} ({e})")
            continue
        try:
            idx = pd.DatetimeIndex(df.index)
            idx = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(ET)
            df = df.set_axis(idx)
            today = df[df.index.date == now_et.date()]
            today = today.between_time("09:30", "16:00")
            if today.empty:
                continue
            last = today.iloc[-1]
            ts = today.index[-1]
            if (now_et - ts) > pd.Timedelta(minutes=45):
                log.warning(f"{ticker}: la última barra {interval} es de "
                            f"{ts:%H:%M} ET, {(now_et - ts).seconds // 60} min atrás")
            return LiveQuote(
                ticker=ticker, price=float(last["Close"]), timestamp=ts,
                source=interval, minutes_to_close=clock["minutes_to_close"],
                degraded=False, session_open=float(today.iloc[0]["Open"]),
                session_volume=float(today["Volume"].sum()))
        except Exception as e:                       # noqa: BLE001
            log.debug(f"{ticker}: fallo procesando {interval} ({e})")
            continue

    # Último recurso: la fila diaria parcial
    d = download(ticker, "1d", cfg, use_cache=False, verbose=0)
    last = d.iloc[-1]
    return LiveQuote(ticker=ticker, price=float(last["Close"]),
                     timestamp=pd.Timestamp(d.index[-1]).tz_localize(ET),
                     source="daily_close",
                     minutes_to_close=clock["minutes_to_close"], degraded=True,
                     session_open=float(last["Open"]),
                     session_volume=float(last["Volume"]))


def sanity_check_quote(quote: LiveQuote, prev_close: float,
                       max_move: float = 0.35) -> None:
    """Rechaza precios imposibles antes de que entren en el modelo.

    Los proveedores gratuitos devuelven de vez en cuando basura: un cero, un
    precio sin ajustar por split, un tick erróneo. Cualquiera de esas cosas
    genera un `tdy_anchor_ret` monstruoso y una señal con toda la apariencia de
    ser correcta. Un salto mayor del 35% en un día es casi siempre un error de
    datos, no mercado.
    """
    p, c = float(quote.price), float(prev_close)
    if not np.isfinite(p) or p <= 0:
        raise StaleQuoteError(f"{quote.ticker}: precio inválido ({p})")
    if not np.isfinite(c) or c <= 0:
        return
    move = p / c - 1
    if abs(move) > max_move:
        raise StaleQuoteError(
            f"{quote.ticker}: el precio {p:.4f} implica un {100 * move:+.1f}% "
            f"frente al cierre anterior ({c:.4f}). Casi seguro es un error de "
            f"datos o un split sin ajustar; no se opera con eso.")


def ensure_today_row(md: MarketData, quote: LiveQuote, verbose: int = 1) -> MarketData:
    """Garantiza que el histórico diario tiene la fila de HOY y que el ancla
    apunta al precio en vivo.

    Sin esto, el modelo predeciría a partir del último día cerrado y la señal
    llegaría con 24 horas de retraso, que es justo lo que queríamos evitar.
    """
    log = get_logger(verbose)
    today = pd.Timestamp(quote.timestamp.date())
    d = md.daily
    if len(d) and today < d.index[-1]:
        raise ValueError(
            f"{md.ticker}: la cotización es del {today.date()} pero el histórico "
            f"llega al {d.index[-1].date()}; se estaría reescribiendo el pasado.")

    p = float(quote.price)
    # Si no conocemos la apertura real, se deja NaN en vez de inventarla con el
    # cierre anterior: eso fabricaba un gap de exactamente 0% que el modelo se
    # creía. Un NaN lo gestiona el imputador y, sobre todo, se ve.
    o = float(quote.session_open) if quote.session_open else np.nan
    if not np.isfinite(o):
        log.warning(f"{md.ticker}: apertura de hoy desconocida; el gap queda sin dato")
    row = pd.DataFrame({
        "Open": [o],
        # High/Low de hoy NO se usan en las features (serían fuga); se rellenan
        # de forma consistente solo para no romper las validaciones.
        "High": [max(o, p) if np.isfinite(o) else p],
        "Low": [min(o, p) if np.isfinite(o) else p], "Close": [p],
        "Volume": [quote.session_volume or float(d["Volume"].iloc[-21:].mean())],
    }, index=[today])

    if today in d.index:
        d.loc[today, ["Open", "High", "Low", "Close", "Volume"]] = row.iloc[0].to_numpy()
        log.debug(f"{md.ticker}: fila de hoy actualizada con el precio en vivo")
    else:
        d = pd.concat([d, row]).sort_index()
        log.info(f"{md.ticker}: añadida la fila de hoy ({today.date()}) al histórico")
    md.daily = d

    a = md.anchor.price.reindex(d.index)
    a.loc[today] = p
    md.anchor.price = a
    return md


def snapshot(cfg: Config, verbose: int | None = None
             ) -> tuple[MarketData, LiveQuote]:
    """Carga el histórico y lo empalma con el precio del momento."""
    from .data import load_market_data
    verbose = cfg.verbose if verbose is None else verbose
    md = load_market_data(cfg, verbose=verbose)
    q = get_live_quote(cfg.data.ticker, cfg.data, verbose=verbose)
    md = ensure_today_row(md, q, verbose=verbose)
    return md, q
