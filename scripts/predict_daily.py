#!/usr/bin/env python3
"""Barrido diario: captura el precio a T−60′ del cierre y publica las señales.

Se ejecuta desde GitHub Actions una hora antes del cierre americano. Escribe
docs/data/latest.json y añade el día a docs/data/history.json, que es lo que
lee el tablón web.

    python scripts/predict_daily.py --universe universe.json --models models/ \
                                    --out docs/data --wait
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from alphaforge.config import Config
from alphaforge.features import build_features
from alphaforge.labeling import build_labels
from alphaforge.live import (LiveQuote, StaleQuoteError, ensure_today_row,
                             get_live_quote, market_clock, sanity_check_quote)
from alphaforge.pipeline import _apply_calibrator, _proba
from alphaforge.backtest import position_from_prob
from alphaforge.data import load_market_data
from alphaforge.report import _jsonable
from alphaforge.utils import get_logger, suppress_noisy_warnings

MAX_HISTORY_DAYS = 400


# ---------------------------------------------------------------------------
def wait_for_anchor(offset_min: int, wait_max_min: float = 90.0,
                    verbose: int = 1) -> dict:
    """Espera hasta que falten `offset_min` minutos para el cierre.

    Hay dos formas de equivocarse aquí y antes solo se vigilaba una:

    * LLEGAR TARDE: el cron de GitHub arranca con retraso variable (1-20 min).
    * LLEGAR PRONTO: lanzarlo a mano a media tarde. Este era el peligroso,
      porque no avisaba de nada. El modelo aprendió a decidir con el precio de
      T−60; darle el de T−120 cambia la feature más importante del sistema
      (cuánto se ha movido el valor en lo que va de sesión) y la señal sale con
      aspecto perfectamente normal.

    Ahora se espera de verdad hasta el instante correcto, y si la espera no cabe
    en el margen disponible, la desviación se registra y viaja hasta el tablón.
    """
    log = get_logger(verbose)
    clock = market_clock()
    delta = clock["minutes_to_close"] - offset_min

    if delta > 0.5:
        secs = min(delta * 60, wait_max_min * 60)
        log.info(f"son las {clock['now_et']:%H:%M} ET y faltan "
                 f"{clock['minutes_to_close']:.0f} min para el cierre; "
                 f"esperando {secs / 60:.0f} min hasta el ancla (T−{offset_min}′)")
        time.sleep(secs)
        clock = market_clock()

    actual = clock["minutes_to_close"]
    mismatch = actual - offset_min
    clock["offset_actual_min"] = round(float(actual), 1)
    clock["mismatch_min"] = round(float(mismatch), 1)
    clock["late"] = mismatch < -10
    clock["early"] = mismatch > 10

    if clock["late"]:
        log.warning(f"ancla TARDÍA: faltan {actual:.0f} min para el cierre "
                    f"en vez de {offset_min}")
    elif clock["early"]:
        log.warning(
            f"ancla PREMATURA: capturada a T−{actual:.0f}′ cuando el modelo se "
            f"entrenó con T−{offset_min}′ ({mismatch:+.0f} min de desviación). "
            f"La señal es orientativa: el precio de referencia no es el que el "
            f"modelo espera.")
    else:
        log.info(f"ancla en hora: T−{actual:.0f}′")
    return clock


def pick_variant(models_dir: str, ticker: str, variants: list[str],
                 site_dir: str, log) -> str | None:
    """De todas las variantes entrenadas para un valor, coge la que ganó.

    Si el backtest ya nombró ganadora, se respeta. Si no, se prefiere el mejor
    veredicto y, en empate, la variante por defecto.
    """
    import json as _json
    path = os.path.join(site_dir, "backtest.json")
    disponibles = []
    for v in variants:
        f = os.path.join(models_dir, _model_file(ticker, v))
        if os.path.exists(f):
            disponibles.append(v)
    if not disponibles:
        return None
    if len(disponibles) == 1:
        return disponibles[0]
    try:
        with open(path) as f:
            models = _json.load(f).get("models", {})
        for v in disponibles:
            key = ticker if v == "close" else f"{ticker}::{v}"
            if models.get(key, {}).get("variant_winner"):
                return v
        rank = {"GO": 2, "REVISAR": 1, "NO-GO": 0}
        disponibles.sort(key=lambda v: -rank.get(
            models.get(ticker if v == "close" else f"{ticker}::{v}", {})
            .get("verdict", "NO-GO"), 0))
    except Exception as e:                                 # noqa: BLE001
        log.debug(f"{ticker}: sin comparación de variantes ({e})")
    return disponibles[0]


def _model_file(ticker: str, variant: str | None) -> str:
    t = ticker.replace(".", "_")
    return f"{t}.pkl" if not variant or variant == "close" else f"{t}__{variant}.pkl"


def load_model(models_dir: str, ticker: str, variant: str | None = None) -> dict | None:
    """Carga un modelo y rechaza los de formato antiguo.

    Un .pkl de una versión anterior no falla al abrirse: falla más tarde, de
    formas raras y silenciosas, en mitad de una predicción. Mejor abortar aquí.
    """
    from alphaforge import ARTIFACT_FORMAT
    p = os.path.join(models_dir, _model_file(ticker, variant))
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        blob = pickle.load(f)
    fmt = blob.get("format", 0)
    if fmt != ARTIFACT_FORMAT:
        raise RuntimeError(
            f"{ticker}: modelo en formato {fmt}, se requiere {ARTIFACT_FORMAT}. "
            f"Vuelve a ejecutar el entrenamiento.")
    return blob


# ---------------------------------------------------------------------------
def predict_one(ticker: str, blob: dict, verbose: int = 1) -> dict:
    """Señal de un valor usando el modelo ya entrenado y el precio del momento."""
    """Señal de un valor usando el modelo ya entrenado y el precio del momento."""
    cfg = Config.from_dict(blob["cfg"])
    cfg.data.ticker = ticker
    cfg.verbose = verbose

    quote = get_live_quote(ticker, cfg.data, verbose=verbose)
    quote.check_fresh()                       # nada de señales con precios rancios
    md = load_market_data(cfg, verbose=0)
    prev = md.daily["Close"]
    prev = prev[prev.index < pd.Timestamp(quote.timestamp.date())]
    if len(prev):
        sanity_check_quote(quote, float(prev.iloc[-1]))
    md = ensure_today_row(md, quote, verbose=verbose)

    X = build_features(md, cfg)
    labels = build_labels(md, cfg)
    art = blob["final"]["artifacts"]

    row = X.iloc[[-1]]
    missing = [f for f in art.features if f not in row.columns]
    if missing:
        raise RuntimeError(f"faltan {len(missing)} features en el snapshot "
                           f"(p.ej. {missing[:3]}); reentrena el modelo")

    Z = art.imputer.transform(row[art.features])
    probs = [_apply_calibrator(art.calibrators[f], _proba(art.models[f], Z))
             for f in art.families]
    p_up = float(np.clip(np.column_stack(probs) @ art.weights, 1e-6, 1 - 1e-6)[0])

    sigma = float(labels.sigma_ex_ante.iloc[-1])
    q = {}
    if art.quantiles is not None:
        qq = art.quantiles.predict(pd.DataFrame(Z, columns=art.features,
                                                index=row.index))
        q = {k: float(v.iloc[0]) * sigma for k, v in qq.items()}

    pos = float(position_from_prob(
        pd.Series([p_up], index=row.index), cfg,
        expected_ret=pd.Series([q.get("q50", 0.0)], index=row.index),
        sigma=pd.Series([sigma], index=row.index)).iloc[0])

    from alphaforge.pipeline import snapshot_health
    health = snapshot_health(art, row)
    m = blob.get("metrics", {})
    return {
        "ticker": ticker,
        "variant": blob.get("variant", "close"),
        "health": health,
        "as_of": str(row.index[-1].date()),
        "anchor_price": round(float(quote.price), 4),
        "anchor_time_et": quote.timestamp.strftime("%Y-%m-%d %H:%M"),
        "anchor_source": quote.source,
        "anchor_degraded": bool(quote.degraded),
        "minutes_to_close": round(float(quote.minutes_to_close), 1),
        "prob_up": round(p_up, 4),
        "expected_return": _r(q.get("q50")),
        "ret_p10": _r(q.get("q10")),
        "ret_p90": _r(q.get("q90")),
        "sigma_ex_ante": _r(sigma),
        "position": round(pos, 3),
        "direction": "LARGO" if pos > 0 else ("CORTO" if pos < 0 else "FUERA"),
        "horizon": cfg.label.horizon,
        "verdict": blob["verdict"]["decision"],
        "operable": blob["verdict"]["decision"] == "GO",
        "model": {
            "sharpe": _r(m.get("sharpe"), 3), "auc": _r(m.get("auc"), 4),
            "auc_oos": _r(m.get("auc_oos"), 4), "max_dd": _r(m.get("max_dd")),
            "deflated_sharpe": _r(m.get("deflated_sharpe"), 3),
            "trained_days_ago": None,
        },
    }


def _r(x, nd: int = 5):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--wait", action="store_true",
                    help="esperar hasta el instante exacto del ancla")
    ap.add_argument("--wait-max", type=float, default=90.0,
                    help="minutos máximos de espera hasta el ancla (90 por defecto)")
    ap.add_argument("--force", action="store_true",
                    help="ejecutar aunque el mercado esté cerrado (pruebas)")
    ap.add_argument("--offset", type=int, default=None,
                    help="minutos antes del cierre (por defecto, el del universo)")
    args = ap.parse_args()

    suppress_noisy_warnings()
    log = get_logger(1)
    with open(args.universe) as f:
        uni = json.load(f)
    offset = args.offset or uni.get("config", {}).get("data", {}).get(
        "anchor_offset_min", 60)

    clock = market_clock()
    if not clock["in_session"] and not args.force:
        log.error(f"mercado cerrado ({clock['now_et']:%Y-%m-%d %H:%M} ET). "
                  f"Usa --force para forzar la ejecución.")
        return 78                                    # código neutro: no es un fallo
    if args.wait:
        clock = wait_for_anchor(offset, wait_max_min=args.wait_max, verbose=1)
    else:
        m = clock["minutes_to_close"] - offset
        clock["offset_actual_min"] = round(float(clock["minutes_to_close"]), 1)
        clock["mismatch_min"] = round(float(m), 1)
        clock["late"], clock["early"] = m < -10, m > 10
        if clock["late"] or clock["early"]:
            log.warning(f"el ancla se capturará a T−{clock['minutes_to_close']:.0f}′ "
                        f"y el modelo espera T−{offset}′ ({m:+.0f} min). Usa --wait "
                        f"para esperar al momento correcto.")

    os.makedirs(args.out, exist_ok=True)
    signals, errors = [], []
    for t in uni["tickers"]:
        try:
            v = pick_variant(args.models, t, list(uni.get("variants", {"close": {}})),
                             args.out, log)
            blob = load_model(args.models, t, v)
        except RuntimeError as e:
            errors.append({"ticker": t, "error": "modelo obsoleto"})
            log.error(str(e))
            continue
        if blob is None:
            errors.append({"ticker": t, "error": "sin modelo entrenado"})
            log.warning(f"{t}: no hay modelo; ejecuta scripts/train_all.py")
            continue
        try:
            s = predict_one(t, blob, verbose=0)
            signals.append(s)
            warn = "  ! " + "; ".join(s["health"]["warnings"]) if \
                s["health"].get("warnings") else ""
            log.info(f"{t:8s}[{s.get('variant', 'close'):5s}] "
                     f"P(sube)={100 * s['prob_up']:5.1f}%  "
                     f"E[ret]={100 * (s['expected_return'] or 0):+.2f}%  "
                     f"{s['direction']:6s} [{s['verdict']}]{warn}")
        except StaleQuoteError as e:
            log.error(f"{t}: {e}")
            errors.append({"ticker": t, "error": "precio no actualizado"})
        except Exception as e:                       # noqa: BLE001
            log.error(f"{t}: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"ticker": t, "error": f"{type(e).__name__}: {e}"})

    if not signals:
        log.error("ninguna señal generada")
        return 1

    signals.sort(key=lambda s: -s["prob_up"])
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "anchor_et": clock["now_et"].strftime("%Y-%m-%d %H:%M"),
        "minutes_to_close": round(float(clock["minutes_to_close"]), 1),
        "offset_target_min": offset,
        "offset_actual_min": clock.get("offset_actual_min"),
        "mismatch_min": clock.get("mismatch_min"),
        "late": bool(clock.get("late", False)),
        "early": bool(clock.get("early", False)),
        "anchor_ok": not (clock.get("late") or clock.get("early")),
        "horizon_note": "Entrada al precio del ancla; salida al cierre de la sesión siguiente.",
        "signals": signals,
        "errors": errors,
        "n_operable": sum(1 for s in signals if s["operable"]),
    }

    with open(os.path.join(args.out, "latest.json"), "w") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)

    _append_history(args.out, payload, log)
    log.info(f"publicadas {len(signals)} señales "
             f"({payload['n_operable']} operables) en {args.out}/latest.json")
    return 0


def _append_history(out_dir: str, payload: dict, log) -> None:
    """Guarda cada día para poder medir después el acierto REAL en vivo.

    Un backtest dice lo que habría pasado; esto dice lo que pasa. Es el único
    número que no admite discusión.
    """
    path = os.path.join(out_dir, "history.json")
    hist = {"entries": []}
    if os.path.exists(path):
        try:
            with open(path) as f:
                hist = json.load(f)
        except Exception as e:                       # noqa: BLE001
            log.warning(f"history.json ilegible, se empieza de cero ({e})")

    day = payload["anchor_et"][:10]
    hist["entries"] = [e for e in hist.get("entries", []) if e.get("date") != day]
    hist["entries"].append({
        "date": day,
        "anchor_et": payload["anchor_et"],
        "signals": [{k: s[k] for k in
                     ("ticker", "prob_up", "expected_return", "ret_p10", "ret_p90",
                      "position", "anchor_price", "verdict", "operable")}
                    for s in payload["signals"]],
        "resolved": False,
    })
    hist["entries"] = sorted(hist["entries"], key=lambda e: e["date"])[-MAX_HISTORY_DAYS:]
    hist["updated_utc"] = payload["generated_utc"]
    with open(path, "w") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
