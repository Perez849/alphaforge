#!/usr/bin/env python3
"""Entrena y valida un modelo por valor. Se ejecuta SEMANALMENTE, no a diario.

Entrenar es caro (minutos por valor) y el mercado no cambia de régimen cada 24
horas. El robot diario solo carga estos modelos y predice, que tarda segundos.

    python scripts/train_all.py --universe universe.json --out models/
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

from alphaforge import ARTIFACT_FORMAT, __version__
from alphaforge.config import Config
from alphaforge.pipeline import run_experiment
from alphaforge.report import _jsonable, save_artifacts
from alphaforge.site import oos_summary
from alphaforge.utils import get_logger, suppress_noisy_warnings


def load_universe(path: str) -> dict:
    with open(path) as f:
        u = json.load(f)
    if not u.get("tickers"):
        raise ValueError(f"{path} no contiene la lista 'tickers'")
    return u


def build_config(uni: dict, ticker: str) -> Config:
    cfg = Config.from_dict(uni.get("config", {}))
    cfg.data.ticker = ticker
    for k, v in (uni.get("overrides", {}).get(ticker, {})).items():
        section, _, field = k.partition(".")
        if field and hasattr(cfg, section):
            setattr(getattr(cfg, section), field, v)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--only", help="entrenar solo estos tickers (coma)")
    ap.add_argument("--skip-fresh-days", type=int, default=5,
                    help="no reentrenar si el modelo tiene menos de N días")
    ap.add_argument("--site-data", default="docs/data",
                    help="dónde publicar el resumen de fiabilidad para la web")
    args = ap.parse_args()

    suppress_noisy_warnings()
    log = get_logger(1)
    uni = load_universe(args.universe)
    tickers = ([t.strip().upper() for t in args.only.split(",")]
               if args.only else uni["tickers"])
    os.makedirs(args.out, exist_ok=True)

    index, failures, summaries = [], [], {}
    # se conserva lo ya publicado para los tickers que hoy no se reentrenan
    site_path = os.path.join(args.site_data, "backtest.json")
    if os.path.exists(site_path):
        try:
            with open(site_path) as f:
                summaries = json.load(f).get("models", {})
        except Exception as e:                       # noqa: BLE001
            log.warning(f"backtest.json previo ilegible ({e})")
    for i, t in enumerate(tickers, 1):
        path = os.path.join(args.out, f"{t.replace('.', '_')}.pkl")
        if os.path.exists(path) and args.skip_fresh_days > 0:
            age = (time.time() - os.path.getmtime(path)) / 86400
            if age < args.skip_fresh_days:
                log.info(f"[{i}/{len(tickers)}] {t}: modelo de hace {age:.1f}d, se omite")
                continue

        log.info(f"[{i}/{len(tickers)}] entrenando {t}...")
        t0 = time.time()
        try:
            cfg = build_config(uni, t)
            res = run_experiment(cfg)
            with open(path, "wb") as f:
                pickle.dump({"format": ARTIFACT_FORMAT, "version": __version__,
                             "cfg": cfg.to_dict(), "final": res.final_model,
                             "verdict": res.verdict,
                             "metrics": _jsonable(res.metrics)}, f)
            if uni.get("save_reports", True):
                save_artifacts(res, os.path.join(args.out, "reports"))
            summaries[t] = oos_summary(res)
            entry = {
                "ticker": t,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "verdict": res.verdict["decision"],
                "sharpe": res.metrics.get("sharpe"),
                "auc": res.metrics.get("auc_oos"),
                "pbo": res.pbo.pbo if res.pbo else None,
                "max_dd": res.metrics.get("max_dd"),
                "n_trades": res.metrics.get("n_active"),
                "anchor_source": res.metrics.get("anchor_source"),
                "anchor_coverage": res.metrics.get("anchor_coverage"),
                "seconds": round(time.time() - t0, 1),
            }
            index.append(_jsonable(entry))
            log.info(f"    {t}: {entry['verdict']} | Sharpe "
                     f"{entry['sharpe']:.2f} | {entry['seconds']:.0f}s"
                     if entry["sharpe"] else f"    {t}: {entry['verdict']}")
        except Exception as e:                       # noqa: BLE001
            log.error(f"    {t} FALLÓ: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append({"ticker": t, "error": f"{type(e).__name__}: {e}"})

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": index, "failures": failures,
        "n_go": sum(1 for m in index if m["verdict"] == "GO"),
    }
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Resumen de fiabilidad: es lo que llena la web desde el primer día, sin
    # esperar a acumular señales en vivo.
    if summaries:
        os.makedirs(args.site_data, exist_ok=True)
        with open(site_path, "w") as f:
            json.dump({"generated": manifest["generated"],
                       "models": summaries,
                       "portfolio": _portfolio(summaries)},
                      f, indent=2, ensure_ascii=False)
        log.info(f"publicada la fiabilidad de {len(summaries)} modelos en {site_path}")

    log.info(f"listo: {len(index)} modelos entrenados, {len(failures)} fallos, "
             f"{manifest['n_go']} con veredicto GO")
    # Solo se considera fallo total si NINGÚN modelo salió adelante
    return 0 if index else 1


def _portfolio(summaries: dict) -> dict:
    """Agrega los valores en una sola cartera equiponderada.

    Un modelo suelto puede tener suerte; ocho a la vez, menos. Además la
    diversificación es lo que convierte un edge pequeño en algo operable.
    """
    import numpy as np
    daily: dict[str, list] = {}
    for tk, s in summaries.items():
        if s.get("verdict") == "NO-GO":
            continue                                  # no se mete basura en la cartera
        for r in s.get("recent", []):
            if r.get("pnl") is not None:
                daily.setdefault(r["date"], []).append(r["pnl"])
    if not daily:
        return {"n_days": 0, "members": 0}
    dates = sorted(daily)
    rets = np.array([float(np.mean(daily[d])) for d in dates])
    eq, curve = 1.0, []
    for d, r in zip(dates, rets):
        eq *= (1 + r)
        curve.append({"d": d, "e": round(eq, 5)})
    sd = float(rets.std(ddof=1)) if len(rets) > 2 else 0.0
    return {
        "n_days": len(dates),
        "members": sum(1 for s in summaries.values() if s.get("verdict") != "NO-GO"),
        "total_return": round(float(eq - 1), 5),
        "sharpe": round(float(rets.mean() / sd * np.sqrt(252)), 2) if sd > 1e-12 else None,
        "hit_rate": round(float((rets > 0).mean()), 3),
        "curve": curve,
        "note": ("Cartera equiponderada de los modelos con veredicto GO o "
                 "REVISAR. Los NO-GO quedan fuera."),
    }


if __name__ == "__main__":
    sys.exit(main())
