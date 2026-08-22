"""Interfaz de línea de comandos de AlphaForge.

  python -m alphaforge selftest
  python -m alphaforge run     --ticker AAPL --start 2008-01-01 --trials 60
  python -m alphaforge predict --ticker AAPL --model af_runs/AAPL_xxx_model.pkl
  python -m alphaforge scan    --tickers AAPL,MSFT,NVDA,SPY
  python -m alphaforge demo    --signal mean_revert
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback

import numpy as np

from .config import Config
from .report import console_report, save_artifacts
from .utils import fmt_num, fmt_pct, get_logger, suppress_noisy_warnings


# ---------------------------------------------------------------------------
def _apply_common(cfg: Config, a) -> Config:
    if a.ticker:
        cfg.data.ticker = a.ticker.upper()
    if a.start:
        cfg.data.start = a.start
    if a.end:
        cfg.data.end = a.end
    if a.trials:
        cfg.model.n_trials = a.trials
    if a.splits:
        cfg.validation.n_splits = a.splits
    if a.horizon:
        cfg.label.horizon = a.horizon
    if a.offset:
        cfg.data.anchor_offset_min = a.offset
    if a.threshold:
        cfg.backtest.prob_threshold = a.threshold
    if a.families:
        cfg.model.families = tuple(f.strip() for f in a.families.split(","))
    if a.permutations is not None:
        cfg.validation.n_permutations = a.permutations
    if a.no_short:
        cfg.backtest.allow_short = False
    if a.sizing:
        cfg.backtest.sizing = a.sizing
    if a.out:
        cfg.out_dir = a.out
    if a.quiet:
        cfg.verbose = 0
    elif a.debug:
        cfg.verbose = 2
    if a.seed is not None:
        cfg.model.random_state = a.seed
    if getattr(a, "leak_alarm", None) is not None:
        cfg.validation.leak_alarm_corr = a.leak_alarm
    if a.config:
        with open(a.config) as f:
            cfg = Config.from_dict(json.load(f))
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ticker", "-t", help="símbolo, p.ej. AAPL o SAN.MC")
    p.add_argument("--start", help="fecha inicial YYYY-MM-DD")
    p.add_argument("--end", help="fecha final YYYY-MM-DD")
    p.add_argument("--trials", type=int, help="nº de configuraciones a probar")
    p.add_argument("--splits", type=int, help="folds del walk-forward")
    p.add_argument("--horizon", choices=["close_next", "open_next", "close_next2"])
    p.add_argument("--offset", type=int, help="minutos antes del cierre (defecto 30)")
    p.add_argument("--threshold", type=float, help="umbral de probabilidad para operar")
    p.add_argument("--families", help="logit,hgb,extratrees,mlp,gru")
    p.add_argument("--permutations", type=int, help="repeticiones del test de permutación")
    p.add_argument("--sizing", choices=["binary", "confidence", "kelly"])
    p.add_argument("--no-short", action="store_true", help="solo posiciones largas")
    p.add_argument("--out", help="directorio de salida")
    p.add_argument("--config", help="JSON con la configuración completa")
    p.add_argument("--seed", type=int)
    p.add_argument("--leak-alarm", type=float,
                   help="|rho| máximo tolerado entre feature y retorno futuro (0.35)")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--debug", action="store_true")


# ---------------------------------------------------------------------------
def cmd_run(a) -> int:
    from .pipeline import run_experiment
    cfg = _apply_common(Config(), a)
    res = run_experiment(cfg)
    print("\n" + console_report(res))
    paths = save_artifacts(res, cfg.out_dir)
    print("\nArchivos generados:")
    for k, v in paths.items():
        print(f"  {k:<8} {v}")
    if a.predict_after:
        from .pipeline import predict_next
        print("\n" + _format_prediction(predict_next(cfg, res)))
    return 0 if res.verdict["decision"] != "NO-GO" else 2


def cmd_predict(a) -> int:
    from .pipeline import predict_next, run_experiment
    cfg = _apply_common(Config(), a)
    res = None
    if a.model and os.path.exists(a.model):
        with open(a.model, "rb") as f:
            blob = pickle.load(f)
        cfg = Config.from_dict(blob["cfg"])
        if a.ticker:
            cfg.data.ticker = a.ticker.upper()

        class _Stub:                                # el modelo ya está entrenado
            pass
        res = _Stub()
        res.final_model = blob["final"]
        res.verdict = blob["verdict"]
        res.metrics = blob["metrics"]
        res.pbo = None
        get_logger(cfg.verbose).info(f"modelo cargado de {a.model}")
    else:
        res = run_experiment(cfg)

    out = predict_next(cfg, res, live_anchor_price=a.price)
    print(_format_prediction(out))
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_scan(a) -> int:
    from .pipeline import predict_next, run_experiment
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    rows = []
    for t in tickers:
        cfg = _apply_common(Config(), a)
        cfg.data.ticker = t
        try:
            res = run_experiment(cfg)
            pred = predict_next(cfg, res)
            rows.append({"ticker": t, **pred})
            save_artifacts(res, cfg.out_dir)
        except Exception as e:                      # noqa: BLE001
            print(f"[{t}] error: {type(e).__name__}: {e}", file=sys.stderr)
            if a.debug:
                traceback.print_exc()
    if not rows:
        return 1
    rows.sort(key=lambda r: -(r["prob_up"] if np.isfinite(r["prob_up"]) else 0))
    print("\n" + "=" * 92)
    print(f"{'ticker':<9}{'P(sube)':>9}{'E[ret]':>9}{'P10':>9}{'P90':>9}"
          f"{'pos':>7}{'Sharpe':>9}{'AUC':>7}{'PBO':>7}{'veredicto':>12}")
    print("-" * 92)
    for r in rows:
        print(f"{r['ticker']:<9}{fmt_pct(r['prob_up'], 1):>9}"
              f"{fmt_pct(r['expected_return'], 2):>9}{fmt_pct(r['ret_p10'], 2):>9}"
              f"{fmt_pct(r['ret_p90'], 2):>9}{fmt_num(r['position'], 2):>7}"
              f"{fmt_num(r['oos_sharpe'], 2):>9}{fmt_num(r['oos_auc'], 2):>7}"
              f"{fmt_num(r['pbo'], 2):>7}{r['verdict']:>12}")
    print("=" * 92)
    print("Solo son operables las filas con veredicto GO.")
    return 0


def cmd_selftest(a) -> int:
    from .selfcheck import run_all
    res = run_all(verbose=True, stop_on_fail=a.stop_on_fail)
    return 0 if all(r.passed for r in res) else 1


def cmd_demo(a) -> int:
    """Ejecuta el pipeline completo sobre un mercado sintético. Sin red."""
    from .pipeline import run_experiment
    from .synthetic import make_market_data
    cfg = _apply_common(Config(), a)
    cfg.data.ticker = f"SYNTH_{a.signal.upper()}"
    cfg.features.cross_asset = False
    if not a.trials:
        cfg.model.n_trials = 12
    if not a.splits:
        cfg.validation.n_splits = 5
    if a.permutations is None:
        cfg.validation.n_permutations = 30
    if a.signal != "none" and getattr(a, "leak_alarm", None) is None:
        # la señal está inyectada a propósito: la guarda anti-fuga se relaja
        cfg.validation.leak_alarm_corr = 0.95
    md = make_market_data(n=a.n, seed=cfg.model.random_state, signal=a.signal,
                          signal_strength=a.strength, with_context=False)
    res = run_experiment(cfg, md=md)
    print("\n" + console_report(res))
    if a.out:
        for k, v in save_artifacts(res, cfg.out_dir).items():
            print(f"  {k:<8} {v}")
    return 0


def _format_prediction(p: dict) -> str:
    up = p["prob_up"]
    direction = "ALCISTA" if up > 0.5 else "BAJISTA"
    bar = "#" * int(40 * up)
    lines = [
        "=" * 66,
        f"SEÑAL — {p['ticker']}   (datos hasta {p['as_of']})",
        "=" * 66,
        f"  Precio de referencia (ancla)  {p['anchor_price']:.4f}  [{p['anchor_source']}]",
        f"  Horizonte                     {p['horizon']}",
        "",
        f"  P(sube)   {fmt_pct(up, 1):>8}   {direction}",
        f"            [{bar:<40}]",
        f"  Retorno esperado (mediana)    {fmt_pct(p['expected_return'], 2)}",
        f"  Intervalo P10 - P90           {fmt_pct(p['ret_p10'], 2)} ... "
        f"{fmt_pct(p['ret_p90'], 2)}",
        f"  Volatilidad ex-ante           {fmt_pct(p['sigma_ex_ante'], 2)}",
        f"  Posición sugerida             {p['position']:+.2f}x",
        "-" * 66,
        f"  Fiabilidad del modelo: Sharpe OOS {fmt_num(p['oos_sharpe'], 2)} | "
        f"AUC {fmt_num(p['oos_auc'])} | PBO {fmt_num(p['pbo'])}",
        f"  Veredicto del backtest: {p['verdict']}",
    ]
    if p["verdict"] == "NO-GO":
        lines.append("  >> El backtest NO valida este modelo. Señal informativa, no operable.")
    lines.append("=" * 66)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alphaforge",
        description="Predicción direccional con ML validada contra sobreajuste.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="entrena y valida un modelo completo")
    _add_common(r)
    r.add_argument("--predict-after", action="store_true",
                   help="muestra la señal para mañana al terminar")
    r.set_defaults(func=cmd_run)

    pr = sub.add_parser("predict", help="señal para el próximo día")
    _add_common(pr)
    pr.add_argument("--model", help="ruta al .pkl guardado por 'run'")
    pr.add_argument("--price", type=float,
                    help="precio en vivo del ancla (si el feed diario aún no cerró)")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_predict)

    s = sub.add_parser("scan", help="varios valores de una tacada")
    _add_common(s)
    s.add_argument("--tickers", required=True, help="lista separada por comas")
    s.set_defaults(func=cmd_scan)

    t = sub.add_parser("selftest", help="autodiagnóstico del sistema")
    t.add_argument("--stop-on-fail", action="store_true")
    t.set_defaults(func=cmd_selftest)

    d = sub.add_parser("demo", help="prueba completa sobre mercado sintético (sin red)")
    _add_common(d)
    d.add_argument("--signal", default="mean_revert",
                   choices=["none", "mean_revert", "momentum", "regime"])
    d.add_argument("--strength", type=float, default=0.45)
    d.add_argument("--n", type=int, default=2500)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    suppress_noisy_warnings()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrumpido", file=sys.stderr)
        return 130
    except Exception as e:                          # noqa: BLE001
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            print("Ejecuta con --debug para ver la traza completa.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
