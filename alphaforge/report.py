"""Informe de resultados: consola, JSON y HTML autocontenido."""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .utils import fmt_num, fmt_pct

W = 76


def _line(ch: str = "-") -> str:
    return ch * W


def console_report(res) -> str:
    m, cfg = res.metrics, res.cfg
    L: list[str] = []
    a = L.append

    a("=" * W)
    a(f"ALPHAFORGE — {cfg.data.ticker}   [{cfg.fingerprint()}]")
    a(f"{datetime.now():%Y-%m-%d %H:%M}   horizonte: {cfg.label.horizon}"
      f"   entrada: T-{cfg.data.anchor_offset_min}' del cierre")
    a("=" * W)

    a("\nANCLA DE EJECUCIÓN")
    a(_line())
    ar = m.get("anchor_residual", {})
    a(f"  fuente                 {m.get('anchor_source')}")
    a(f"  cobertura real         {fmt_pct(m.get('anchor_coverage', 0), 1)}")
    if ar.get("n", 0) > 5:
        a(f"  residuo Close/ancla    media {1e4 * ar['mean']:+.1f} bps | "
          f"sigma {1e4 * ar['std']:.1f} bps | n={ar['n']}")
        a(f"  intervalo 5-95%        [{1e4 * ar['q05']:+.0f}, {1e4 * ar['q95']:+.0f}] bps")
    if m.get("anchor_coverage", 1) < 0.5:
        a("  AVISO: la mayor parte del histórico usa Close como proxy del ancla.")
        a("         Mira la fila 'Sharpe bajo estrés' antes de creerte nada.")

    a("\nRENDIMIENTO FUERA DE MUESTRA (walk-forward)")
    a(_line())
    bh = m.get("buy_hold", {})
    rows = [
        ("Sharpe",              fmt_num(m.get("sharpe"), 2),  fmt_num(bh.get("sharpe"), 2)),
        ("Sortino",             fmt_num(m.get("sortino"), 2), fmt_num(bh.get("sortino"), 2)),
        ("CAGR",                fmt_pct(m.get("cagr")),       fmt_pct(bh.get("cagr"))),
        ("Volatilidad anual",   fmt_pct(m.get("vol_annual")), fmt_pct(bh.get("vol_annual"))),
        ("Máx. drawdown",       fmt_pct(m.get("max_dd")),     fmt_pct(bh.get("max_dd"))),
        ("Calmar",              fmt_num(m.get("calmar"), 2),  fmt_num(bh.get("calmar"), 2)),
        ("Retorno total",       fmt_pct(m.get("total_return")), fmt_pct(bh.get("total_return"))),
    ]
    a(f"  {'métrica':<24}{'estrategia':>14}{'buy & hold':>16}")
    for k, v1, v2 in rows:
        a(f"  {k:<24}{v1:>14}{v2:>16}")

    a("")
    a(f"  {'AUC (OOS)':<24}{fmt_num(m.get('auc_oos')):>14}")
    a(f"  {'Acierto direccional':<24}{fmt_pct(m.get('directional_accuracy'), 1):>14}")
    a(f"  {'Tasa de acierto PnL':<24}{fmt_pct(m.get('hit_rate'), 1):>14}")
    a(f"  {'Profit factor':<24}{fmt_num(m.get('profit_factor'), 2):>14}")
    a(f"  {'Operaciones OOS':<24}{m.get('n_active', 0):>14}")
    a(f"  {'Exposición':<24}{fmt_pct(m.get('exposure'), 1):>14}")
    a(f"  {'Rotación media':<24}{fmt_num(m.get('turnover_mean'), 2):>14}")
    a(f"  {'Coste anual':<24}{fmt_pct(m.get('cost_drag_annual')):>14}")
    a(f"  {'t-stat':<24}{fmt_num(m.get('t_stat'), 2):>14}")
    a("")
    a(f"  Solapamiento de operaciones (el ancla de t+1 cae dentro de la ventana de t):")
    a(f"    días encadenados {fmt_pct(m.get('overlap_frac'), 0)} | "
      f"autocorrelación {fmt_num(m.get('autocorr_1'))} | "
      f"Sharpe corregido {fmt_num(m.get('sharpe_overlap_adj'), 2)} "
      f"(factor {fmt_num(m.get('newey_west_factor'), 2)})")

    if "sharpe_anchor_stress_p05" in m:
        a("")
        a(f"  Sharpe bajo estrés del ancla (Monte Carlo, ruido real medido):")
        a(f"    media {fmt_num(m['sharpe_anchor_stress_mean'], 2)} | "
          f"p05 {fmt_num(m['sharpe_anchor_stress_p05'], 2)} | "
          f"p95 {fmt_num(m['sharpe_anchor_stress_p95'], 2)}")

    a("\nDIAGNÓSTICO DE SOBREAJUSTE")
    a(_line())
    if res.pbo is not None and np.isfinite(res.pbo.pbo):
        a(f"  PBO                       {fmt_num(res.pbo.pbo)}  "
          f"({res.pbo.n_configs} configs, {res.pbo.n_combinations} particiones)")
        a(f"  pendiente IS->OOS         {fmt_num(res.pbo.is_oos_slope)}  "
          f"(negativa = el backtest engaña)")
        a(f"  correlación IS/OOS        {fmt_num(res.pbo.is_oos_corr)}")
    a(f"  Deflated Sharpe           {fmt_num(m.get('deflated_sharpe'))}")
    a(f"  Sharpe exigible por azar  {fmt_num(m.get('dsr_threshold_sharpe_annual'), 2)} anual")
    a(f"  PSR vs. cero              {fmt_num(m.get('psr_vs_zero'))}")
    p = res.permutation
    if p.get("n", 0):
        a(f"  p-valor permutación       Sharpe {fmt_num(p.get('p_sharpe'))} | "
          f"AUC {fmt_num(p.get('p_auc'))}  (n={p['n']})")
        a(f"  Sharpe medio del nulo     {fmt_num(p.get('null_sharpe_mean'), 2)} "
          f"(p95 {fmt_num(p.get('null_sharpe_p95'), 2)})")
    s = m.get("stability", {})
    a(f"  Estabilidad entre folds   media {fmt_num(s.get('mean'), 2)} ± "
      f"{fmt_num(s.get('std'), 2)} | {fmt_pct(s.get('frac_positive'), 0)} positivos | "
      f"peor {fmt_num(s.get('worst'), 2)}")

    c = res.calibration
    a("\nCALIDAD DE LA PROBABILIDAD")
    a(_line())
    a(f"  Brier {fmt_num(c.get('brier'))} | Brier skill {fmt_num(c.get('brier_skill'))} | "
      f"log-loss {fmt_num(c.get('log_loss'))} | ECE {fmt_num(c.get('ece'))}")
    if c.get("curve"):
        a("  curva de fiabilidad (predicho -> observado):")
        for b in c["curve"]:
            bar = "#" * int(30 * b["p_obs"])
            a(f"    {b['p_pred']:.2f} -> {b['p_obs']:.2f}  n={b['n']:<5} {bar}")

    a("\nRESULTADO POR FOLD")
    a(_line())
    a(f"  {'#':<3}{'periodo':<26}{'AUC':>8}{'Sharpe':>9}{'retorno':>10}{'DD':>9}")
    for f in res.fold_metrics:
        a(f"  {f['fold']:<3}{f['test_start']}..{f['test_end']:<12}"
          f"{fmt_num(f.get('auc')):>8}{fmt_num(f.get('sharpe'), 2):>9}"
          f"{fmt_pct(f.get('total_return'), 1):>10}{fmt_pct(f.get('max_dd'), 1):>9}")

    a("\nBÚSQUEDA DE HIPERPARÁMETROS")
    a(_line())
    a(f"  configuraciones probadas  {res.search['n_trials']}")
    a(f"  Sharpe CV mediano / máx   {fmt_num(res.search['median_sharpe_cv'], 2)} / "
      f"{fmt_num(res.search['max_sharpe_cv'], 2)}")
    for fam, spec in res.search["best_by_family"].items():
        a(f"  · {fam:<12} AUC {fmt_num(spec.get('auc'))}  "
          f"Sharpe CV {fmt_num(spec.get('sharpe_cv'), 2)}")

    a("\nVEREDICTO")
    a("=" * W)
    for ch in res.verdict["checks"]:
        mark = "OK " if ch["pass"] else ("!! " if ch["critical"] else " ~ ")
        a(f"  [{mark}] {ch['check']:<32} {ch['detail']}")
    a(_line())
    d = res.verdict["decision"]
    a(f"  DECISIÓN: {d}   ({res.verdict['n_failed_critical']}/"
      f"{res.verdict['n_critical']} comprobaciones críticas fallidas)")
    a(_verdict_text(d))
    a("=" * W)
    return "\n".join(L)


def _verdict_text(d: str) -> str:
    if d == "GO":
        return ("  El edge sobrevive a purga, deflación, permutación y estrés de\n"
                "  ejecución. Sigue siendo una apuesta estadística: dimensiona en\n"
                "  consecuencia y revisa el modelo cada trimestre.")
    if d == "REVISAR":
        return ("  Hay señal pero alguna defensa ha saltado. Mira qué comprobación\n"
                "  falla antes de arriesgar dinero: casi siempre es muestra corta,\n"
                "  PBO alto o inestabilidad entre folds.")
    return ("  No hay evidencia de edge por encima del azar. Lo honesto es no\n"
            "  operar esto. Prueba otro activo, otro horizonte o acepta que este\n"
            "  valor no es predecible con estas features.")


def save_artifacts(res, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{res.cfg.data.ticker}_{res.cfg.fingerprint()}"
    paths = {}

    p = os.path.join(out_dir, f"{tag}_oos.csv")
    res.oos.to_csv(p)
    paths["oos"] = p

    summary = {
        "ticker": res.cfg.data.ticker,
        "fingerprint": res.cfg.fingerprint(),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": res.cfg.to_dict(),
        "metrics": _jsonable(res.metrics),
        "fold_metrics": _jsonable(res.fold_metrics),
        "calibration": _jsonable(res.calibration),
        "permutation": _jsonable(res.permutation),
        "verdict": _jsonable(res.verdict),
        "pbo": (_jsonable({"pbo": res.pbo.pbo, "n_configs": res.pbo.n_configs,
                           "is_oos_slope": res.pbo.is_oos_slope,
                           "is_oos_corr": res.pbo.is_oos_corr})
                if res.pbo else None),
        "search": _jsonable({k: v for k, v in res.search.items()
                             if k not in ("perf_matrix",)}),
        "data_report": _jsonable(res.data_report),
    }
    p = os.path.join(out_dir, f"{tag}_report.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    paths["json"] = p

    p = os.path.join(out_dir, f"{tag}_report.txt")
    with open(p, "w") as f:
        f.write(console_report(res))
    paths["txt"] = p

    try:
        p = os.path.join(out_dir, f"{tag}_report.html")
        with open(p, "w") as f:
            f.write(html_report(res))
        paths["html"] = p
    except Exception:                              # noqa: BLE001
        pass

    try:
        import pickle
        p = os.path.join(out_dir, f"{tag}_model.pkl")
        with open(p, "wb") as f:
            pickle.dump({"cfg": res.cfg.to_dict(), "final": res.final_model,
                         "verdict": res.verdict, "metrics": _jsonable(res.metrics)}, f)
        paths["model"] = p
    except Exception:                              # noqa: BLE001
        pass
    return paths


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def html_report(res) -> str:
    eq = res.oos["equity"]
    bh = (1 + res.oos["y_ret"].fillna(0)).cumprod()
    dates = [str(d.date()) for d in eq.index]
    v = res.verdict["decision"]
    color = {"GO": "#16a34a", "REVISAR": "#d97706", "NO-GO": "#dc2626"}[v]
    m = res.metrics
    dd = (eq / eq.cummax() - 1)

    checks = "".join(
        f"<tr><td>{'✔' if c['pass'] else ('✘' if c['critical'] else '~')}</td>"
        f"<td>{c['check']}</td><td class='mono'>{c['detail']}</td></tr>"
        for c in res.verdict["checks"])
    folds = "".join(
        f"<tr><td>{f['fold']}</td><td>{f['test_start']} → {f['test_end']}</td>"
        f"<td class='mono'>{fmt_num(f.get('auc'))}</td>"
        f"<td class='mono'>{fmt_num(f.get('sharpe'), 2)}</td>"
        f"<td class='mono'>{fmt_pct(f.get('total_return'), 1)}</td></tr>"
        for f in res.fold_metrics)

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>AlphaForge — {res.cfg.data.ticker}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;
   background:#0b0f14;color:#e6edf3}}
 .wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 80px}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:15px;text-transform:uppercase;
   letter-spacing:.08em;color:#8b949e;margin:36px 0 12px;font-weight:600}}
 .sub{{color:#8b949e;font-size:13px;margin-bottom:24px}}
 .badge{{display:inline-block;padding:6px 16px;border-radius:999px;font-weight:700;
   background:{color};color:#fff;font-size:14px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
 .card{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px}}
 .k{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
 .val{{font-size:22px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 td,th{{padding:7px 10px;border-bottom:1px solid #21262d;text-align:left}}
 .mono{{font-family:ui-monospace,SFMono-Regular,monospace}}
 canvas{{background:#161b22;border-radius:10px;padding:10px;border:1px solid #21262d}}
</style></head><body><div class="wrap">
<h1>{res.cfg.data.ticker} <span class="badge">{v}</span></h1>
<div class="sub">Entrada a T-{res.cfg.data.anchor_offset_min}′ del cierre ·
 horizonte {res.cfg.label.horizon} · ancla {m.get('anchor_source')}
 ({fmt_pct(m.get('anchor_coverage', 0), 0)} real) · huella {res.cfg.fingerprint()}</div>

<div class="grid">
 <div class="card"><div class="k">Sharpe OOS</div><div class="val">{fmt_num(m.get('sharpe'), 2)}</div></div>
 <div class="card"><div class="k">CAGR</div><div class="val">{fmt_pct(m.get('cagr'), 1)}</div></div>
 <div class="card"><div class="k">Máx DD</div><div class="val">{fmt_pct(m.get('max_dd'), 1)}</div></div>
 <div class="card"><div class="k">AUC</div><div class="val">{fmt_num(m.get('auc_oos'))}</div></div>
 <div class="card"><div class="k">PBO</div><div class="val">{fmt_num(res.pbo.pbo) if res.pbo else 'n/a'}</div></div>
 <div class="card"><div class="k">Deflated Sharpe</div><div class="val">{fmt_num(m.get('deflated_sharpe'))}</div></div>
 <div class="card"><div class="k">p-valor</div><div class="val">{fmt_num(res.permutation.get('p_sharpe'))}</div></div>
 <div class="card"><div class="k">Operaciones</div><div class="val">{m.get('n_active', 0)}</div></div>
</div>

<h2>Curva de resultados (fuera de muestra)</h2>
<canvas id="eq" height="110"></canvas>
<h2>Drawdown</h2>
<canvas id="dd" height="70"></canvas>
<h2>Comprobaciones</h2><table>{checks}</table>
<h2>Folds</h2><table><tr><th>#</th><th>Periodo</th><th>AUC</th><th>Sharpe</th>
<th>Retorno</th></tr>{folds}</table>
</div>
<script>
const L={json.dumps(dates)};
const o={{responsive:true,plugins:{{legend:{{labels:{{color:'#8b949e'}}}}}},
 scales:{{x:{{ticks:{{color:'#8b949e',maxTicksLimit:10}},grid:{{color:'#21262d'}}}},
 y:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}}}},
 elements:{{point:{{radius:0}}}}}};
new Chart(document.getElementById('eq'),{{type:'line',data:{{labels:L,datasets:[
 {{label:'Estrategia',data:{json.dumps([round(float(x), 4) for x in eq])},
   borderColor:'{color}',borderWidth:2,fill:false}},
 {{label:'Buy & hold',data:{json.dumps([round(float(x), 4) for x in bh])},
   borderColor:'#6e7681',borderWidth:1.5,borderDash:[5,4],fill:false}}]}},options:o}});
new Chart(document.getElementById('dd'),{{type:'line',data:{{labels:L,datasets:[
 {{label:'Drawdown',data:{json.dumps([round(float(x), 4) for x in dd])},
   borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.2)',borderWidth:1,fill:true}}]}},
 options:o}});
</script></body></html>"""
