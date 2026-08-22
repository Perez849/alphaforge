"""Configuración central de AlphaForge.

Todo parámetro que afecte al resultado vive aquí y se serializa junto al
modelo entrenado. Objetivo: cualquier backtest es reproducible bit a bit.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Constantes de mercado
# ---------------------------------------------------------------------------
US_CLOSE_ET = "16:00"
DEFAULT_ANCHOR_OFFSET_MIN = 30      # minutos antes del cierre en los que operamos
TRADING_DAYS_YEAR = 252


@dataclass
class DataConfig:
    ticker: str = "SPY"
    start: str = "2005-01-01"
    end: str | None = None
    # activos de contexto macro / cross-sectional
    context_tickers: tuple[str, ...] = ("SPY", "QQQ", "^VIX", "TLT", "UUP", "GLD")
    cache_dir: str = "~/.alphaforge/cache"
    max_cache_age_hours: float = 12.0
    # Ancla intradía
    anchor_offset_min: int = DEFAULT_ANCHOR_OFFSET_MIN
    anchor_intervals: tuple[str, ...] = ("30m", "1h")   # se prueban en orden
    min_rows: int = 750                                  # ~3 años; por debajo abortamos


@dataclass
class LabelConfig:
    # 'close_next'  : y = Close(t+1) / P_anchor(t) - 1   <- captura gap + sesión completa
    # 'open_next'   : y = Open(t+1)  / P_anchor(t) - 1   <- solo captura el gap
    # 'close_next2' : y = Close(t+2) / P_anchor(t) - 1
    horizon: str = "close_next"
    # Umbral de clasificación: 'zero' | 'cost' | 'vol'
    #   cost -> sube si y > coste_ida_y_vuelta
    #   vol  -> sube si y > k * sigma_t (banda neutra simétrica)
    threshold_mode: str = "cost"
    neutral_band_k: float = 0.15          # solo si threshold_mode == 'vol'
    # ponderación de muestras
    time_decay_halflife_days: float = 750.0
    vol_normalize_target: bool = True     # objetivo de regresión escalado por sigma
    max_calendar_gap_days: int = 5        # descarta etiquetas que salten huecos


@dataclass
class FeatureConfig:
    rsi_periods: tuple[int, ...] = (2, 7, 14, 28)
    roc_periods: tuple[int, ...] = (1, 2, 3, 5, 10, 21, 63, 126)
    ma_periods: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
    vol_windows: tuple[int, ...] = (5, 10, 21, 63)
    atr_periods: tuple[int, ...] = (5, 14)
    bb_periods: tuple[int, ...] = (20,)
    stoch_periods: tuple[int, ...] = (14,)
    adx_periods: tuple[int, ...] = (14,)
    range_windows: tuple[int, ...] = (10, 21, 63, 252)
    weekly: bool = True
    monthly: bool = True
    calendar: bool = True
    cross_asset: bool = True
    # Poda de redundancia
    max_abs_corr: float = 0.95
    min_variance: float = 1e-10


@dataclass
class ModelConfig:
    families: tuple[str, ...] = ("logit", "hgb", "extratrees", "mlp", "gru")
    n_trials: int = 60                    # búsqueda aleatoria de hiperparámetros
    max_features_selected: int = 40
    calibration: str = "isotonic"         # 'isotonic' | 'sigmoid' | 'none'
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    use_torch: bool = True                # si torch no está, cae a sklearn
    random_state: int = 20260820


@dataclass
class ValidationConfig:
    n_splits: int = 8                     # folds de walk-forward externo
    inner_splits: int = 4                 # folds internos para hiperparámetros
    purge_days: int = 3                   # >= horizonte + 1
    embargo_pct: float = 0.01
    min_train_days: int = 1000
    anchored: bool = True                 # train expansivo (True) o rolling (False)
    # Diagnóstico de overfitting
    cscv_blocks: int = 12                 # S en el algoritmo CSCV (par)
    n_permutations: int = 100             # test de permutación de etiquetas
    pbo_alarm: float = 0.35               # PBO por encima de esto => veredicto NO-GO
    leak_alarm_corr: float = 0.35         # |rho| feature vs. retorno futuro que aborta
    dsr_alarm: float = 0.90               # DSR por debajo de esto => sospechoso
    min_oos_trades: int = 60


@dataclass
class BacktestConfig:
    commission_bps: float = 0.5           # por lado
    spread_bps: float = 1.0               # medio spread por lado
    slippage_bps: float = 1.0             # impacto por lado
    prob_threshold: float = 0.55          # umbral de entrada
    sizing: str = "confidence"            # 'binary' | 'confidence' | 'kelly'
    kelly_fraction: float = 0.25
    max_leverage: float = 1.0
    allow_short: bool = True
    coherence_gate: bool = True           # no operar si las dos cabezas discrepan
    benchmark: str = "buy_hold"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    out_dir: str = "./af_runs"
    verbose: int = 1

    # -- utilidades ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash estable de la configuración: identifica el experimento."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        def build(klass, sub):
            fields = {f.name for f in dataclasses.fields(klass)}
            clean = {}
            for k, v in (sub or {}).items():
                if k not in fields:
                    continue
                # las tuplas se serializan como listas en JSON
                ftype = {f.name: f.type for f in dataclasses.fields(klass)}[k]
                if isinstance(v, list) and "tuple" in str(ftype):
                    v = tuple(v)
                clean[k] = v
            return klass(**clean)

        return cls(
            data=build(DataConfig, d.get("data")),
            label=build(LabelConfig, d.get("label")),
            features=build(FeatureConfig, d.get("features")),
            model=build(ModelConfig, d.get("model")),
            validation=build(ValidationConfig, d.get("validation")),
            backtest=build(BacktestConfig, d.get("backtest")),
            out_dir=d.get("out_dir", "./af_runs"),
            verbose=d.get("verbose", 1),
        )

    def validate(self) -> None:
        """Coherencia interna. Falla pronto y fuerte."""
        errs = []
        h = {"close_next": 1, "open_next": 1, "close_next2": 2}.get(self.label.horizon)
        if h is None:
            errs.append(f"label.horizon desconocido: {self.label.horizon}")
        elif self.validation.purge_days < h + 1:
            errs.append(
                f"validation.purge_days ({self.validation.purge_days}) debe ser >= "
                f"horizonte+1 ({h + 1}) o habrá solapamiento de etiquetas."
            )
        if self.validation.cscv_blocks % 2 != 0:
            errs.append("validation.cscv_blocks debe ser par (CSCV parte en mitades).")
        if self.validation.cscv_blocks < 6:
            errs.append("validation.cscv_blocks < 6 hace el PBO inestable.")
        if not 0.5 <= self.backtest.prob_threshold < 1.0:
            errs.append("backtest.prob_threshold debe estar en [0.5, 1).")
        if self.data.anchor_offset_min <= 0 or self.data.anchor_offset_min > 180:
            errs.append("data.anchor_offset_min fuera de rango razonable (1-180).")
        if self.model.n_trials < 5:
            errs.append("model.n_trials < 5: la búsqueda no tiene sentido estadístico.")
        if errs:
            raise ValueError("Configuración inválida:\n  - " + "\n  - ".join(errs))

    @property
    def horizon_days(self) -> int:
        return {"close_next": 1, "open_next": 1, "close_next2": 2}[self.label.horizon]

    @property
    def roundtrip_cost(self) -> float:
        bt = self.backtest
        return 2.0 * (bt.commission_bps + bt.spread_bps + bt.slippage_bps) / 10_000.0
