"""
Portfolio strategy evaluator - v12 with metric comparison plots and Markdown report.

Requested v6:
    start="2026-04-05"
    end="2026-05-31"
    interval="15m"
    margin_rate=20% for margin-based policies

Note:
    Yahoo 15m data availability is limited. If a ticker fails to download,
    move the date range closer to today or switch interval to "1h".

    Policy 4 = Policy 2 framework + Policy 3 signal selection:
        - Starts with the listed portfolio holdings.
        - Uses Policy 2-style account equity, profit pyramiding, margin/exposure caps.
        - But when increasing exposure, it does NOT add proportionally to all holdings.
          It adds only to the ticker where Policy 3 detects a buy signal.
        - Uses Policy 3-style target/trailing/VWMA logic to decide where to reduce holding.
        - It can also apply portfolio-level partial de-risk like Policy 2.

Policies:
    1. Hold portfolio.
    2. Profit pyramid + partial de-risk.
    3. Sequential single-ticker dip/VWMA scanner.
    4. Policy 2 + Policy 3 guided add/reduce.

Install:
    pip install yfinance pandas numpy matplotlib

Run:
    python portfolio_policy_eval_v12_report.py
"""

from __future__ import annotations

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================================
# User positions and ticker mapping
# ============================================================

MARKET_POSITIONS = [
    ("AI Index ($1)", 2.5),
    ("Broadcom Inc (24 Hours)", 6),
    ("Quanta Services Inc (24 Hours)", 4),
    ("Micron Technology Inc (24 Hours)", 2),
    ("Applied Materials Inc (24 Hours)", 4),
    ("Lumentum Holdings Inc", 2),
    ("Astera Labs Inc (24 Hours)", 4),
    ("AXT Inc", 10),
    ("Marvell Technology Group Ltd (24 Hours)", 4),
    ("COHERENT CORP", 2),
]

MARKET_MAP = {
    "AI Index ($1)": "AIQ",   # Proxy. Change to BOTZ, IRBO, QQQ, etc. if preferred.
    "Broadcom Inc (24 Hours)": "AVGO",
    "Quanta Services Inc (24 Hours)": "PWR",
    "Micron Technology Inc (24 Hours)": "MU",
    "Applied Materials Inc (24 Hours)": "AMAT",
    "Lumentum Holdings Inc": "LITE",
    "Astera Labs Inc (24 Hours)": "ALAB",
    "AXT Inc": "AXTI",
    "Marvell Technology Group Ltd (24 Hours)": "MRVL",
    "COHERENT CORP": "COHR",
}


# Optional IG epic mapping for live engine synchronization.
# Fill with platform epics when using IG live mode, e.g. {"AVGO": "..."}.
IG_EPIC_MAP: dict[str, str] = {}

ENGINE_POSITION_SOURCES = {"ig", "manual", "zero"}


def is_ig_configured() -> bool:
    """Return True when enough IG credentials are present for a live account sync."""
    return all(os.getenv(name) for name in ("IG_USERNAME", "IG_PASSWORD", "IG_API_KEY"))


def normalize_engine_position_source(
    source: str | None = None,
    *,
    ig_configured: bool | None = None,
    mode: str | None = None,
) -> str:
    """
    Resolve the engine's initial-position source.

    Explicit CLI/env values win.  Without an explicit value, prefer an IG account
    sync whenever IG is configured; otherwise keep the hand-entered portfolio only
    for backtest/paper modes and fall back to zero units for live-like modes.
    """
    raw_source = source or os.getenv("ENGINE_POSITION_SOURCE")
    if raw_source:
        normalized = raw_source.strip().lower()
        if normalized not in ENGINE_POSITION_SOURCES:
            raise ValueError(
                "ENGINE_POSITION_SOURCE must be one of "
                f"{sorted(ENGINE_POSITION_SOURCES)}, got {raw_source!r}"
            )
        return normalized

    configured = is_ig_configured() if ig_configured is None else ig_configured
    if configured:
        return "ig"

    normalized_mode = (mode or os.getenv("ENGINE_MODE") or "paper").strip().lower()
    if normalized_mode in {"backtest", "paper"}:
        return "manual"
    return "zero"


def _lookup_ticker_for_epic(epic: str, epic_map: dict[str, str]) -> str | None:
    if epic in epic_map:
        return epic_map[epic]
    for ticker, mapped_epic in epic_map.items():
        if mapped_epic == epic:
            return ticker
    if epic in MARKET_MAP:
        return MARKET_MAP[epic]
    if epic in set(MARKET_MAP.values()):
        return epic
    return None


def units_from_ig_open_positions(
    positions: Any,
    epic_map: dict[str, str],
) -> dict[str, float]:
    """
    Convert IG open-position payloads into ticker -> signed contract counts.

    BUY positions add units, SELL positions subtract units.  The helper accepts
    either the raw IG response with a ``positions`` list or a positions list.
    ``epic_map`` may be either ticker -> epic or epic -> ticker.
    """
    raw_positions = positions.get("positions", positions) if isinstance(positions, dict) else positions
    units: dict[str, float] = {}

    for item in raw_positions or []:
        market = item.get("market", {}) if isinstance(item, dict) else {}
        position = item.get("position", {}) if isinstance(item, dict) else {}

        epic = (
            market.get("epic")
            or position.get("epic")
            or (item.get("epic") if isinstance(item, dict) else None)
        )
        if not epic:
            continue

        ticker = _lookup_ticker_for_epic(str(epic), epic_map)
        if not ticker:
            continue

        direction = str(
            position.get("direction")
            or (item.get("direction") if isinstance(item, dict) else "")
            or "BUY"
        ).upper()
        raw_size = (
            position.get("size")
            or position.get("dealSize")
            or position.get("contractSize")
            or (item.get("size") if isinstance(item, dict) else None)
            or 0.0
        )
        size = float(raw_size)
        signed_size = -size if direction == "SELL" else size
        units[ticker] = units.get(ticker, 0.0) + signed_size

    return units


def current_positions_synced_action(source: str, units: dict[str, float]) -> dict[str, Any]:
    """Dashboard/audit action showing the account state was checked first."""
    tickers = sorted(units)
    counts = {ticker: float(units[ticker]) for ticker in tickers}
    return {
        "action": "CURRENT_POSITIONS_SYNCED",
        "source": source,
        "tickers": ",".join(tickers),
        "counts": counts,
        "fee": 0.0,
    }


# ============================================================
# Config
# ============================================================

@dataclass
class BacktestConfig:
    start: str = "2026-04-05"
    end: str = "2026-05-31"
    interval: str = "15m"

    # If None, initial account equity equals listed portfolio notional at first date.
    initial_account_equity: Optional[float] = None

    # Trading platform cost model:
    # commission_rate is charged on trade notional or trade cash amount.
    # 0.001 = 0.10%.
    commission_rate: float = 0.0010

    # Fixed deterministic slippage. Normally keep this at 0 because v9 uses random adverse slippage.
    slippage_rate: float = 0.0000

    # Random adverse execution delta.
    # 0.001 = max 0.10%.
    # Buy fill = reference_price * (1 + fixed_slippage + random[0, max])
    # Sell fill = reference_price * (1 - fixed_slippage - random[0, max])
    random_slippage_max_rate: float = 0.0010
    random_seed: int = 42

    data_dir: str = "data"
    plot_dir: str = "plots"
    report_dir: str = "report"


@dataclass
class Policy2Config:
    margin_rate: float = 0.20
    max_gross_exposure_to_equity: float = 3.0
    max_margin_usage: float = 0.50
    add_trigger_return: float = 0.05
    profit_reinvest_fraction: float = 0.50
    derisk_drawdown_from_peak: float = 0.08
    derisk_fraction: float = 0.35
    min_profit_to_derisk: float = 0.02
    require_new_peak_after_derisk: bool = True
    enable_hard_exit: bool = False
    hard_exit_drawdown_from_peak: float = 0.20


@dataclass
class Policy3SequentialConfig:
    confirm_bars: int = 3
    max_lows_per_plan: int = 3
    invest_fraction_of_cash: float = 0.50
    max_volume_ratio_for_buy: float = 1.20
    min_vwma20_slope_5bars: float = -0.02
    min_vwma50_slope_5bars: float = -0.03
    rebound_observe_ratio: float = 0.75
    trailing_drawdown: float = 0.04
    sell_on_close_below_vwma10: bool = True
    signal_priority: str = "best_expected_rebound"


@dataclass
class Policy4Config:
    """
    Hybrid:
        Policy 2 account/risk framework.
        Policy 3 decides which ticker to add and which ticker to reduce.
    """
    margin_rate: float = 0.20

    max_gross_exposure_to_equity: float = 3.0
    max_margin_usage: float = 0.50

    # Use profit as margin for signal-guided add.
    profit_reinvest_fraction: float = 0.50

    # Do not add unless there is at least this much profit.
    min_profit_to_add: float = 0.01

    # Policy 3 signal engine
    confirm_bars: int = 3
    max_lows_per_plan: int = 3
    max_volume_ratio_for_buy: float = 1.20
    min_vwma20_slope_5bars: float = -0.02
    min_vwma50_slope_5bars: float = -0.03
    signal_priority: str = "best_expected_rebound"

    # Reduce ticker after Policy 3 target/turn condition
    rebound_observe_ratio: float = 0.75
    trailing_drawdown: float = 0.04
    sell_on_close_below_vwma10: bool = True
    guided_reduce_fraction: float = 0.35

    # Portfolio-level safety de-risk from Policy 2
    enable_portfolio_derisk: bool = True
    derisk_drawdown_from_peak: float = 0.08
    derisk_fraction: float = 0.20
    min_profit_to_derisk: float = 0.02
    require_new_peak_after_derisk: bool = True

    enable_hard_exit: bool = False
    hard_exit_drawdown_from_peak: float = 0.20


@dataclass
class Policy5Config:
    """
    Ratio-constrained hybrid:
        Policy 2 account/risk framework.
        Policy 3 decides candidate add/reduce tickers.
        Target ratios come from the initial portfolio notional weights.

    Restriction:
        - Do not buy a ticker if its current notional weight is already above
          target_weight * (1 + max_overweight_ratio).
        - Do not sell/reduce a ticker if its current notional weight is already below
          target_weight * (1 - max_underweight_ratio).
        - When reducing, limit the sale so it should not push that ticker below
          the lower target band.
    """
    margin_rate: float = 0.20

    max_gross_exposure_to_equity: float = 3.0
    max_margin_usage: float = 0.50

    profit_reinvest_fraction: float = 0.50
    min_profit_to_add: float = 0.01

    confirm_bars: int = 3
    max_lows_per_plan: int = 3
    max_volume_ratio_for_buy: float = 1.20
    min_vwma20_slope_5bars: float = -0.02
    min_vwma50_slope_5bars: float = -0.03
    signal_priority: str = "best_expected_rebound"

    rebound_observe_ratio: float = 0.75
    trailing_drawdown: float = 0.04
    sell_on_close_below_vwma10: bool = True
    guided_reduce_fraction: float = 0.35

    # Ratio guard. 0.25 means allow +/-25% relative drift from target weight.
    max_overweight_ratio: float = 0.25
    max_underweight_ratio: float = 0.25

    enable_portfolio_derisk: bool = True
    derisk_drawdown_from_peak: float = 0.08
    derisk_fraction: float = 0.20
    min_profit_to_derisk: float = 0.02
    require_new_peak_after_derisk: bool = True

    enable_hard_exit: bool = False
    hard_exit_drawdown_from_peak: float = 0.20


class IGClient:
    """Minimal IG client interface used by the live Policy 5 engine path."""

    def login(self) -> None:
        raise NotImplementedError("Configure a concrete IGClient before live trading.")

    def open_positions(self) -> Any:
        raise NotImplementedError("Configure a concrete IGClient before live trading.")


class Policy5TradingEngine:
    """
    Lightweight live-engine state container for Policy 5.

    The historical simulations still use ``simulate_policy_5_ratio_guarded_guided_pyramid``.
    This class holds the synchronized live starting units so order decisions can
    use the current IG account state instead of stale ``MARKET_POSITIONS`` data.
    """

    def __init__(
        self,
        cfg: BacktestConfig | None = None,
        cfg5: Policy5Config | None = None,
        *,
        initial_units: dict[str, float] | None = None,
        position_source: str = "manual",
    ):
        self.cfg = cfg or BacktestConfig()
        self.cfg5 = cfg5 or Policy5Config()
        self.position_source = position_source
        self.units = dict(initial_units) if initial_units is not None else initial_units_from_positions()
        self.actions: list[dict[str, Any]] = []
        self.record_current_positions_synced(position_source, self.units)

    def record_action(self, action: dict[str, Any]) -> None:
        self.actions.append(action)

    def record_current_positions_synced(self, source: str, units: dict[str, float]) -> None:
        self.record_action(current_positions_synced_action(source, units))


def synchronized_initial_units(
    *,
    source: str,
    ig_client: IGClient | None = None,
    epic_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Fetch or construct the units used to initialize ``Policy5TradingEngine``."""
    if source == "ig":
        if ig_client is None:
            raise ValueError("ig_client is required when ENGINE_POSITION_SOURCE=ig")
        positions = ig_client.open_positions()
        return units_from_ig_open_positions(positions, epic_map or IG_EPIC_MAP)
    if source == "manual":
        return initial_units_from_positions()
    if source == "zero":
        return {ticker: 0.0 for ticker in all_tickers()}
    raise ValueError(f"Unknown engine position source: {source!r}")


def run_trading_engine(
    *,
    ig_client: IGClient | None = None,
    position_source: str | None = None,
    epic_map: dict[str, str] | None = None,
    mode: str | None = None,
    cfg: BacktestConfig | None = None,
    cfg5: Policy5Config | None = None,
) -> Policy5TradingEngine:
    """
    Initialize the live Policy 5 trading engine after synchronizing positions.

    When IG is selected, this logs in, calls ``IGClient.open_positions()``, maps
    those positions to ticker units, records ``CURRENT_POSITIONS_SYNCED``, and
    only then returns an engine ready for live order decisions.
    """
    source = normalize_engine_position_source(
        position_source,
        ig_configured=ig_client is not None or is_ig_configured(),
        mode=mode,
    )
    if source == "ig":
        if ig_client is None:
            ig_client = IGClient()
        ig_client.login()

    initial_units = synchronized_initial_units(
        source=source,
        ig_client=ig_client,
        epic_map=epic_map or IG_EPIC_MAP,
    )
    return Policy5TradingEngine(
        cfg=cfg,
        cfg5=cfg5,
        initial_units=initial_units,
        position_source=source,
    )

# ============================================================
# Data
# ============================================================

def ensure_dirs(cfg: BacktestConfig) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)


def download_one(ticker: str, cfg: BacktestConfig) -> pd.DataFrame:
    fname = f"{ticker}_{cfg.interval}_{cfg.start}_{cfg.end}.csv".replace(":", "-")
    path = os.path.join(cfg.data_dir, fname)

    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    raw = yf.download(
        ticker,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )

    if raw.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"

    df = df.rename(columns={
        date_col: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
    except Exception:
        pass

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0)

    df.to_csv(path, index=False)
    return df


def download_all(cfg: BacktestConfig) -> dict[str, pd.DataFrame]:
    ensure_dirs(cfg)
    out = {}
    for t in sorted(set(MARKET_MAP.values())):
        print(f"Downloading/loading {t}...")
        out[t] = download_one(t, cfg)
    return out


def align_closes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, df in data.items():
        frames.append(df[["date", "close"]].rename(columns={"close": ticker}))

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")

    merged = merged.sort_values("date").ffill().dropna().reset_index(drop=True)
    return merged


def align_ohlcv_for_ticker(data: dict[str, pd.DataFrame], ticker: str, dates: pd.Series) -> pd.DataFrame:
    df = data[ticker].copy()
    merged = pd.DataFrame({"date": dates}).merge(df, on="date", how="left")
    merged = merged.sort_values("date").ffill().dropna().reset_index(drop=True)
    return merged


def initial_units_from_positions() -> dict[str, float]:
    units = {}
    for market, size in MARKET_POSITIONS:
        ticker = MARKET_MAP[market]
        units[ticker] = units.get(ticker, 0.0) + float(size)
    return units


def all_tickers() -> list[str]:
    return sorted(set(MARKET_MAP.values()))


def unit_snapshot(prefix: str, units: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_units_{t}": float(units.get(t, 0.0)) for t in all_tickers()}


def zero_unit_snapshot(prefix: str) -> dict[str, float]:
    return {f"{prefix}_units_{t}": 0.0 for t in all_tickers()}


def portfolio_notional(prices: pd.Series, units: dict[str, float]) -> float:
    return sum(abs(u) * float(prices[t]) for t, u in units.items())


def portfolio_unrealized(prices: pd.Series, units: dict[str, float], avg_entry: dict[str, float]) -> float:
    return sum(u * (float(prices[t]) - avg_entry[t]) for t, u in units.items())


# ============================================================
# Costs
# ============================================================

def random_adverse_delta(cfg: BacktestConfig) -> float:
    """Random adverse fill delta in rate terms, between 0 and max configured rate."""
    return random.uniform(0.0, cfg.random_slippage_max_rate)


def buy_fill(price: float, cfg: BacktestConfig) -> float:
    """
    Simulate that you can never buy exactly at the displayed/current price.
    Buy fills are always worse: higher than reference price.
    """
    return price * (1 + cfg.slippage_rate + random_adverse_delta(cfg))


def sell_fill(price: float, cfg: BacktestConfig) -> float:
    """
    Simulate that you can never sell exactly at the displayed/current price.
    Sell fills are always worse: lower than reference price.
    """
    return price * (1 - cfg.slippage_rate - random_adverse_delta(cfg))


def fee(amount: float, cfg: BacktestConfig) -> float:
    """Commission fee charged by the trading platform."""
    return abs(amount) * cfg.commission_rate


# ============================================================
# Indicators and swings
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    vol = d["volume"].replace(0, pd.NA)
    pv = d["close"] * d["volume"]

    for w in [10, 20, 50]:
        d[f"vwma{w}"] = pv.rolling(w).sum() / vol.rolling(w).sum()

    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_ratio20"] = d["volume"] / d["vol_ma20"]
    d["vwma20_slope_5bars"] = d["vwma20"] / d["vwma20"].shift(5) - 1
    d["vwma50_slope_5bars"] = d["vwma50"] / d["vwma50"].shift(5) - 1

    return d


def confirmed_swings(df: pd.DataFrame, confirm_bars: int) -> pd.DataFrame:
    d = df.copy()
    k = confirm_bars
    n = len(d)

    d["confirmed_swing_high"] = np.nan
    d["confirmed_swing_low"] = np.nan

    for i in range(k, n - k):
        if d["high"].iloc[i] == d["high"].iloc[i - k: i + k + 1].max():
            d.loc[i + k, "confirmed_swing_high"] = d["high"].iloc[i]

        if d["low"].iloc[i] == d["low"].iloc[i - k: i + k + 1].min():
            d.loc[i + k, "confirmed_swing_low"] = d["low"].iloc[i]

    return d


class SignalScannerState:
    def __init__(
        self,
        ticker: str,
        confirm_bars: int,
        max_lows_per_plan: int,
        max_volume_ratio_for_buy: float,
        min_vwma20_slope_5bars: float,
        min_vwma50_slope_5bars: float,
    ):
        self.ticker = ticker
        self.confirm_bars = confirm_bars
        self.max_lows_per_plan = max_lows_per_plan
        self.max_volume_ratio_for_buy = max_volume_ratio_for_buy
        self.min_vwma20_slope_5bars = min_vwma20_slope_5bars
        self.min_vwma50_slope_5bars = min_vwma50_slope_5bars

        self.swing_lows: list[float] = []
        self.active_H: Optional[float] = None
        self.buy_levels: list[float] = []
        self.next_level_idx: int = 0

    def update_and_get_signal(self, row: pd.Series) -> Optional[dict]:
        if pd.notna(row.get("confirmed_swing_low", np.nan)):
            self.swing_lows.append(float(row["confirmed_swing_low"]))
            self.swing_lows = self.swing_lows[-10:]

        if pd.notna(row.get("confirmed_swing_high", np.nan)):
            H = float(row["confirmed_swing_high"])
            lows = list(reversed(self.swing_lows))[: self.max_lows_per_plan]
            lows = [x for x in lows if x < H]

            if lows:
                self.active_H = H
                self.buy_levels = sorted(set((H + L) / 2 for L in lows), reverse=True)
                self.next_level_idx = 0

        if self.active_H is None or self.next_level_idx >= len(self.buy_levels):
            return None

        trigger = self.buy_levels[self.next_level_idx]
        touched = float(row["low"]) <= trigger

        vwma_ok = (
            pd.notna(row.get("vwma20_slope_5bars", np.nan))
            and pd.notna(row.get("vwma50_slope_5bars", np.nan))
            and row["vwma20_slope_5bars"] >= self.min_vwma20_slope_5bars
            and row["vwma50_slope_5bars"] >= self.min_vwma50_slope_5bars
        )

        volume_ok = (
            pd.notna(row.get("vol_ratio20", np.nan))
            and row["vol_ratio20"] <= self.max_volume_ratio_for_buy
        )

        if not (touched and vwma_ok and volume_ok):
            return None

        vwma20 = row.get("vwma20", np.nan)
        close = float(row["close"])
        vwma_discount = 0.0
        if pd.notna(vwma20) and vwma20 != 0:
            vwma_discount = close / float(vwma20) - 1

        expected_rebound = (self.active_H - trigger) / trigger if trigger > 0 else 0.0

        return {
            "ticker": self.ticker,
            "trigger": trigger,
            "active_H": self.active_H,
            "level_idx": self.next_level_idx,
            "expected_rebound": expected_rebound,
            "vwma_discount": vwma_discount,
        }

    def consume_signal_level(self) -> None:
        self.next_level_idx += 1


def choose_signal(signals: list[dict], priority: str) -> dict:
    if priority == "largest_vwma_discount":
        return sorted(signals, key=lambda x: x["vwma_discount"])[0]
    return sorted(signals, key=lambda x: x["expected_rebound"], reverse=True)[0]


def prepare_frames(
    data: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    confirm_bars: int,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for t in sorted(set(MARKET_MAP.values())):
        df = align_ohlcv_for_ticker(data, t, closes["date"])
        df = add_indicators(df)
        df = confirmed_swings(df, confirm_bars)
        frames[t] = df.reset_index(drop=True)
    return frames


# ============================================================
# Policy 1
# ============================================================

def simulate_policy_1_hold(closes: pd.DataFrame, init_units: dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, r in closes.iterrows():
        row = {
            "date": r["date"],
            "Policy 1: Hold Portfolio": portfolio_notional(r, init_units),
        }
        row.update(unit_snapshot("policy_1", init_units))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Common CFD-style account state
# ============================================================

def account_state(
    prices: pd.Series,
    balance: float,
    units: dict[str, float],
    avg_entry: dict[str, float],
    margin_rate: float,
) -> dict:
    notional = portfolio_notional(prices, units)
    unrealized = portfolio_unrealized(prices, units, avg_entry)
    equity = balance + unrealized
    used_margin = notional * margin_rate
    free_margin = equity - used_margin
    return {
        "notional": notional,
        "unrealized": unrealized,
        "equity": equity,
        "used_margin": used_margin,
        "free_margin": free_margin,
        "margin_usage": used_margin / equity if equity > 0 else float("inf"),
        "exposure_ratio": notional / equity if equity > 0 else float("inf"),
    }


def target_weights_from_initial(prices: pd.Series, init_units: dict[str, float]) -> dict[str, float]:
    total = portfolio_notional(prices, init_units)
    if total <= 0:
        return {t: 0.0 for t in init_units}
    return {t: abs(init_units.get(t, 0.0)) * float(prices[t]) / total for t in init_units}


def current_weights(prices: pd.Series, units: dict[str, float]) -> dict[str, float]:
    total = portfolio_notional(prices, units)
    if total <= 0:
        return {t: 0.0 for t in units}
    return {t: abs(units.get(t, 0.0)) * float(prices[t]) / total for t in units}


def is_ticker_overweight(
    prices: pd.Series,
    units: dict[str, float],
    ticker: str,
    target_weights: dict[str, float],
    max_overweight_ratio: float,
) -> bool:
    weights = current_weights(prices, units)
    target = target_weights.get(ticker, 0.0)
    current = weights.get(ticker, 0.0)
    return current >= target * (1 + max_overweight_ratio)


def is_ticker_underweight(
    prices: pd.Series,
    units: dict[str, float],
    ticker: str,
    target_weights: dict[str, float],
    max_underweight_ratio: float,
) -> bool:
    weights = current_weights(prices, units)
    target = target_weights.get(ticker, 0.0)
    current = weights.get(ticker, 0.0)
    return current <= target * (1 - max_underweight_ratio)


def max_reduce_fraction_to_respect_lower_band(
    prices: pd.Series,
    units: dict[str, float],
    ticker: str,
    target_weights: dict[str, float],
    max_underweight_ratio: float,
    requested_fraction: float,
) -> float:
    """
    Approximate maximum fraction of ticker holdings that can be closed without
    pushing the ticker below its lower target-weight band.

    Uses current prices and current notional. Ignores fee/slippage for the guard,
    which makes it slightly optimistic but stable.
    """
    total_notional = portfolio_notional(prices, units)
    ticker_notional = abs(units.get(ticker, 0.0)) * float(prices[ticker])

    if total_notional <= 0 or ticker_notional <= 0:
        return 0.0

    target = target_weights.get(ticker, 0.0)
    lower_weight = max(0.0, target * (1 - max_underweight_ratio))

    # After reducing x of ticker notional:
    # new_weight = ticker_notional*(1-x) / (total_notional - ticker_notional*x)
    # Need new_weight >= lower_weight.
    if lower_weight <= 0:
        return requested_fraction

    current_weight = ticker_notional / total_notional
    if current_weight <= lower_weight:
        return 0.0

    denom = ticker_notional * (1 - lower_weight)
    if denom <= 0:
        return 0.0

    max_x = (ticker_notional - lower_weight * total_notional) / denom
    max_x = max(0.0, min(1.0, max_x))

    return min(requested_fraction, max_x)


def add_single_ticker_exposure(
    prices: pd.Series,
    ticker: str,
    units: dict[str, float],
    avg_entry: dict[str, float],
    add_margin: float,
    margin_rate: float,
    cfg: BacktestConfig,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """
    Adds exposure only to one ticker.
    Margin is a capacity concept; balance is reduced only by fees.
    """
    if add_margin <= 0:
        return units, avg_entry, 0.0, 0.0

    p = buy_fill(float(prices[ticker]), cfg)
    add_notional = add_margin / margin_rate
    add_units = add_notional / p
    trade_fee = fee(add_notional, cfg)

    old_u = units.get(ticker, 0.0)
    old_avg = avg_entry.get(ticker, p)
    new_u = old_u + add_units
    avg_entry[ticker] = (old_u * old_avg + add_units * p) / new_u
    units[ticker] = new_u

    return units, avg_entry, add_units, trade_fee


def add_proportional_exposure(
    prices: pd.Series,
    units: dict[str, float],
    avg_entry: dict[str, float],
    add_margin: float,
    margin_rate: float,
    cfg: BacktestConfig,
) -> tuple[dict[str, float], dict[str, float], float]:
    current_notional = portfolio_notional(prices, units)
    if current_notional <= 0 or add_margin <= 0:
        return units, avg_entry, 0.0

    add_notional_total = add_margin / margin_rate
    total_fee = 0.0

    for t, u in list(units.items()):
        p = buy_fill(float(prices[t]), cfg)
        weight = abs(u) * float(prices[t]) / current_notional
        add_notional = add_notional_total * weight
        add_units = add_notional / p
        trade_fee = fee(add_notional, cfg)
        total_fee += trade_fee

        old_u = units[t]
        new_u = old_u + add_units
        if new_u > 0:
            avg_entry[t] = (old_u * avg_entry[t] + add_units * p) / new_u
        units[t] = new_u

    return units, avg_entry, total_fee


def close_ticker_fraction(
    prices: pd.Series,
    ticker: str,
    units: dict[str, float],
    avg_entry: dict[str, float],
    fraction: float,
    cfg: BacktestConfig,
) -> tuple[dict[str, float], float, float, float]:
    u = units.get(ticker, 0.0)
    if u <= 0:
        return units, 0.0, 0.0, 0.0

    close_u = u * fraction
    p = sell_fill(float(prices[ticker]), cfg)
    notional_closed = close_u * p
    realized = close_u * (p - avg_entry[ticker])
    trade_fee = fee(notional_closed, cfg)

    units[ticker] = u - close_u
    if units[ticker] <= 1e-12:
        units[ticker] = 0.0

    return units, close_u, realized, trade_fee


def close_all_fraction(
    prices: pd.Series,
    units: dict[str, float],
    avg_entry: dict[str, float],
    fraction: float,
    cfg: BacktestConfig,
) -> tuple[dict[str, float], float, float]:
    realized = 0.0
    total_fee = 0.0

    for t in list(units.keys()):
        units, _, pnl, trade_fee = close_ticker_fraction(prices, t, units, avg_entry, fraction, cfg)
        realized += pnl
        total_fee += trade_fee

    return units, realized, total_fee


# ============================================================
# Policy 2
# ============================================================

def simulate_policy_2_profit_pyramid_derisk(
    closes: pd.DataFrame,
    init_units: dict[str, float],
    cfg: BacktestConfig,
    cfg2: Policy2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    first = closes.iloc[0]
    units = dict(init_units)
    avg_entry = {t: float(first[t]) for t in units}

    initial_notional = portfolio_notional(first, units)
    initial_equity = cfg.initial_account_equity if cfg.initial_account_equity is not None else initial_notional

    balance = initial_equity
    max_equity = initial_equity
    last_add_equity = initial_equity
    can_derisk_from_current_peak = True

    rows = []
    trades = []

    for _, r in closes.iterrows():
        date = r["date"]
        state = account_state(r, balance, units, avg_entry, cfg2.margin_rate)
        equity = state["equity"]

        if equity > max_equity:
            max_equity = equity
            can_derisk_from_current_peak = True

        if cfg2.enable_hard_exit and equity <= max_equity * (1 - cfg2.hard_exit_drawdown_from_peak):
            units, realized, trade_fee = close_all_fraction(r, units, avg_entry, 1.0, cfg)
            balance += realized - trade_fee
            trades.append({
                "date": date,
                "policy": "2",
                "action": "EXIT_ALL",
                "equity_before": equity,
                "realized_pnl": realized,
                "fee": trade_fee,
            })
            max_equity = balance
            last_add_equity = balance
            can_derisk_from_current_peak = False

        else:
            profit_return = equity / initial_equity - 1
            should_derisk = (
                can_derisk_from_current_peak
                and profit_return >= cfg2.min_profit_to_derisk
                and equity <= max_equity * (1 - cfg2.derisk_drawdown_from_peak)
            )

            if should_derisk:
                units, realized, trade_fee = close_all_fraction(r, units, avg_entry, cfg2.derisk_fraction, cfg)
                balance += realized - trade_fee
                trades.append({
                    "date": date,
                    "policy": "2",
                    "action": "DERISK",
                    "equity_before": equity,
                    "max_equity": max_equity,
                    "fraction_closed": cfg2.derisk_fraction,
                    "realized_pnl": realized,
                    "fee": trade_fee,
                })
                if cfg2.require_new_peak_after_derisk:
                    can_derisk_from_current_peak = False

            state = account_state(r, balance, units, avg_entry, cfg2.margin_rate)
            equity = state["equity"]

            if equity >= last_add_equity * (1 + cfg2.add_trigger_return):
                profit_pool = max(0.0, equity - initial_equity)
                max_notional = equity * cfg2.max_gross_exposure_to_equity
                notional_capacity = max(0.0, max_notional - state["notional"])
                max_margin = equity * cfg2.max_margin_usage
                margin_capacity = max(0.0, max_margin - state["used_margin"])
                desired_add_margin = profit_pool * cfg2.profit_reinvest_fraction

                add_margin = min(
                    desired_add_margin,
                    margin_capacity,
                    notional_capacity * cfg2.margin_rate,
                    max(0.0, state["free_margin"]),
                )

                if add_margin > 0:
                    units, avg_entry, trade_fee = add_proportional_exposure(
                        r, units, avg_entry, add_margin, cfg2.margin_rate, cfg
                    )
                    balance -= trade_fee
                    trades.append({
                        "date": date,
                        "policy": "2",
                        "action": "ADD",
                        "add_margin": add_margin,
                        "profit_pool": profit_pool,
                        "fee": trade_fee,
                    })
                    last_add_equity = equity

        state = account_state(r, balance, units, avg_entry, cfg2.margin_rate)
        row = {
            "date": date,
            "Policy 2: Profit Pyramid + De-risk": state["equity"],
            "policy_2_balance": balance,
            "policy_2_notional": state["notional"],
            "policy_2_unrealized": state["unrealized"],
            "policy_2_used_margin": state["used_margin"],
            "policy_2_free_margin": state["free_margin"],
            "policy_2_margin_usage": state["margin_usage"],
            "policy_2_exposure_ratio": state["exposure_ratio"],
        }
        row.update(unit_snapshot("policy_2", units))
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(trades)


# ============================================================
# Policy 3: sequential single-ticker scanner
# ============================================================

def simulate_policy_3_sequential(
    data: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    init_units: dict[str, float],
    cfg: BacktestConfig,
    cfg3: Policy3SequentialConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    initial_equity = portfolio_notional(closes.iloc[0], init_units)
    cash = initial_equity

    frames = prepare_frames(data, closes, cfg3.confirm_bars)
    scanners = {
        t: SignalScannerState(
            t,
            cfg3.confirm_bars,
            cfg3.max_lows_per_plan,
            cfg3.max_volume_ratio_for_buy,
            cfg3.min_vwma20_slope_5bars,
            cfg3.min_vwma50_slope_5bars,
        )
        for t in frames
    }

    active_ticker: Optional[str] = None
    shares = 0.0
    avg_cost = math.nan
    active_H = math.nan
    target_active = False
    pmax_after_target = -math.inf

    rows = []
    trades = []

    for i in range(len(closes)):
        date = closes.loc[i, "date"]

        signals = []
        for t, scanner in scanners.items():
            sig = scanner.update_and_get_signal(frames[t].iloc[i])
            if sig is not None:
                signals.append(sig)

        if active_ticker is None and signals and cash > 0:
            sig = choose_signal(signals, cfg3.signal_priority)
            t = sig["ticker"]
            trigger = sig["trigger"]
            fill = buy_fill(trigger, cfg)

            gross = cash * cfg3.invest_fraction_of_cash
            trade_fee = fee(gross, cfg)
            net = gross - trade_fee

            shares = net / fill
            cash -= gross
            avg_cost = fill
            active_ticker = t
            active_H = sig["active_H"]
            target_active = False
            pmax_after_target = -math.inf
            scanners[t].consume_signal_level()

            trades.append({
                "date": date,
                "policy": "3",
                "action": "BUY",
                "ticker": t,
                "trigger": trigger,
                "fill": fill,
                "gross": gross,
                "fee": trade_fee,
                "cash_after": cash,
                "shares": shares,
                "avg_cost": avg_cost,
                "active_H": active_H,
            })

        if active_ticker is not None:
            row = frames[active_ticker].iloc[i]
            close = float(row["close"])
            high = float(row["high"])

            if not target_active:
                observe = avg_cost + cfg3.rebound_observe_ratio * (active_H - avg_cost)
                if high >= observe:
                    target_active = True
                    pmax_after_target = max(high, close)
                    trades.append({
                        "date": date,
                        "policy": "3",
                        "action": "TARGET_ZONE_REACHED",
                        "ticker": active_ticker,
                        "price": observe,
                    })

            if target_active:
                pmax_after_target = max(pmax_after_target, high, close)
                trailing_stop = pmax_after_target * (1 - cfg3.trailing_drawdown)
                hit_trailing = close <= trailing_stop
                below_vwma10 = (
                    cfg3.sell_on_close_below_vwma10
                    and pd.notna(row.get("vwma10", np.nan))
                    and close < float(row["vwma10"])
                )

                if hit_trailing or below_vwma10:
                    fill = sell_fill(close, cfg)
                    gross = shares * fill
                    trade_fee = fee(gross, cfg)
                    cash += gross - trade_fee

                    trades.append({
                        "date": date,
                        "policy": "3",
                        "action": "SELL",
                        "ticker": active_ticker,
                        "fill": fill,
                        "gross": gross,
                        "fee": trade_fee,
                        "cash_after": cash,
                        "shares_sold": shares,
                        "reason": "TRAILING_STOP" if hit_trailing else "BELOW_VWMA10",
                    })

                    active_ticker = None
                    shares = 0.0
                    avg_cost = math.nan
                    active_H = math.nan
                    target_active = False
                    pmax_after_target = -math.inf

        if active_ticker is None:
            equity = cash
            held_ticker = "CASH"
        else:
            close = float(frames[active_ticker].iloc[i]["close"])
            equity = cash + shares * close
            held_ticker = active_ticker

        row = {
            "date": date,
            "Policy 3: Sequential Single-Ticker Dip + VWMA": equity,
            "policy_3_active_ticker": held_ticker,
        }
        units_snapshot = zero_unit_snapshot("policy_3")
        if active_ticker is not None:
            units_snapshot[f"policy_3_units_{active_ticker}"] = float(shares)
        row.update(units_snapshot)
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(trades)


# ============================================================
# Policy 4: Policy2 framework + Policy3 guided add/reduce
# ============================================================

def simulate_policy_4_signal_guided_pyramid(
    data: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    init_units: dict[str, float],
    cfg: BacktestConfig,
    cfg4: Policy4Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    first = closes.iloc[0]
    units = dict(init_units)
    avg_entry = {t: float(first[t]) for t in units}

    initial_notional = portfolio_notional(first, units)
    initial_equity = cfg.initial_account_equity if cfg.initial_account_equity is not None else initial_notional
    balance = initial_equity

    frames = prepare_frames(data, closes, cfg4.confirm_bars)
    scanners = {
        t: SignalScannerState(
            t,
            cfg4.confirm_bars,
            cfg4.max_lows_per_plan,
            cfg4.max_volume_ratio_for_buy,
            cfg4.min_vwma20_slope_5bars,
            cfg4.min_vwma50_slope_5bars,
        )
        for t in frames
    }

    max_equity = initial_equity
    can_derisk_from_current_peak = True

    # Per-ticker guided exit state, activated after signal-guided add.
    guided_H: dict[str, float] = {}
    guided_target_active: dict[str, bool] = {}
    guided_pmax: dict[str, float] = {}

    rows = []
    trades = []

    for i in range(len(closes)):
        r = closes.iloc[i]
        date = r["date"]

        state = account_state(r, balance, units, avg_entry, cfg4.margin_rate)
        equity = state["equity"]

        if equity > max_equity:
            max_equity = equity
            can_derisk_from_current_peak = True

        # Update scanners and collect buy signals.
        signals = []
        for t, scanner in scanners.items():
            sig = scanner.update_and_get_signal(frames[t].iloc[i])
            if sig is not None:
                signals.append(sig)

        # Optional portfolio-level hard exit.
        if cfg4.enable_hard_exit and equity <= max_equity * (1 - cfg4.hard_exit_drawdown_from_peak):
            units, realized, trade_fee = close_all_fraction(r, units, avg_entry, 1.0, cfg)
            balance += realized - trade_fee
            trades.append({
                "date": date,
                "policy": "4",
                "action": "EXIT_ALL",
                "reason": "portfolio_hard_exit",
                "equity_before": equity,
                "realized_pnl": realized,
                "fee": trade_fee,
            })
            max_equity = balance
            can_derisk_from_current_peak = False
            guided_H.clear()
            guided_target_active.clear()
            guided_pmax.clear()

        else:
            # Portfolio-level partial de-risk from Policy 2.
            if cfg4.enable_portfolio_derisk:
                profit_return = equity / initial_equity - 1
                should_derisk = (
                    can_derisk_from_current_peak
                    and profit_return >= cfg4.min_profit_to_derisk
                    and equity <= max_equity * (1 - cfg4.derisk_drawdown_from_peak)
                )
                if should_derisk:
                    units, realized, trade_fee = close_all_fraction(r, units, avg_entry, cfg4.derisk_fraction, cfg)
                    balance += realized - trade_fee
                    trades.append({
                        "date": date,
                        "policy": "4",
                        "action": "PORTFOLIO_DERISK",
                        "reason": "portfolio_pullback",
                        "fraction_closed": cfg4.derisk_fraction,
                        "equity_before": equity,
                        "max_equity": max_equity,
                        "realized_pnl": realized,
                        "fee": trade_fee,
                    })
                    if cfg4.require_new_peak_after_derisk:
                        can_derisk_from_current_peak = False

            # Guided add: choose ONE signal, add only that ticker.
            state = account_state(r, balance, units, avg_entry, cfg4.margin_rate)
            equity = state["equity"]
            profit_return = equity / initial_equity - 1

            if signals and profit_return >= cfg4.min_profit_to_add:
                sig = choose_signal(signals, cfg4.signal_priority)
                t = sig["ticker"]

                profit_pool = max(0.0, equity - initial_equity)
                max_notional = equity * cfg4.max_gross_exposure_to_equity
                notional_capacity = max(0.0, max_notional - state["notional"])
                max_margin = equity * cfg4.max_margin_usage
                margin_capacity = max(0.0, max_margin - state["used_margin"])
                desired_add_margin = profit_pool * cfg4.profit_reinvest_fraction

                add_margin = min(
                    desired_add_margin,
                    margin_capacity,
                    notional_capacity * cfg4.margin_rate,
                    max(0.0, state["free_margin"]),
                )

                if add_margin > 0:
                    units, avg_entry, add_units, trade_fee = add_single_ticker_exposure(
                        r, t, units, avg_entry, add_margin, cfg4.margin_rate, cfg
                    )
                    balance -= trade_fee
                    scanners[t].consume_signal_level()

                    guided_H[t] = sig["active_H"]
                    guided_target_active[t] = False
                    guided_pmax[t] = -math.inf

                    trades.append({
                        "date": date,
                        "policy": "4",
                        "action": "GUIDED_ADD",
                        "ticker": t,
                        "reason": "policy3_buy_signal",
                        "add_margin": add_margin,
                        "add_units": add_units,
                        "fee": trade_fee,
                        "equity_before": equity,
                        "active_H": sig["active_H"],
                        "trigger": sig["trigger"],
                        "expected_rebound": sig["expected_rebound"],
                        "vwma_discount": sig["vwma_discount"],
                    })

            # Guided ticker-specific reduce based on Policy 3 turn/exit.
            for t in list(guided_H.keys()):
                if units.get(t, 0.0) <= 0:
                    guided_H.pop(t, None)
                    guided_target_active.pop(t, None)
                    guided_pmax.pop(t, None)
                    continue

                row_t = frames[t].iloc[i]
                close_t = float(row_t["close"])
                high_t = float(row_t["high"])
                H_t = guided_H[t]
                avg_t = avg_entry[t]

                if not guided_target_active.get(t, False):
                    observe = avg_t + cfg4.rebound_observe_ratio * (H_t - avg_t)
                    if high_t >= observe:
                        guided_target_active[t] = True
                        guided_pmax[t] = max(high_t, close_t)
                        trades.append({
                            "date": date,
                            "policy": "4",
                            "action": "GUIDED_TARGET_ZONE_REACHED",
                            "ticker": t,
                            "price": observe,
                            "avg_entry": avg_t,
                            "active_H": H_t,
                        })

                if guided_target_active.get(t, False):
                    guided_pmax[t] = max(guided_pmax.get(t, -math.inf), high_t, close_t)
                    trailing_stop = guided_pmax[t] * (1 - cfg4.trailing_drawdown)
                    hit_trailing = close_t <= trailing_stop
                    below_vwma10 = (
                        cfg4.sell_on_close_below_vwma10
                        and pd.notna(row_t.get("vwma10", np.nan))
                        and close_t < float(row_t["vwma10"])
                    )

                    if hit_trailing or below_vwma10:
                        units, closed_units, realized, trade_fee = close_ticker_fraction(
                            r, t, units, avg_entry, cfg4.guided_reduce_fraction, cfg
                        )
                        balance += realized - trade_fee

                        trades.append({
                            "date": date,
                            "policy": "4",
                            "action": "GUIDED_REDUCE",
                            "ticker": t,
                            "reason": "TRAILING_STOP" if hit_trailing else "BELOW_VWMA10",
                            "fraction_closed": cfg4.guided_reduce_fraction,
                            "units_closed": closed_units,
                            "realized_pnl": realized,
                            "fee": trade_fee,
                            "balance_after": balance,
                        })

                        guided_target_active[t] = False
                        guided_pmax[t] = -math.inf
                        # Keep guided_H so a later target can reduce again only after target reactivates,
                        # but it will not re-fire immediately unless the price reaches target again.

        state = account_state(r, balance, units, avg_entry, cfg4.margin_rate)
        row = {
            "date": date,
            "Policy 4: Policy2 + Policy3 Guided Add/Reduce": state["equity"],
            "policy_4_balance": balance,
            "policy_4_notional": state["notional"],
            "policy_4_unrealized": state["unrealized"],
            "policy_4_used_margin": state["used_margin"],
            "policy_4_free_margin": state["free_margin"],
            "policy_4_margin_usage": state["margin_usage"],
            "policy_4_exposure_ratio": state["exposure_ratio"],
        }
        row.update(unit_snapshot("policy_4", units))
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(trades)



# ============================================================
# Policy 5: Policy4 with target-ratio guard
# ============================================================

def simulate_policy_5_ratio_guarded_guided_pyramid(
    data: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    init_units: dict[str, float],
    cfg: BacktestConfig,
    cfg5: Policy5Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    first = closes.iloc[0]
    units = dict(init_units)
    avg_entry = {t: float(first[t]) for t in units}

    target_weights = target_weights_from_initial(first, init_units)

    initial_notional = portfolio_notional(first, units)
    initial_equity = cfg.initial_account_equity if cfg.initial_account_equity is not None else initial_notional
    balance = initial_equity

    frames = prepare_frames(data, closes, cfg5.confirm_bars)
    scanners = {
        t: SignalScannerState(
            t,
            cfg5.confirm_bars,
            cfg5.max_lows_per_plan,
            cfg5.max_volume_ratio_for_buy,
            cfg5.min_vwma20_slope_5bars,
            cfg5.min_vwma50_slope_5bars,
        )
        for t in frames
    }

    max_equity = initial_equity
    can_derisk_from_current_peak = True

    guided_H: dict[str, float] = {}
    guided_target_active: dict[str, bool] = {}
    guided_pmax: dict[str, float] = {}

    rows = []
    trades = []

    for i in range(len(closes)):
        r = closes.iloc[i]
        date = r["date"]

        state = account_state(r, balance, units, avg_entry, cfg5.margin_rate)
        equity = state["equity"]

        if equity > max_equity:
            max_equity = equity
            can_derisk_from_current_peak = True

        signals = []
        for t, scanner in scanners.items():
            sig = scanner.update_and_get_signal(frames[t].iloc[i])
            if sig is not None:
                signals.append(sig)

        if cfg5.enable_hard_exit and equity <= max_equity * (1 - cfg5.hard_exit_drawdown_from_peak):
            units, realized, trade_fee = close_all_fraction(r, units, avg_entry, 1.0, cfg)
            balance += realized - trade_fee
            trades.append({
                "date": date,
                "policy": "5",
                "action": "EXIT_ALL",
                "reason": "portfolio_hard_exit",
                "equity_before": equity,
                "realized_pnl": realized,
                "fee": trade_fee,
            })
            max_equity = balance
            can_derisk_from_current_peak = False
            guided_H.clear()
            guided_target_active.clear()
            guided_pmax.clear()

        else:
            if cfg5.enable_portfolio_derisk:
                profit_return = equity / initial_equity - 1
                should_derisk = (
                    can_derisk_from_current_peak
                    and profit_return >= cfg5.min_profit_to_derisk
                    and equity <= max_equity * (1 - cfg5.derisk_drawdown_from_peak)
                )
                if should_derisk:
                    # Portfolio-wide proportional derisk preserves ratios approximately.
                    units, realized, trade_fee = close_all_fraction(r, units, avg_entry, cfg5.derisk_fraction, cfg)
                    balance += realized - trade_fee
                    trades.append({
                        "date": date,
                        "policy": "5",
                        "action": "PORTFOLIO_DERISK",
                        "reason": "portfolio_pullback",
                        "fraction_closed": cfg5.derisk_fraction,
                        "equity_before": equity,
                        "max_equity": max_equity,
                        "realized_pnl": realized,
                        "fee": trade_fee,
                    })
                    if cfg5.require_new_peak_after_derisk:
                        can_derisk_from_current_peak = False

            state = account_state(r, balance, units, avg_entry, cfg5.margin_rate)
            equity = state["equity"]
            profit_return = equity / initial_equity - 1

            # Guided add with overweight filter.
            if signals and profit_return >= cfg5.min_profit_to_add:
                eligible = [
                    s for s in signals
                    if not is_ticker_overweight(
                        r, units, s["ticker"], target_weights, cfg5.max_overweight_ratio
                    )
                ]

                if eligible:
                    sig = choose_signal(eligible, cfg5.signal_priority)
                    t = sig["ticker"]

                    profit_pool = max(0.0, equity - initial_equity)
                    max_notional = equity * cfg5.max_gross_exposure_to_equity
                    notional_capacity = max(0.0, max_notional - state["notional"])
                    max_margin = equity * cfg5.max_margin_usage
                    margin_capacity = max(0.0, max_margin - state["used_margin"])
                    desired_add_margin = profit_pool * cfg5.profit_reinvest_fraction

                    add_margin = min(
                        desired_add_margin,
                        margin_capacity,
                        notional_capacity * cfg5.margin_rate,
                        max(0.0, state["free_margin"]),
                    )

                    if add_margin > 0:
                        units, avg_entry, add_units, trade_fee = add_single_ticker_exposure(
                            r, t, units, avg_entry, add_margin, cfg5.margin_rate, cfg
                        )
                        balance -= trade_fee
                        scanners[t].consume_signal_level()

                        guided_H[t] = sig["active_H"]
                        guided_target_active[t] = False
                        guided_pmax[t] = -math.inf

                        trades.append({
                            "date": date,
                            "policy": "5",
                            "action": "RATIO_GUARDED_GUIDED_ADD",
                            "ticker": t,
                            "reason": "policy3_buy_signal_and_not_overweight",
                            "add_margin": add_margin,
                            "add_units": add_units,
                            "fee": trade_fee,
                            "equity_before": equity,
                            "active_H": sig["active_H"],
                            "trigger": sig["trigger"],
                            "target_weight": target_weights.get(t, 0.0),
                            "current_weight_before": current_weights(r, units).get(t, 0.0),
                        })
                else:
                    # Consume no signal; the same signal can remain relevant later if weights change.
                    trades.append({
                        "date": date,
                        "policy": "5",
                        "action": "SKIP_ADD_OVERWEIGHT",
                        "reason": "all_policy3_signals_overweight",
                        "signal_tickers": ",".join(sorted([s["ticker"] for s in signals])),
                        "fee": 0.0,
                    })

            # Guided ticker-specific reduce with underweight guard.
            for t in list(guided_H.keys()):
                if units.get(t, 0.0) <= 0:
                    guided_H.pop(t, None)
                    guided_target_active.pop(t, None)
                    guided_pmax.pop(t, None)
                    continue

                row_t = frames[t].iloc[i]
                close_t = float(row_t["close"])
                high_t = float(row_t["high"])
                H_t = guided_H[t]
                avg_t = avg_entry[t]

                if not guided_target_active.get(t, False):
                    observe = avg_t + cfg5.rebound_observe_ratio * (H_t - avg_t)
                    if high_t >= observe:
                        guided_target_active[t] = True
                        guided_pmax[t] = max(high_t, close_t)
                        trades.append({
                            "date": date,
                            "policy": "5",
                            "action": "GUIDED_TARGET_ZONE_REACHED",
                            "ticker": t,
                            "price": observe,
                            "avg_entry": avg_t,
                            "active_H": H_t,
                            "fee": 0.0,
                        })

                if guided_target_active.get(t, False):
                    guided_pmax[t] = max(guided_pmax.get(t, -math.inf), high_t, close_t)
                    trailing_stop = guided_pmax[t] * (1 - cfg5.trailing_drawdown)
                    hit_trailing = close_t <= trailing_stop
                    below_vwma10 = (
                        cfg5.sell_on_close_below_vwma10
                        and pd.notna(row_t.get("vwma10", np.nan))
                        and close_t < float(row_t["vwma10"])
                    )

                    if hit_trailing or below_vwma10:
                        reduce_fraction = max_reduce_fraction_to_respect_lower_band(
                            r,
                            units,
                            t,
                            target_weights,
                            cfg5.max_underweight_ratio,
                            cfg5.guided_reduce_fraction,
                        )

                        if reduce_fraction <= 0:
                            trades.append({
                                "date": date,
                                "policy": "5",
                                "action": "SKIP_REDUCE_UNDERWEIGHT",
                                "ticker": t,
                                "reason": "ticker_at_or_below_lower_target_band",
                                "target_weight": target_weights.get(t, 0.0),
                                "current_weight": current_weights(r, units).get(t, 0.0),
                                "fee": 0.0,
                            })
                            guided_target_active[t] = False
                            guided_pmax[t] = -math.inf
                        else:
                            units, closed_units, realized, trade_fee = close_ticker_fraction(
                                r, t, units, avg_entry, reduce_fraction, cfg
                            )
                            balance += realized - trade_fee

                            trades.append({
                                "date": date,
                                "policy": "5",
                                "action": "RATIO_GUARDED_GUIDED_REDUCE",
                                "ticker": t,
                                "reason": "TRAILING_STOP" if hit_trailing else "BELOW_VWMA10",
                                "requested_fraction": cfg5.guided_reduce_fraction,
                                "actual_fraction": reduce_fraction,
                                "units_closed": closed_units,
                                "realized_pnl": realized,
                                "fee": trade_fee,
                                "balance_after": balance,
                                "target_weight": target_weights.get(t, 0.0),
                                "current_weight_after": current_weights(r, units).get(t, 0.0),
                            })

                            guided_target_active[t] = False
                            guided_pmax[t] = -math.inf

        state = account_state(r, balance, units, avg_entry, cfg5.margin_rate)
        row = {
            "date": date,
            "Policy 5: Ratio-Guarded Guided Add/Reduce": state["equity"],
            "policy_5_balance": balance,
            "policy_5_notional": state["notional"],
            "policy_5_unrealized": state["unrealized"],
            "policy_5_used_margin": state["used_margin"],
            "policy_5_free_margin": state["free_margin"],
            "policy_5_margin_usage": state["margin_usage"],
            "policy_5_exposure_ratio": state["exposure_ratio"],
        }
        row.update(unit_snapshot("policy_5", units))
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(trades)


# ============================================================
# Plotting/output
# ============================================================

def build_share_counts_table(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    p3: pd.DataFrame,
    p4: pd.DataFrame,
    p5: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for df, prefix in [
        (p1, "policy_1_units_"),
        (p2, "policy_2_units_"),
        (p3, "policy_3_units_"),
        (p4, "policy_4_units_"),
        (p5, "policy_5_units_"),
    ]:
        cols = ["date"] + [c for c in df.columns if c.startswith(prefix)]
        parts.append(df[cols].copy())

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="date", how="outer")

    return out.sort_values("date").ffill().fillna(0).reset_index(drop=True)


def plot_share_counts(
    share_counts: pd.DataFrame,
    policy_prefix: str,
    title: str,
    path: str,
) -> None:
    cols = [c for c in share_counts.columns if c.startswith(policy_prefix)]
    if not cols:
        return

    plt.figure(figsize=(13, 6))

    for c in cols:
        ticker = c.replace(policy_prefix, "")
        series = share_counts[c]
        # Skip invisible all-zero lines, but keep policy 1 constants.
        if series.abs().max() == 0:
            continue
        plt.plot(share_counts["date"], series, label=ticker)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Share / Contract Count")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_all_share_count_panels(share_counts: pd.DataFrame, cfg: BacktestConfig) -> None:
    specs = [
        ("policy_1_units_", "Policy 1 Share Counts"),
        ("policy_2_units_", "Policy 2 Share Counts"),
        ("policy_3_units_", "Policy 3 Share Counts"),
        ("policy_4_units_", "Policy 4 Share Counts"),
        ("policy_5_units_", "Policy 5 Share Counts"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)

    for ax, (prefix, title) in zip(axes, specs):
        cols = [c for c in share_counts.columns if c.startswith(prefix)]
        for c in cols:
            ticker = c.replace(prefix, "")
            series = share_counts[c]
            if series.abs().max() == 0:
                continue
            ax.plot(share_counts["date"], series, label=ticker)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.legend(ncol=3, fontsize=7)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plot_dir, "all_policies_share_counts_v12.png"), dpi=160)
    plt.close()


def build_cumulative_fee_table(result: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative commission-fee table for Policy 1-4.

    Policy 1 is buy-and-hold in this script and has no simulated trade events,
    so its cumulative fee is zero unless you later add explicit opening/closing fees.
    """
    fee_table = result[["date"]].copy()

    for policy_id in ["1", "2", "3", "4", "5"]:
        fee_table[f"Policy {policy_id}: Cumulative Commission Fee"] = 0.0

    if trades is None or trades.empty or "fee" not in trades.columns or "policy" not in trades.columns:
        return fee_table

    fee_events = trades[["date", "policy", "fee"]].copy()
    fee_events["date"] = pd.to_datetime(fee_events["date"])
    fee_events["policy"] = fee_events["policy"].astype(str)
    fee_events["fee"] = pd.to_numeric(fee_events["fee"], errors="coerce").fillna(0.0)

    for policy_id in ["1", "2", "3", "4", "5"]:
        events = (
            fee_events[fee_events["policy"] == policy_id]
            .groupby("date", as_index=False)["fee"]
            .sum()
            .rename(columns={"fee": f"Policy {policy_id}: Cumulative Commission Fee"})
        )

        if events.empty:
            continue

        fee_table = fee_table.drop(columns=[f"Policy {policy_id}: Cumulative Commission Fee"]).merge(
            events,
            on="date",
            how="left",
        )
        fee_table[f"Policy {policy_id}: Cumulative Commission Fee"] = (
            fee_table[f"Policy {policy_id}: Cumulative Commission Fee"]
            .fillna(0.0)
            .cumsum()
        )

    return fee_table


def plot_cumulative_fees(fee_table: pd.DataFrame, cfg: BacktestConfig) -> None:
    cols = [
        "Policy 1: Cumulative Commission Fee",
        "Policy 2: Cumulative Commission Fee",
        "Policy 3: Cumulative Commission Fee",
        "Policy 4: Cumulative Commission Fee",
        "Policy 5: Cumulative Commission Fee",
    ]

    # Combined plot.
    plt.figure(figsize=(12, 6))
    for col in cols:
        if col in fee_table.columns:
            plt.plot(fee_table["date"], fee_table[col], label=col)
    plt.title("Cumulative Commission Fees by Policy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Commission Fee")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plot_dir, "all_policies_cumulative_commission_fees_v12.png"), dpi=160)
    plt.close()

    # Separate plot for each policy.
    for col in cols:
        if col not in fee_table.columns:
            continue

        policy_id = col.split(":")[0].replace("Policy ", "")
        plt.figure(figsize=(12, 5))
        plt.plot(fee_table["date"], fee_table[col], label=col)
        plt.title(col)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Commission Fee")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.plot_dir, f"policy_{policy_id}_cumulative_commission_fee_v10.png"), dpi=160)
        plt.close()


def plot_single(result: pd.DataFrame, col: str, title: str, path: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(result["date"], result[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_panels(result: pd.DataFrame, cfg: BacktestConfig) -> None:
    cols = [
        "Policy 1: Hold Portfolio",
        "Policy 2: Profit Pyramid + De-risk",
        "Policy 3: Sequential Single-Ticker Dip + VWMA",
        "Policy 4: Policy2 + Policy3 Guided Add/Reduce",
        "Policy 5: Ratio-Guarded Guided Add/Reduce",
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 17), sharex=True)

    for ax, col in zip(axes, cols):
        ax.plot(result["date"], result[col], label=col)
        ax.set_title(col)
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plot_dir, "all_policies_separate_panels_v12.png"), dpi=160)
    plt.close()


POLICY_NAMES = {
    "1": "Policy 1: Hold Portfolio",
    "2": "Policy 2: Profit Pyramid + De-risk",
    "3": "Policy 3: Sequential Single-Ticker Dip + VWMA",
    "4": "Policy 4: Policy2 + Policy3 Guided Add/Reduce",
    "5": "Policy 5: Ratio-Guarded Guided Add/Reduce",
}

POLICY_SHORT_NAMES = {
    "1": "P1 Hold",
    "2": "P2 Pyramid",
    "3": "P3 Sequential",
    "4": "P4 Guided",
    "5": "P5 Ratio Guard",
}


def units_from_row(row: pd.Series, prefix: str) -> dict[str, float]:
    return {t: float(row.get(f"{prefix}{t}", 0.0)) for t in all_tickers()}


def compute_turnover_and_weight_metrics(
    share_counts: pd.DataFrame,
    closes: pd.DataFrame,
    policy_id: str,
    target_weights: dict[str, float],
) -> dict:
    prefix = f"policy_{policy_id}_units_"

    merged = share_counts[["date"] + [c for c in share_counts.columns if c.startswith(prefix)]].merge(
        closes,
        on="date",
        how="left",
    ).sort_values("date").ffill().dropna().reset_index(drop=True)

    if merged.empty:
        return {
            "Turnover": 0.0,
            "Max Single Ticker Weight": 0.0,
            "Average Target-Weight Deviation": 0.0,
            "Final Target-Weight Deviation": 0.0,
        }

    initial_equity_ref = None
    total_trade_value = 0.0
    max_single_weight = 0.0
    deviations = []
    prev_units = None

    for _, row in merged.iterrows():
        units = units_from_row(row, prefix)

        notionals = {t: abs(units.get(t, 0.0)) * float(row[t]) for t in all_tickers() if t in row}
        total_notional = sum(notionals.values())

        if initial_equity_ref is None:
            initial_equity_ref = total_notional if total_notional > 0 else 1.0

        if prev_units is not None:
            step_trade_value = 0.0
            for t in all_tickers():
                if t in row:
                    delta_units = abs(units.get(t, 0.0) - prev_units.get(t, 0.0))
                    step_trade_value += delta_units * float(row[t])
            total_trade_value += step_trade_value

        prev_units = units

        if total_notional > 0:
            weights = {t: notionals.get(t, 0.0) / total_notional for t in all_tickers()}
        else:
            weights = {t: 0.0 for t in all_tickers()}

        max_single_weight = max(max_single_weight, max(weights.values()) if weights else 0.0)

        # 0.5 * L1 distance between current and target weights.
        # This gives a 0-100% portfolio-weight drift measure.
        deviation = 0.5 * sum(abs(weights.get(t, 0.0) - target_weights.get(t, 0.0)) for t in all_tickers())
        deviations.append(deviation)

    turnover = total_trade_value / initial_equity_ref if initial_equity_ref and initial_equity_ref > 0 else 0.0

    return {
        "Turnover": float(turnover),
        "Max Single Ticker Weight": float(max_single_weight),
        "Average Target-Weight Deviation": float(np.mean(deviations) if deviations else 0.0),
        "Final Target-Weight Deviation": float(deviations[-1] if deviations else 0.0),
    }


def compute_advanced_metrics(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    fee_table: pd.DataFrame,
    share_counts: pd.DataFrame,
    closes: pd.DataFrame,
    init_units: dict[str, float],
) -> pd.DataFrame:
    target_weights = target_weights_from_initial(closes.iloc[0], init_units)

    rows = []

    for policy_id in ["1", "2", "3", "4", "5"]:
        policy_name = POLICY_NAMES[policy_id]
        summary_row = summary[summary["Policy"] == policy_name].iloc[0]

        initial_equity = float(summary_row["Initial Equity"])
        final_equity = float(summary_row["Final Equity"])
        net_profit = final_equity - initial_equity

        fee_col = f"Policy {policy_id}: Cumulative Commission Fee"
        total_commission = float(fee_table[fee_col].iloc[-1]) if fee_col in fee_table.columns and not fee_table.empty else 0.0

        gross_profit_proxy = max(net_profit, 0.0) + total_commission
        commission_to_gross_profit = (
            total_commission / gross_profit_proxy
            if gross_profit_proxy > 0
            else np.nan
        )

        weight_metrics = compute_turnover_and_weight_metrics(
            share_counts=share_counts,
            closes=closes,
            policy_id=policy_id,
            target_weights=target_weights,
        )

        rows.append({
            "Policy ID": policy_id,
            "Policy": policy_name,
            "Policy Short": POLICY_SHORT_NAMES[policy_id],
            "Initial Equity": initial_equity,
            "Final Equity": final_equity,
            "Total Return": float(summary_row["Total Return"]),
            "Max Drawdown": float(summary_row["Max Drawdown"]),
            "Peak Equity": float(summary_row["Peak Equity"]),
            "Total Commission": total_commission,
            "Commission / Gross Profit": commission_to_gross_profit,
            **weight_metrics,
        })

    return pd.DataFrame(rows)


def plot_metric_bar(
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: str,
    percent: bool = False,
    absolute: bool = False,
) -> None:
    values = metrics[metric].copy()
    if absolute:
        values = values.abs()
    if percent:
        plot_values = values * 100.0
    else:
        plot_values = values

    plt.figure(figsize=(10, 5))
    plt.bar(metrics["Policy Short"], plot_values)
    plt.title(title)
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_metric_plots(metrics: pd.DataFrame, cfg: BacktestConfig) -> list[str]:
    plot_specs = [
        ("Final Equity", "Final Equity by Policy", "Final Equity", "metric_final_equity_v12.png", False, False),
        ("Max Drawdown", "Max Drawdown by Policy", "Max Drawdown (%)", "metric_max_drawdown_v12.png", True, True),
        ("Total Commission", "Total Commission by Policy", "Total Commission", "metric_total_commission_v12.png", False, False),
        ("Commission / Gross Profit", "Commission / Gross Profit by Policy", "Commission / Gross Profit (%)", "metric_commission_to_gross_profit_v12.png", True, False),
        ("Turnover", "Turnover by Policy", "Turnover (x Initial Equity)", "metric_turnover_v12.png", False, False),
        ("Max Single Ticker Weight", "Max Single Ticker Weight by Policy", "Max Single Ticker Weight (%)", "metric_max_single_ticker_weight_v12.png", True, False),
        ("Average Target-Weight Deviation", "Average Target-Weight Deviation by Policy", "Average Deviation (%)", "metric_avg_target_weight_deviation_v12.png", True, False),
        ("Final Target-Weight Deviation", "Final Target-Weight Deviation by Policy", "Final Deviation (%)", "metric_final_target_weight_deviation_v12.png", True, False),
    ]

    generated = []

    for metric, title, ylabel, filename, percent, absolute in plot_specs:
        path = os.path.join(cfg.plot_dir, filename)
        plot_metric_bar(metrics, metric, title, ylabel, path, percent=percent, absolute=absolute)
        generated.append(filename)

    # Grid view.
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for ax, (metric, title, ylabel, filename, percent, absolute) in zip(axes, plot_specs):
        values = metrics[metric].copy()
        if absolute:
            values = values.abs()
        if percent:
            values = values * 100.0

        ax.bar(metrics["Policy Short"], values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    grid_filename = "metric_comparison_grid_v12.png"
    plt.savefig(os.path.join(cfg.plot_dir, grid_filename), dpi=160)
    plt.close()
    generated.append(grid_filename)

    return generated


def format_metric_table_for_markdown(metrics: pd.DataFrame) -> str:
    columns = [
        "Policy Short",
        "Final Equity",
        "Total Return",
        "Max Drawdown",
        "Total Commission",
        "Commission / Gross Profit",
        "Turnover",
        "Max Single Ticker Weight",
        "Average Target-Weight Deviation",
        "Final Target-Weight Deviation",
    ]

    rows = []
    for _, row in metrics.iterrows():
        rows.append([
            row["Policy Short"],
            f"{row['Final Equity']:,.2f}",
            f"{row['Total Return']:.2%}",
            f"{row['Max Drawdown']:.2%}",
            f"{row['Total Commission']:,.2f}",
            "" if pd.isna(row["Commission / Gross Profit"]) else f"{row['Commission / Gross Profit']:.2%}",
            f"{row['Turnover']:.2f}x",
            f"{row['Max Single Ticker Weight']:.2%}",
            f"{row['Average Target-Weight Deviation']:.2%}",
            f"{row['Final Target-Weight Deviation']:.2%}",
        ])

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([header, separator] + body)


def generate_report_md(
    metrics: pd.DataFrame,
    metric_images: list[str],
    cfg: BacktestConfig,
) -> str:
    report_path = os.path.join(cfg.report_dir, "report.md")

    best_final = metrics.sort_values("Final Equity", ascending=False).iloc[0]
    lowest_commission = metrics.sort_values("Total Commission", ascending=True).iloc[0]
    lowest_weight_drift = metrics.sort_values("Average Target-Weight Deviation", ascending=True).iloc[0]

    lines = []
    lines.append("# Portfolio Policy Comparison Report")
    lines.append("")
    lines.append("## Backtest Configuration")
    lines.append("")
    lines.append(f"- Start: `{cfg.start}`")
    lines.append(f"- End: `{cfg.end}`")
    lines.append(f"- Interval: `{cfg.interval}`")
    lines.append(f"- Commission rate: `{cfg.commission_rate:.4%}`")
    lines.append(f"- Random adverse slippage max: `{cfg.random_slippage_max_rate:.4%}`")
    lines.append(f"- Random seed: `{cfg.random_seed}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Highest final equity: **{best_final['Policy Short']}** ({best_final['Final Equity']:,.2f}).")
    lines.append(f"- Lowest commission cost: **{lowest_commission['Policy Short']}** ({lowest_commission['Total Commission']:,.2f}).")
    lines.append(f"- Lowest average target-weight deviation: **{lowest_weight_drift['Policy Short']}** ({lowest_weight_drift['Average Target-Weight Deviation']:.2%}).")
    lines.append("")
    lines.append("Policy 5 is designed to preserve target position ratios better than Policy 4 by skipping buys when a ticker is already overweight and limiting sells when a ticker is already underweight.")
    lines.append("")
    lines.append("## Metrics Table")
    lines.append("")
    lines.append(format_metric_table_for_markdown(metrics))
    lines.append("")
    lines.append("## Metric Comparison Charts")
    lines.append("")

    for image in metric_images:
        title = image.replace("_", " ").replace(".png", "").title()
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}](../plots/{image})")
        lines.append("")

    lines.append("## Existing Strategy Charts")
    lines.append("")
    existing_images = [
        "all_policies_separate_panels_v12.png",
        "all_policies_share_counts_v12.png",
        "all_policies_cumulative_commission_fees_v12.png",
    ]
    for image in existing_images:
        title = image.replace("_", " ").replace(".png", "").title()
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}](../plots/{image})")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first = result.iloc[0]
    last = result.iloc[-1]

    for col in [
        "Policy 1: Hold Portfolio",
        "Policy 2: Profit Pyramid + De-risk",
        "Policy 3: Sequential Single-Ticker Dip + VWMA",
        "Policy 4: Policy2 + Policy3 Guided Add/Reduce",
        "Policy 5: Ratio-Guarded Guided Add/Reduce",
    ]:
        s = result[col]
        peak = s.cummax()
        dd = s / peak - 1

        rows.append({
            "Policy": col,
            "Initial Equity": float(first[col]),
            "Final Equity": float(last[col]),
            "Total Return": float(last[col] / first[col] - 1),
            "Max Drawdown": float(dd.min()),
            "Peak Equity": float(s.max()),
        })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio Policy 5 v12 evaluator")
    parser.add_argument(
        "--engine-position-source",
        choices=sorted(ENGINE_POSITION_SOURCES),
        default=None,
        help=(
            "Initial live-engine position source. Overrides ENGINE_POSITION_SOURCE. "
            "Defaults to ig when IG is configured, otherwise manual for paper/backtest."
        ),
    )
    parser.add_argument(
        "--engine-mode",
        default=os.getenv("ENGINE_MODE", "paper"),
        help="Engine mode used when defaulting the position source (paper, backtest, or live).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.engine_position_source:
        os.environ["ENGINE_POSITION_SOURCE"] = args.engine_position_source
    os.environ["ENGINE_MODE"] = args.engine_mode

    cfg = BacktestConfig(
        start="2026-04-05",
        end="2026-05-31",
        interval="15m",
        initial_account_equity=None,

        # v9 cost model:
        # 0.001 = 0.10% commission.
        commission_rate=0.0010,

        # Fixed slippage. Keep zero because random adverse slippage is enabled below.
        slippage_rate=0.0000,

        # Max random adverse fill delta: under 0.1%.
        random_slippage_max_rate=0.0010,
        random_seed=42,
    )

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    cfg2 = Policy2Config(
        margin_rate=0.20,
        max_gross_exposure_to_equity=3.0,
        max_margin_usage=0.50,
        add_trigger_return=0.05,
        profit_reinvest_fraction=0.50,
        derisk_drawdown_from_peak=0.08,
        derisk_fraction=0.35,
        min_profit_to_derisk=0.02,
        require_new_peak_after_derisk=True,
        enable_hard_exit=False,
        hard_exit_drawdown_from_peak=0.20,
    )

    cfg3 = Policy3SequentialConfig(
        confirm_bars=3,
        max_lows_per_plan=3,
        invest_fraction_of_cash=0.50,
        max_volume_ratio_for_buy=1.20,
        min_vwma20_slope_5bars=-0.02,
        min_vwma50_slope_5bars=-0.03,
        rebound_observe_ratio=0.75,
        trailing_drawdown=0.04,
        sell_on_close_below_vwma10=True,
        signal_priority="best_expected_rebound",
    )

    cfg4 = Policy4Config(
        margin_rate=0.20,
        max_gross_exposure_to_equity=3.0,
        max_margin_usage=0.50,
        profit_reinvest_fraction=0.50,
        min_profit_to_add=0.01,
        confirm_bars=3,
        max_lows_per_plan=3,
        max_volume_ratio_for_buy=1.20,
        min_vwma20_slope_5bars=-0.02,
        min_vwma50_slope_5bars=-0.03,
        signal_priority="best_expected_rebound",
        rebound_observe_ratio=0.75,
        trailing_drawdown=0.04,
        sell_on_close_below_vwma10=True,
        guided_reduce_fraction=0.35,
        enable_portfolio_derisk=True,
        derisk_drawdown_from_peak=0.08,
        derisk_fraction=0.20,
        min_profit_to_derisk=0.02,
        require_new_peak_after_derisk=True,
        enable_hard_exit=False,
        hard_exit_drawdown_from_peak=0.20,
    )

    cfg5 = Policy5Config(
        margin_rate=0.20,
        max_gross_exposure_to_equity=3.0,
        max_margin_usage=0.50,
        profit_reinvest_fraction=0.50,
        min_profit_to_add=0.01,
        confirm_bars=3,
        max_lows_per_plan=3,
        max_volume_ratio_for_buy=1.20,
        min_vwma20_slope_5bars=-0.02,
        min_vwma50_slope_5bars=-0.03,
        signal_priority="best_expected_rebound",
        rebound_observe_ratio=0.75,
        trailing_drawdown=0.04,
        sell_on_close_below_vwma10=True,
        guided_reduce_fraction=0.35,
        max_overweight_ratio=0.25,
        max_underweight_ratio=0.25,
        enable_portfolio_derisk=True,
        derisk_drawdown_from_peak=0.08,
        derisk_fraction=0.20,
        min_profit_to_derisk=0.02,
        require_new_peak_after_derisk=True,
        enable_hard_exit=False,
        hard_exit_drawdown_from_peak=0.20,
    )

    ensure_dirs(cfg)
    data = download_all(cfg)
    closes = align_closes(data)
    init_units = initial_units_from_positions()

    p1 = simulate_policy_1_hold(closes, init_units)
    p2, trades2 = simulate_policy_2_profit_pyramid_derisk(closes, init_units, cfg, cfg2)
    p3, trades3 = simulate_policy_3_sequential(data, closes, init_units, cfg, cfg3)
    p4, trades4 = simulate_policy_4_signal_guided_pyramid(data, closes, init_units, cfg, cfg4)
    p5, trades5 = simulate_policy_5_ratio_guarded_guided_pyramid(data, closes, init_units, cfg, cfg5)

    result = (
        p1.merge(p2[["date", "Policy 2: Profit Pyramid + De-risk"]], on="date", how="outer")
          .merge(p3[["date", "Policy 3: Sequential Single-Ticker Dip + VWMA"]], on="date", how="outer")
          .merge(p4[["date", "Policy 4: Policy2 + Policy3 Guided Add/Reduce"]], on="date", how="outer")
          .merge(p5[["date", "Policy 5: Ratio-Guarded Guided Add/Reduce"]], on="date", how="outer")
          .sort_values("date")
          .ffill()
          .dropna()
          .reset_index(drop=True)
    )

    trades = pd.concat([trades2, trades3, trades4, trades5], ignore_index=True, sort=False)
    summary = summarize(result)
    share_counts = build_share_counts_table(p1, p2, p3, p4, p5)
    fee_table = build_cumulative_fee_table(result, trades)
    advanced_metrics = compute_advanced_metrics(result, summary, fee_table, share_counts, closes, init_units)

    plot_single(
        result,
        "Policy 1: Hold Portfolio",
        "Policy 1: Hold Current Position List",
        os.path.join(cfg.plot_dir, "policy_1_hold_portfolio_v12.png"),
    )
    plot_single(
        result,
        "Policy 2: Profit Pyramid + De-risk",
        "Policy 2: Profit Pyramid + Partial De-risk",
        os.path.join(cfg.plot_dir, "policy_2_profit_pyramid_derisk_v12.png"),
    )
    plot_single(
        result,
        "Policy 3: Sequential Single-Ticker Dip + VWMA",
        "Policy 3: Sequential Single-Ticker Dip + VWMA",
        os.path.join(cfg.plot_dir, "policy_3_sequential_v12.png"),
    )
    plot_single(
        result,
        "Policy 4: Policy2 + Policy3 Guided Add/Reduce",
        "Policy 4: Policy2 + Policy3 Guided Add/Reduce",
        os.path.join(cfg.plot_dir, "policy_4_guided_pyramid_v12.png"),
    )
    plot_single(
        result,
        "Policy 5: Ratio-Guarded Guided Add/Reduce",
        "Policy 5: Ratio-Guarded Guided Add/Reduce",
        os.path.join(cfg.plot_dir, "policy_5_ratio_guarded_guided_pyramid_v12.png"),
    )
    plot_panels(result, cfg)

    plot_share_counts(
        share_counts,
        "policy_1_units_",
        "Policy 1: Share Counts Over Time",
        os.path.join(cfg.plot_dir, "policy_1_share_counts_v12.png"),
    )
    plot_share_counts(
        share_counts,
        "policy_2_units_",
        "Policy 2: Share Counts Over Time",
        os.path.join(cfg.plot_dir, "policy_2_share_counts_v12.png"),
    )
    plot_share_counts(
        share_counts,
        "policy_3_units_",
        "Policy 3: Share Counts Over Time",
        os.path.join(cfg.plot_dir, "policy_3_share_counts_v12.png"),
    )
    plot_share_counts(
        share_counts,
        "policy_4_units_",
        "Policy 4: Share Counts Over Time",
        os.path.join(cfg.plot_dir, "policy_4_share_counts_v12.png"),
    )
    plot_share_counts(
        share_counts,
        "policy_5_units_",
        "Policy 5: Share Counts Over Time",
        os.path.join(cfg.plot_dir, "policy_5_share_counts_v12.png"),
    )
    plot_all_share_count_panels(share_counts, cfg)
    plot_cumulative_fees(fee_table, cfg)

    metric_images = generate_metric_plots(advanced_metrics, cfg)
    report_path = generate_report_md(advanced_metrics, metric_images, cfg)

    result.to_csv("portfolio_policy_equity_v12.csv", index=False)
    trades.to_csv("portfolio_policy_trades_v12.csv", index=False)
    summary.to_csv("portfolio_policy_summary_v12.csv", index=False)
    share_counts.to_csv("portfolio_policy_share_counts_v12.csv", index=False)
    fee_table.to_csv("portfolio_policy_cumulative_fees_v12.csv", index=False)
    advanced_metrics.to_csv("portfolio_policy_advanced_metrics_v12.csv", index=False)

    position_map = pd.DataFrame([
        {"Market": m, "Ticker": MARKET_MAP[m], "Size": s}
        for m, s in MARKET_POSITIONS
    ])
    position_map.to_csv("portfolio_position_mapping_v12.csv", index=False)

    print("\nBacktest config:")
    print(f"  start={cfg.start}, end={cfg.end}, interval={cfg.interval}")
    print("  margin_rate=20% for Policy 2 and Policy 4")
    print(f"  commission_rate={cfg.commission_rate:.4%}")
    print(f"  random_slippage_max_rate={cfg.random_slippage_max_rate:.4%}")
    print(f"  random_seed={cfg.random_seed}")

    print("\nPosition mapping:")
    print(position_map.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False, formatters={
        "Initial Equity": lambda x: f"{x:,.2f}",
        "Final Equity": lambda x: f"{x:,.2f}",
        "Total Return": lambda x: f"{x:.2%}",
        "Max Drawdown": lambda x: f"{x:.2%}",
        "Peak Equity": lambda x: f"{x:,.2f}",
    }))

    print("\nAdvanced metrics:")
    print(advanced_metrics.to_string(index=False, formatters={
        "Initial Equity": lambda x: f"{x:,.2f}",
        "Final Equity": lambda x: f"{x:,.2f}",
        "Total Return": lambda x: f"{x:.2%}",
        "Max Drawdown": lambda x: f"{x:.2%}",
        "Peak Equity": lambda x: f"{x:,.2f}",
        "Total Commission": lambda x: f"{x:,.2f}",
        "Commission / Gross Profit": lambda x: "" if pd.isna(x) else f"{x:.2%}",
        "Turnover": lambda x: f"{x:.2f}x",
        "Max Single Ticker Weight": lambda x: f"{x:.2%}",
        "Average Target-Weight Deviation": lambda x: f"{x:.2%}",
        "Final Target-Weight Deviation": lambda x: f"{x:.2%}",
    }))

    print("\nPolicy 4 trades:")
    print(trades[trades["policy"].astype(str) == "4"].to_string(index=False))

    print("\nSaved:")
    print("  portfolio_policy_equity_v12.csv")
    print("  portfolio_policy_trades_v12.csv")
    print("  portfolio_policy_summary_v12.csv")
    print("  portfolio_policy_share_counts_v12.csv")
    print("  portfolio_policy_cumulative_fees_v12.csv")
    print("  portfolio_policy_advanced_metrics_v12.csv")
    print("  portfolio_position_mapping_v12.csv")
    print(f"  {cfg.plot_dir}/policy_1_hold_portfolio_v12.png")
    print(f"  {cfg.plot_dir}/policy_2_profit_pyramid_derisk_v12.png")
    print(f"  {cfg.plot_dir}/policy_3_sequential_v12.png")
    print(f"  {cfg.plot_dir}/policy_4_guided_pyramid_v12.png")
    print(f"  {cfg.plot_dir}/policy_5_ratio_guarded_guided_pyramid_v12.png")
    print(f"  {cfg.plot_dir}/all_policies_separate_panels_v12.png")
    print(f"  {cfg.plot_dir}/policy_1_share_counts_v12.png")
    print(f"  {cfg.plot_dir}/policy_2_share_counts_v12.png")
    print(f"  {cfg.plot_dir}/policy_3_share_counts_v12.png")
    print(f"  {cfg.plot_dir}/policy_4_share_counts_v12.png")
    print(f"  {cfg.plot_dir}/policy_5_share_counts_v12.png")
    print(f"  {cfg.plot_dir}/all_policies_share_counts_v12.png")
    print(f"  {cfg.plot_dir}/all_policies_cumulative_commission_fees_v12.png")
    print(f"  {cfg.plot_dir}/policy_1_cumulative_commission_fee_v12.png")
    print(f"  {cfg.plot_dir}/policy_2_cumulative_commission_fee_v12.png")
    print(f"  {cfg.plot_dir}/policy_3_cumulative_commission_fee_v12.png")
    print(f"  {cfg.plot_dir}/policy_4_cumulative_commission_fee_v12.png")
    print(f"  {cfg.plot_dir}/policy_5_cumulative_commission_fee_v12.png")
    print(f"  {cfg.plot_dir}/metric_comparison_grid_v12.png")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
