import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_policy_eval_v12_report as report


def _market_data(price_by_ticker):
    dates = pd.date_range("2026-01-01", periods=3, freq="15min")
    closes = pd.DataFrame({"date": dates})
    data = {}
    for ticker in report.all_tickers():
        price = float(price_by_ticker.get(ticker, 100.0))
        closes[ticker] = price
        data[ticker] = pd.DataFrame(
            {
                "date": dates,
                "open": [price] * len(dates),
                "high": [price * 1.01] * len(dates),
                "low": [price * 0.99] * len(dates),
                "close": [price] * len(dates),
                "volume": [1000] * len(dates),
            }
        )
    return data, closes


class OneShotBuyScanner:
    def __init__(self, ticker, *_args):
        self.ticker = ticker
        self.consumed = False

    def update_and_get_signal(self, _row):
        if self.ticker != "AVGO" or self.consumed:
            return None
        return {
            "ticker": self.ticker,
            "trigger": 99.0,
            "active_H": 110.0,
            "level_idx": 0,
            "expected_rebound": 0.10,
            "vwma_discount": -0.01,
        }

    def consume_signal_level(self):
        self.consumed = True


def test_target_contract_counts_use_latest_prices_and_ratios():
    prices = pd.Series({ticker: 100.0 for ticker in report.all_tickers()})
    prices["AVGO"] = 200.0
    prices["MRVL"] = 50.0

    counts = report.target_contract_counts(
        prices,
        account_equity_or_allocation=1_000.0,
        target_contract_ratios={"AVGO": 6, "MRVL": 4},
    )

    assert counts["AVGO"] == 3.0  # (1000 * 60%) / 200
    assert counts["MRVL"] == 8.0  # (1000 * 40%) / 50


def test_policy5_zero_current_units_positive_starting_equity_buys_candidate(monkeypatch):
    data, closes = _market_data({"AVGO": 100.0, "MRVL": 50.0})
    zero_units = {ticker: 0.0 for ticker in report.all_tickers()}
    cfg = report.BacktestConfig(
        initial_account_equity=None,
        engine_starting_equity=10_000.0,
        engine_position_source="zero",
        engine_target_contract_ratios={"AVGO": 6, "MRVL": 4},
        commission_rate=0.0,
        slippage_rate=0.0,
        random_slippage_max_rate=0.0,
    )
    cfg5 = report.Policy5Config(
        margin_rate=0.20,
        max_gross_exposure_to_equity=3.0,
        max_margin_usage=0.50,
        min_profit_to_add=0.01,
        enable_portfolio_derisk=False,
        enable_hard_exit=False,
    )

    monkeypatch.setattr(report, "prepare_frames", lambda _data, _closes, _confirm_bars: data)
    monkeypatch.setattr(report, "SignalScannerState", OneShotBuyScanner)

    results, trades = report.simulate_policy_5_ratio_guarded_guided_pyramid(
        data, closes, zero_units, cfg, cfg5
    )

    buys = trades[trades["action"] == "RATIO_GUARDED_GUIDED_ADD"]
    assert not buys.empty
    first_buy = buys.iloc[0]
    assert first_buy["ticker"] == "AVGO"
    assert first_buy["reason"] == "first_entry_to_positive_target_count"
    assert first_buy["target_count"] > 0
    assert first_buy["current_count_before"] == 0
    assert results["policy_5_units_AVGO"].max() > 0
