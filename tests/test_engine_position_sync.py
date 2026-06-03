import importlib
import sys
import types


def _install_import_stubs():
    for name in ("numpy", "pandas", "yfinance"):
        sys.modules.setdefault(name, types.ModuleType(name))
    matplotlib = sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    pyplot = sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
    setattr(matplotlib, "pyplot", pyplot)


def _load_module():
    _install_import_stubs()
    sys.modules.pop("portfolio_policy_eval_v12_report", None)
    return importlib.import_module("portfolio_policy_eval_v12_report")


class FakeIGClient:
    def __init__(self):
        self.logged_in = False
        self.open_positions_called = False

    def login(self):
        self.logged_in = True

    def open_positions(self):
        self.open_positions_called = True
        return {
            "positions": [
                {
                    "market": {"epic": "EPIC.AVGO"},
                    "position": {"direction": "BUY", "size": "1.25"},
                },
                {
                    "market": {"epic": "EPIC.MU"},
                    "position": {"direction": "SELL", "size": 3},
                },
            ]
        }


def test_run_trading_engine_uses_ig_units_instead_of_manual_market_positions():
    module = _load_module()
    fake_ig = FakeIGClient()

    engine = module.run_trading_engine(
        ig_client=fake_ig,
        position_source="ig",
        epic_map={"AVGO": "EPIC.AVGO", "MU": "EPIC.MU"},
    )

    assert fake_ig.logged_in is True
    assert fake_ig.open_positions_called is True
    assert engine.units["AVGO"] == 1.25
    assert engine.units["MU"] == -3.0
    assert engine.units["AVGO"] != module.initial_units_from_positions()["AVGO"]
    assert engine.units["MU"] != module.initial_units_from_positions()["MU"]

    sync_actions = [a for a in engine.actions if a["action"] == "CURRENT_POSITIONS_SYNCED"]
    assert sync_actions == [
        {
            "action": "CURRENT_POSITIONS_SYNCED",
            "source": "ig",
            "tickers": "AVGO,MU",
            "counts": {"AVGO": 1.25, "MU": -3.0},
            "fee": 0.0,
        }
    ]
