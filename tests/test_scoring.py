import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("radar_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strong_multi_signal_scores_high():
    m = load_module()
    row = pd.Series({
        "volume_ratio": 3.2, "return_5d": 8, "return_20d": 15, "return_60d": 25,
        "above_ma20": True, "distance_52w_high": -1, "forecast_revision": 15,
        "sales_growth": 18, "profit_growth": 30, "consecutive_growth_years": 4,
        "record_profit": True, "operating_margin": 16, "dividend_yield": 4.8,
        "payout_ratio": 45, "equity_ratio": 60, "free_cash_flow_bil": 20,
        "market_cap_bil": 900, "dividend_cuts_5y": 0,
    })
    got = m.score_row(row)
    assert got["overlap"] >= 4
    assert got["total_score"] >= 85


def test_scores_are_bounded():
    m = load_module()
    got = m.score_row(pd.Series(dtype=object))
    assert all(0 <= got[key] <= 100 for key in ["money_score", "revision_score", "breakout_score", "growth_score", "dividend_score"])
