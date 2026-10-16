import numpy as np
import pytest

from src.taxi import (ExponentialSmoothingForecaster, LinearTrendForecaster, MovingAverageForecaster,
                      Route, SeasonalNaiveForecaster, backtest_table, best_model_for, equilibrium,
                      market_position, rolling_origin_mae, simulate_weekly_series, tune_alpha,
                      vehicles_needed)


def test_route_revenue_and_stats():
    r = Route("T", [10, 20, 30], 2000)
    assert r.daily_revenue().tolist() == [20000, 40000, 60000] and r.total_revenue() == 120000
    d = r.describe()
    assert d["mean"] == 20 and d["variance"] == 100 and d["stdev"] == 10


@pytest.mark.parametrize("args", [("x", [], 1000), ("x", [1, -2], 1000), ("x", [1, 2], 0)])
def test_route_validation(args):
    with pytest.raises(ValueError):
        Route(*args)


def test_equilibrium_hand_calculation():
    # 120 - 0.02P = 10 + 0.03P  ->  P* = 2200, Q* = 76
    p, q = equilibrium()
    assert p == pytest.approx(2200) and q == pytest.approx(76)
    pos = market_position(2000)
    assert "below" in pos["state"] and pos["excess_demand"] == pytest.approx(10)
    assert "above" in market_position(2500)["state"]


def test_equilibrium_parallel_curves_edge_case():
    with pytest.raises(ValueError):
        equilibrium(b=-0.03, d=0.03)       # slopes cancel -> singular


def test_moving_average_known_value():
    assert MovingAverageForecaster(3).fit([1, 2, 3, 4, 5]).predict(1)[0] == pytest.approx(4.0)
    assert MovingAverageForecaster(3).fit([1, 2]).predict(1)[0] == pytest.approx(1.5)   # short history ok


def test_ses_alpha_extremes_and_validation():
    y = [10, 20, 30]
    assert ExponentialSmoothingForecaster(1.0).fit(y).predict(1)[0] == 30       # alpha=1 -> last value
    assert ExponentialSmoothingForecaster(0.5).fit(y).predict(2).tolist() == [22.5, 22.5]
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(0)


def test_linear_trend_exact_and_needs_two_points():
    assert LinearTrendForecaster().fit([2, 4, 6, 8]).predict(2) == pytest.approx([10, 12])
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([5])


def test_seasonal_naive_repeats_last_season():
    y = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
    assert SeasonalNaiveForecaster(7).fit(y).predict(3).tolist() == [4, 5, 6]
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(7).fit([1, 2, 3])


def test_predict_before_fit_and_bad_horizon():
    with pytest.raises(RuntimeError):
        MovingAverageForecaster().predict(1)
    with pytest.raises(ValueError):
        MovingAverageForecaster().fit([1, 2, 3]).predict(0)


def test_rolling_origin_perfect_forecast_has_zero_error():
    assert rolling_origin_mae(SeasonalNaiveForecaster(2), [5, 7, 5, 7, 5, 7, 5, 7], first_origin=2) == 0
    assert rolling_origin_mae(LinearTrendForecaster(), [1, 2, 3, 4, 5, 6], first_origin=3) == pytest.approx(0)
    with pytest.raises(ValueError):
        rolling_origin_mae(MovingAverageForecaster(), [1, 2, 3], first_origin=3)


def test_tune_alpha_constant_series_and_table():
    alpha, err = tune_alpha([5] * 10)
    assert err == 0 and 0 < alpha <= 1
    r = Route("R", [35, 40, 42, 50, 55, 60, 48, 52, 47, 45], 2000)
    table = backtest_table([r])
    assert len(table["R"]) == 3 and all(v >= 0 for v in table["R"].values())
    assert best_model_for(r).predict(1).shape == (1,)


def test_vehicle_planning_rounding():
    assert vehicles_needed(0) == 0
    assert vehicles_needed(112) == 2                 # 112*1.15 = 128.8 > 112 -> round up to 2
    assert vehicles_needed(97) == 1                  # 97*1.15 = 111.55 <= 112
    with pytest.raises(ValueError):
        vehicles_needed(-1)


def test_simulated_series_reproducible_and_seasonal_beats_ma():
    a, b = simulate_weekly_series(60, seed=3), simulate_weekly_series(60, seed=3)
    assert np.array_equal(a, b) and len(a) == 60 and (a >= 0).all()
    assert rolling_origin_mae(SeasonalNaiveForecaster(7), a, 14) < rolling_origin_mae(MovingAverageForecaster(3), a, 14)
