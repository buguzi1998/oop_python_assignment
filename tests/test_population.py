import numpy as np
import pytest

from src.population import (CAGRForecaster, DistrictPopulation, FibonacciRatioForecaster,
                            LinearTrendForecaster, additional_classrooms, bootstrap_intervals,
                            classrooms_needed, evaluate_models, fibonacci, load_districts,
                            mae, mape, rmse, select_best)

YEARS = list(range(2015, 2025))


def make(vals=None):
    return DistrictPopulation("X", YEARS, vals or [100 * 1.05 ** i for i in range(10)])


# ---- DistrictPopulation ----------------------------------------------------
def test_district_valid_and_dunders():
    d = make()
    assert len(d) == 10
    assert "X" in repr(d)


@pytest.mark.parametrize("years,pop", [
    ([2015, 2016], [1, 2, 3]),        # unequal lengths
    ([2015, 2016], [1, -2]),           # negative value
    ([], []),                          # empty
    ([2016, 2015], [1, 2]),            # years not increasing
])
def test_district_validation_errors(years, pop):
    with pytest.raises(ValueError):
        DistrictPopulation("bad", years, pop)


def test_variance_ddof_difference():
    d = make([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s, n0, n1 = d.stats_statistics(), d.stats_numpy(0), d.stats_numpy(1)
    n = len(d)
    assert s["variance"] == pytest.approx(n1["variance"])                 # ddof=1 matches statistics
    assert n0["variance"] == pytest.approx(s["variance"] * (n - 1) / n)   # ddof=0 is smaller


def test_growth_and_cagr_known_values():
    d = make([100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171, 214.358881, 235.7947691])
    assert np.allclose(d.growth_rates(), 0.10)
    assert d.cagr() == pytest.approx(0.10)


def test_cagr_edge_cases():
    with pytest.raises(ValueError):
        DistrictPopulation("z", [2015, 2016], [0, 5]).cagr()
    with pytest.raises(ValueError):
        DistrictPopulation("one", [2015], [5]).cagr()


# ---- Forecasters -----------------------------------------------------------
def test_linear_forecaster_exact_on_line():
    y = 10 + 5 * np.arange(8)
    m = LinearTrendForecaster().fit(np.arange(2015, 2023), y)
    assert np.allclose(m.predict(3), [50, 55, 60])


def test_cagr_forecaster_exact_on_exponential():
    y = 100 * 1.07 ** np.arange(8)
    m = CAGRForecaster().fit(np.arange(2015, 2023), y)
    assert np.allclose(m.predict(2), [y[-1] * 1.07, y[-1] * 1.07 ** 2])


def test_fibonacci_sequence_and_raw_mode_explodes():
    assert fibonacci(8) == [1, 1, 2, 3, 5, 8, 13, 21]
    assert fibonacci(0) == []
    m = FibonacciRatioForecaster(mode="raw").fit(np.arange(2015, 2023), np.arange(1, 9) * 10.0)
    assert m.predict(5)[-1] > 5 * m.y_[-1]      # explosive


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(3)


# ---- Metrics ---------------------------------------------------------------
def test_metrics_known_values():
    a, p = [100, 200], [110, 180]
    assert mae(a, p) == 15
    assert rmse(a, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(a, p) == pytest.approx((10 + 10) / 2)


def test_metrics_edge_cases():
    with pytest.raises(ZeroDivisionError):
        mape([0, 1], [1, 1])
    with pytest.raises(ValueError):
        mae([], [])


# ---- Pipeline --------------------------------------------------------------
def test_evaluate_and_select_best():
    rows = evaluate_models(load_districts()["Kampala"])
    assert {r["model"] for r in rows} == {"Linear trend", "Exponential (CAGR)", "Fibonacci ratio"}
    assert select_best(rows) in {r["model"] for r in rows}
    with pytest.raises(ValueError):
        select_best([])


def test_bootstrap_interval_contains_point_forecast_and_is_reproducible():
    d = load_districts()["Gulu"]
    m = LinearTrendForecaster()
    lo, hi = bootstrap_intervals(m, d.years, d.population, 5, n_boot=200, seed=1)
    lo2, hi2 = bootstrap_intervals(m, d.years, d.population, 5, n_boot=200, seed=1)
    point = LinearTrendForecaster().fit(d.years, d.population).predict(5)
    assert np.all(lo <= point) and np.all(point <= hi)
    assert np.allclose(lo, lo2) and np.allclose(hi, hi2)


# ---- Planning --------------------------------------------------------------
def test_classrooms():
    assert classrooms_needed(1000) == int(np.ceil(1000 * 1000 * 0.18 / 53))
    assert classrooms_needed(0) == 0
    assert additional_classrooms(1000, 900) == 0       # shrinking population -> 0 extra
    with pytest.raises(ValueError):
        classrooms_needed(-1)
    with pytest.raises(ValueError):
        classrooms_needed(10, pupils_per_class=0)
