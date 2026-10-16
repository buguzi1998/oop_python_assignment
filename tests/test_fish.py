import numpy as np
import pytest

from src.fish import (FishStock, PriceModel, RiskAssessor, closed_season_effect,
                      closed_season_mask, describe_revenue, fibonacci_stock, weekly_revenue)


def test_fibonacci_stock():
    assert fibonacci_stock(10) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert len(fibonacci_stock(15)) == 15 and fibonacci_stock(0) == []


def test_logistic_step_matches_hand_calculation():
    fs = FishStock(r=0.4, K=10_000, n0=4_000, h=0.10)
    # N1 = 4000 + 0.4*4000*(1-0.4) - 0.1*4000 = 4000 + 960 - 400 = 4560
    assert fs.simulate(1)["stock"][1] == pytest.approx(4560.0)


def test_stock_converges_to_equilibrium_and_msy():
    fs = FishStock(h=0.10)
    end = fs.simulate(300)["stock"][-1]
    assert end == pytest.approx(fs.equilibrium(), rel=1e-3)          # K(1-h/r) = 7500
    assert FishStock().msy == 1000.0 and FishStock().h_msy == 0.2
    best = FishStock(h=0.2)
    assert best.equilibrium_yield() == pytest.approx(best.msy)        # yield peaks at h = r/2


def test_stock_never_negative_and_collapse_when_h_ge_r():
    sim = FishStock(h=0.9, n0=10).simulate(50)["stock"]
    assert np.all(sim >= 0) and sim[-1] < 1e-6
    assert FishStock(h=0.5).equilibrium() == 0.0


@pytest.mark.parametrize("kw", [dict(r=0), dict(K=-1), dict(n0=-5), dict(h=1.5), dict(h=-0.1)])
def test_fishstock_validation(kw):
    with pytest.raises(ValueError):
        FishStock(**kw)


def test_price_model_bounds_and_reproducibility():
    pm = PriceModel(seed=5)
    a, b = pm.simulate(52), PriceModel(seed=5).simulate(52)
    assert np.array_equal(a, b) and a[0] == 12_000
    assert a.min() >= 9_000 and a.max() <= 16_000
    paths = pm.simulate_paths(500, 52)
    assert paths.shape == (500, 52) and paths.min() >= 9_000 and paths.max() <= 16_000
    with pytest.raises(ValueError):
        PriceModel(start=20_000)


def test_revenue_and_describe():
    assert weekly_revenue(np.array([2.0]), np.array([10_000.0]))[0] == 2 * 1000 * 10_000
    d = describe_revenue([10, 20, 30])
    assert d["mean"] == 20 and d["variance"] == 100 and d["cv"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        describe_revenue([5])


def test_cv_is_scale_free_but_variance_is_not():
    base = [10, 12, 9, 11, 13]
    scaled = [x * 1000 for x in base]
    assert describe_revenue(base)["cv"] == pytest.approx(describe_revenue(scaled)["cv"])
    assert describe_revenue(scaled)["variance"] == pytest.approx(describe_revenue(base)["variance"] * 1e6)


def test_risk_classification_rule():
    ra = RiskAssessor()
    assert ra.classify(0.05) == "Low" and ra.classify(0.15) == "Moderate" and ra.classify(0.4) == "High"
    with pytest.raises(ValueError):
        ra.classify(-0.1)
    with pytest.raises(ValueError):
        RiskAssessor(0.3, 0.1)


def test_monte_carlo_var_and_minimum_paths():
    ra = RiskAssessor()
    out = ra.monte_carlo(FishStock(h=0.2), PriceModel(), n_paths=1000, seed=1)
    assert out["var_5"] > 0 and out["p5"] < out["mean"]
    assert len(out["annual"]) == 1000
    with pytest.raises(ValueError):
        ra.monte_carlo(FishStock(), PriceModel(), n_paths=100)


def test_scenarios_zero_harvest_gives_zero_revenue():
    rows = RiskAssessor().scenarios([0.0, 0.2], n_paths=1000)
    assert rows[0]["total_revenue_UGX"] == 0 and rows[0]["VaR5_UGX"] == 0
    assert rows[1]["total_revenue_UGX"] > 0


def test_closed_season_mask_and_effect():
    m = closed_season_mask(104, closed_weeks=8, start_week=20)
    assert m.sum() == 16 and m[20] and not m[28]
    assert not closed_season_mask(52, 0).any()
    res = closed_season_effect(0.3, years=5)
    assert res["final_stock_closed"] >= res["final_stock_open"] * 0.9
