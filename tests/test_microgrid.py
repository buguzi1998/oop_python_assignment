import numpy as np
import pytest

from src.microgrid import (HybridMicroGrid, MicroGrid, SingularSystemError, generate_demand_csv,
                           interactive_day, load_demand_csv, parse_demand, prompt_demand)


def test_solve_day_matches_hand_calculation():
    # x = (2*D2 - D1)/5 ; y = (4*D1 - 3*D2)/5  ->  D1=40, D2=45 gives x=10, y=5
    x, y = MicroGrid().solve_day(40, 45)
    assert x == pytest.approx(10.0) and y == pytest.approx(5.0)


def test_determinant_and_condition():
    g = MicroGrid()
    assert g.determinant() == pytest.approx(-5.0)
    assert g.condition_number() > 1 and g.is_well_posed()


def test_singular_matrix_rejected():
    g = MicroGrid([[1, 2], [2, 4]])
    with pytest.raises(SingularSystemError):
        g.solve_day(1, 2)


def test_negative_or_wrong_size_demand_rejected():
    g = MicroGrid()
    with pytest.raises(ValueError):
        g.solve_day(-1, 5)
    with pytest.raises(ValueError):
        g.solve_day(1)


def test_loop_and_vectorised_agree():
    d = np.random.default_rng(0).uniform(30, 60, size=(2, 30))
    g = MicroGrid()
    assert np.allclose(g.solve_loop(d), g.solve_vectorised(d))


def test_infeasible_day_detected_and_nnls_is_nonnegative():
    g = MicroGrid()
    d = np.array([[40.0, 20.0], [45.0, 45.0]])    # day 2: D1/D2 < 0.75 -> negative battery
    res = g.dispatch(d, "nnls")
    assert res.infeasible.tolist() == [False, True]
    assert np.all(res.final >= 0)
    assert np.allclose(res.final[:, 0], res.raw[:, 0])        # feasible day untouched
    assert np.all(np.clip(g.dispatch(d, "clip").final, 0, None) >= 0)


def test_parse_demand():
    assert parse_demand(" 12.5 ") == 12.5
    for bad in ["", "   ", "abc", "-3", "nan", "inf"]:
        with pytest.raises(ValueError):
            parse_demand(bad)


def test_prompt_reprompts_until_valid():
    answers = iter(["", "abc", "-4", "42"])
    assert prompt_demand("D1", input_fn=lambda _: next(answers)) == 42.0


def test_interactive_day():
    answers = iter(["40", "oops", "45"])
    out = interactive_day(MicroGrid(), input_fn=lambda _: next(answers))
    assert np.allclose(out, [10, 5])


def test_csv_roundtrip_is_seeded(tmp_path):
    p1 = generate_demand_csv(tmp_path / "a.csv", 30, seed=7)
    p2 = generate_demand_csv(tmp_path / "b.csv", 30, seed=7)
    a, b = load_demand_csv(p1), load_demand_csv(p2)
    assert a.shape == (2, 30) and np.array_equal(a, b)


def test_csv_bad_rows(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("day,d1_kwh,d2_kwh\n1,40,abc\n")
    with pytest.raises(ValueError):
        load_demand_csv(bad)
    empty = tmp_path / "empty.csv"
    empty.write_text("day,d1_kwh,d2_kwh\n")
    with pytest.raises(ValueError):
        load_demand_csv(empty)


def test_describe_and_cost():
    s = MicroGrid.describe([1, 2, 3, 4])
    assert s["mean"] == 2.5 and s["variance"] == pytest.approx(5 / 3)
    assert MicroGrid.daily_cost(np.array([10.0]), np.array([5.0]))[0] == 10 * 150 + 5 * 450


def test_hybrid_solves_3x3_and_detects_dependence():
    h = HybridMicroGrid()
    d = np.array([50.0, 45.0, 30.0])
    assert np.allclose(h.A @ h.solve_day(*d), d)
    dep = HybridMicroGrid([[3, 2, 1], [4, 1, 0], [7, 3, 1]])     # row3 = row1 + row2
    assert dep.rank() == 2
    with pytest.raises(SingularSystemError):
        dep.solve_day(50, 45, 95)
    sol, how = dep.solve_day_safe(50, 45, 95)
    assert "least-squares" in how and np.all(np.isfinite(sol))


def test_sensitivity_respects_condition_bound():
    g = MicroGrid()
    out = g.sensitivity([40, 45], pct=0.05, n_draws=500, seed=1)
    assert out["max_amplification"] <= out["cond"] + 1e-9
    assert out["solutions"].shape == (2, 500)
