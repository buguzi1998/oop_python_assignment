"""Project 3 - Lake Victoria Fish Stock & Export Risk Model.

* :func:`fibonacci_stock` - the flawed baseline
* :class:`FishStock` - discrete logistic growth with harvesting
* :class:`PriceModel` - bounded seeded random walk for the price (UGX/kg)
* :class:`RiskAssessor` - CV-based risk class, Monte-Carlo VaR, scenario analysis
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

import numpy as np

TONNES_TO_KG = 1000.0


def fibonacci_stock(n: int = 15) -> List[int]:
    """First ``n`` Fibonacci numbers used as a (biologically unrealistic) 'stock'."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq: List[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def closed_season_mask(weeks: int, closed_weeks: int = 8, start_week: int = 20,
                       weeks_per_year: int = 52) -> np.ndarray:
    """Boolean mask, True in weeks when fishing is closed (repeats every year)."""
    if closed_weeks < 0 or closed_weeks > weeks_per_year:
        raise ValueError("closed_weeks must be between 0 and 52")
    week_of_year = np.arange(weeks) % weeks_per_year
    return (week_of_year >= start_week) & (week_of_year < start_week + closed_weeks)


class FishStock:
    """Discrete logistic growth with proportional harvesting.

    ``N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)``  (tonnes, one step = one week)

    Parameters
    ----------
    r : intrinsic growth rate per week (0 < r)
    K : carrying capacity in tonnes (K > 0)
    n0 : initial stock in tonnes (0 <= n0)
    h : harvest rate per week, the fraction of the stock caught (0 <= h <= 1)
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, n0: float = 4_000.0, h: float = 0.10) -> None:
        if r <= 0 or K <= 0:
            raise ValueError("r and K must be positive")
        if n0 < 0:
            raise ValueError("initial stock cannot be negative")
        if not 0 <= h <= 1:
            raise ValueError("harvest rate h must be in [0, 1]")
        self.r, self.K, self.n0, self.h = float(r), float(K), float(n0), float(h)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K}, n0={self.n0}, h={self.h})"

    # -- theory ------------------------------------------------------------
    @property
    def msy(self) -> float:
        """Theoretical maximum sustainable yield, rK/4 (tonnes per week)."""
        return self.r * self.K / 4.0

    @property
    def h_msy(self) -> float:
        """Harvest rate that attains MSY: r/2."""
        return self.r / 2.0

    def equilibrium(self) -> float:
        """Non-zero equilibrium stock ``K (1 - h/r)`` (0 if h >= r: stock collapses)."""
        return max(0.0, self.K * (1.0 - self.h / self.r))

    def equilibrium_yield(self) -> float:
        """Long-run weekly catch (tonnes) at equilibrium: ``h * N*``."""
        return self.h * self.equilibrium()

    # -- simulation --------------------------------------------------------
    def simulate(self, weeks: int = 52, closed: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """Simulate ``weeks`` steps.

        Returns dict with ``stock`` (length weeks+1) and ``harvest`` (tonnes caught in each
        week, length ``weeks``). ``closed`` optionally switches harvesting off in some weeks.
        """
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        if closed is not None and len(closed) != weeks:
            raise ValueError("closed mask must have length equal to weeks")
        stock = np.empty(weeks + 1)
        harvest = np.zeros(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            h_t = 0.0 if (closed is not None and closed[t]) else self.h
            harvest[t] = h_t * stock[t]
            nxt = stock[t] + self.r * stock[t] * (1 - stock[t] / self.K) - harvest[t]
            stock[t + 1] = max(nxt, 0.0)
        return {"stock": stock, "harvest": harvest}


class PriceModel:
    """Weekly price (UGX/kg) as a bounded Gaussian random walk (seeded).

    ``p[t+1] = clip(p[t] + N(0, sigma), lower, upper)``
    """

    def __init__(self, start: float = 12_000.0, sigma: float = 400.0,
                 lower: float = 9_000.0, upper: float = 16_000.0, seed: int = 42) -> None:
        if not lower <= start <= upper:
            raise ValueError("start price must lie within [lower, upper]")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.start, self.sigma, self.lower, self.upper, self.seed = start, sigma, lower, upper, seed

    def __repr__(self) -> str:
        return f"PriceModel(start={self.start}, sigma={self.sigma}, bounds=({self.lower}, {self.upper}))"

    def simulate(self, weeks: int = 52, rng: np.random.Generator | None = None) -> np.ndarray:
        """One price path of ``weeks`` weekly prices (the first one is ``start``)."""
        return self.simulate_paths(1, weeks, rng)[0]

    def simulate_paths(self, n_paths: int, weeks: int = 52,
                       rng: np.random.Generator | None = None) -> np.ndarray:
        """``(n_paths, weeks)`` array of independent price paths."""
        if n_paths < 1 or weeks < 1:
            raise ValueError("n_paths and weeks must be >= 1")
        rng = np.random.default_rng(self.seed) if rng is None else rng
        steps = rng.normal(0.0, self.sigma, size=(n_paths, weeks))
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + steps[:, t], self.lower, self.upper)
        return prices


def weekly_revenue(harvest_tonnes: np.ndarray, price_per_kg: np.ndarray) -> np.ndarray:
    """Revenue in UGX = harvest (kg) x price (UGX/kg)."""
    h, p = np.asarray(harvest_tonnes, float), np.asarray(price_per_kg, float)
    if h.shape[-1] != p.shape[-1]:
        raise ValueError("harvest and price must cover the same number of weeks")
    return h * TONNES_TO_KG * p


def describe_revenue(revenue: Sequence[float]) -> Dict[str, float]:
    """Mean, median, variance (UGX^2), st.dev. (UGX) and CV (unit-free) with ``statistics``."""
    r = [float(x) for x in revenue]
    if len(r) < 2:
        raise ValueError("need at least two observations")
    mean = statistics.mean(r)
    sd = statistics.stdev(r)
    return {"mean": mean, "median": statistics.median(r), "variance": statistics.variance(r),
            "stdev": sd, "cv": sd / mean if mean else float("inf")}


class RiskAssessor:
    """Classify revenue risk from the coefficient of variation and simulate VaR.

    Rule (documented, adjustable): CV < ``low`` -> "Low"; CV < ``high`` -> "Moderate";
    otherwise "High". CV = st.dev./mean is unit-free, so the same thresholds work at any
    revenue scale (unlike a raw variance limit such as 50,000).
    """

    def __init__(self, low: float = 0.10, high: float = 0.25) -> None:
        if not 0 < low < high:
            raise ValueError("thresholds must satisfy 0 < low < high")
        self.low, self.high = low, high

    def classify(self, cv: float) -> str:
        if cv < 0 or np.isnan(cv):
            raise ValueError("cv must be a non-negative number")
        return "Low" if cv < self.low else ("Moderate" if cv < self.high else "High")

    @staticmethod
    def sustainability(stock: FishStock, final_stock: float) -> str:
        """'Overfished' if the harvest rate exceeds r/2 (beyond MSY) or the stock ends
        materially below K/2 (the MSY biomass, 1 % tolerance); otherwise 'Sustainable'."""
        if stock.h >= stock.r:
            return "Collapse"
        if stock.h > stock.h_msy or final_stock < 0.99 * stock.K / 2:   # 1 % tolerance
            return "Overfished"
        return "Sustainable"

    def monte_carlo(self, stock: FishStock, price_model: PriceModel, n_paths: int = 2000,
                    weeks: int = 52, seed: int = 123, closed: np.ndarray | None = None) -> Dict[str, object]:
        """Annual revenue distribution over ``n_paths`` random price paths."""
        if n_paths < 1000:
            raise ValueError("use at least 1,000 price paths")
        rng = np.random.default_rng(seed)
        harvest = stock.simulate(weeks, closed)["harvest"]
        prices = price_model.simulate_paths(n_paths, weeks, rng)
        annual = (harvest * TONNES_TO_KG * prices).sum(axis=1)
        p5 = float(np.percentile(annual, 5))
        mean = float(annual.mean())
        cv = float(annual.std(ddof=1) / mean) if mean > 0 else float("inf")
        return {"annual": annual, "mean": mean, "p5": p5, "var_5": mean - p5,
                "cv": cv, "risk": self.classify(cv) if np.isfinite(cv) else "High"}

    def scenarios(self, harvest_rates: Sequence[float], r: float = 0.4, K: float = 10_000.0,
                  n0: float = 4_000.0, weeks: int = 52, price_model: PriceModel | None = None,
                  n_paths: int = 2000, seed: int = 123) -> List[Dict[str, object]]:
        """Run several harvest rates and summarise each one."""
        pm = price_model or PriceModel()
        out = []
        for h in harvest_rates:
            fs = FishStock(r, K, n0, h)
            sim = fs.simulate(weeks)
            single_price = pm.simulate(weeks)
            rev = weekly_revenue(sim["harvest"], single_price)
            mc = self.monte_carlo(fs, pm, n_paths, weeks, seed)
            weekly_cv = describe_revenue(rev)["cv"]
            out.append({"h": h, "final_stock_t": float(sim["stock"][-1]),
                        "total_catch_t": float(sim["harvest"].sum()),
                        "total_revenue_UGX": float(rev.sum()),
                        "weekly_revenue_CV": weekly_cv,
                        "annual_revenue_CV_MC": mc["cv"], "risk_class": mc["risk"],
                        "VaR5_UGX": mc["var_5"], "sustainability": self.sustainability(fs, sim["stock"][-1]),
                        "equilibrium_yield_t_per_wk": fs.equilibrium_yield(), "MSY_t_per_wk": fs.msy})
        return out


def closed_season_effect(h: float, years: int = 5, closed_weeks: int = 8, start_week: int = 20,
                         price_model: PriceModel | None = None, seed: int = 7, **fish_kw) -> Dict[str, float]:
    """Compare total revenue with and without a closed season over ``years`` years,
    using the same price path in both runs."""
    if years < 1:
        raise ValueError("years must be >= 1")
    weeks = 52 * years
    pm = price_model or PriceModel()
    prices = pm.simulate(weeks, np.random.default_rng(seed))
    fs = FishStock(h=h, **fish_kw)
    open_run = fs.simulate(weeks)
    mask = closed_season_mask(weeks, closed_weeks, start_week)
    closed_run = fs.simulate(weeks, mask)
    rev_open = weekly_revenue(open_run["harvest"], prices).sum()
    rev_closed = weekly_revenue(closed_run["harvest"], prices).sum()
    return {"h": h, "revenue_open": float(rev_open), "revenue_closed": float(rev_closed),
            "change_pct": float((rev_closed - rev_open) / rev_open * 100) if rev_open else float("nan"),
            "final_stock_open": float(open_run["stock"][-1]),
            "final_stock_closed": float(closed_run["stock"][-1])}
