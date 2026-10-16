"""Project 1 - UBOS District Population Forecaster.

Contains the domain class :class:`DistrictPopulation`, an abstract
:class:`Forecaster` base class with three concrete models (linear trend,
exponential/CAGR and a Fibonacci-ratio model), error metrics, model selection,
bootstrap prediction intervals and the school-classroom planning calculation.
"""
from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Data (thousands of people, illustrative). Mbarara and Jinja are added by us.
# --------------------------------------------------------------------------
YEARS = list(range(2015, 2025))
DATA: Dict[str, List[float]] = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    # Two extra districts chosen by us (illustrative, thousands)
    "Mbarara": [195, 200, 207, 214, 222, 229, 237, 246, 254, 263],
    "Jinja": [480, 490, 505, 520, 535, 555, 570, 590, 610, 630],
}


# --------------------------------------------------------------------------
# Domain class
# --------------------------------------------------------------------------
class DistrictPopulation:
    """Population time series (in thousands) of one district.

    Parameters
    ----------
    name : str
        District name.
    years : sequence of int
        Calendar years, strictly increasing.
    population : sequence of float
        Population in thousands, same length as ``years``, never negative.
    """

    def __init__(self, name: str, years: Sequence[int], population: Sequence[float]) -> None:
        years_arr = np.asarray(years, dtype=float)
        pop_arr = np.asarray(population, dtype=float)
        if years_arr.ndim != 1 or pop_arr.ndim != 1:
            raise ValueError("years and population must be one-dimensional")
        if len(years_arr) == 0:
            raise ValueError("years and population must not be empty")
        if len(years_arr) != len(pop_arr):
            raise ValueError("years and population must have equal length")
        if np.any(np.isnan(pop_arr)) or np.any(np.isnan(years_arr)):
            raise ValueError("NaN values are not allowed")
        if np.any(pop_arr < 0):
            raise ValueError("population must not contain negative values")
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("years must be strictly increasing")
        self.name = name
        self.years = years_arr.astype(int)
        self.population = pop_arr

    # -- dunder methods ----------------------------------------------------
    def __repr__(self) -> str:
        return (f"DistrictPopulation(name={self.name!r}, years={self.years[0]}-{self.years[-1]}, "
                f"n={len(self)}, last={self.population[-1]:.0f}k)")

    def __len__(self) -> int:
        return len(self.years)

    # -- descriptive statistics -------------------------------------------
    def stats_statistics(self) -> Dict[str, float]:
        """Mean, median, variance, st.dev. with the ``statistics`` module (sample: n-1)."""
        data = self.population.tolist()
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),   # sample variance (n-1)
            "stdev": statistics.stdev(data),          # sample st.dev.
        }

    def stats_numpy(self, ddof: int = 0) -> Dict[str, float]:
        """Same statistics with NumPy. ``ddof=0`` (default) gives the *population* variance."""
        return {
            "mean": float(np.mean(self.population)),
            "median": float(np.median(self.population)),
            "variance": float(np.var(self.population, ddof=ddof)),
            "stdev": float(np.std(self.population, ddof=ddof)),
        }

    # -- growth ------------------------------------------------------------
    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth rates (fractions), length n-1."""
        if np.any(self.population[:-1] == 0):
            raise ZeroDivisionError("cannot compute growth from a zero population")
        return np.diff(self.population) / self.population[:-1]

    def cagr(self) -> float:
        """Compound Annual Growth Rate: (last/first)**(1/periods) - 1."""
        if len(self) < 2:
            raise ValueError("CAGR needs at least two observations")
        if self.population[0] <= 0:
            raise ValueError("CAGR needs a positive starting value")
        periods = self.years[-1] - self.years[0]
        return float((self.population[-1] / self.population[0]) ** (1.0 / periods) - 1.0)

    def split(self, last_train_year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(train_years, train_pop, test_years, test_pop)``."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("split year leaves an empty train or test set")
        return self.years[mask], self.population[mask], self.years[~mask], self.population[~mask]


# --------------------------------------------------------------------------
# Forecasters
# --------------------------------------------------------------------------
class Forecaster(ABC):
    """Abstract forecaster: ``fit`` on a time series, then ``predict(horizon)``."""

    name: str = "Forecaster"

    def __init__(self) -> None:
        self.years_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, years: Sequence[int], values: Sequence[float]) -> "Forecaster":
        years_arr = np.asarray(years, dtype=float)
        y = np.asarray(values, dtype=float)
        if len(years_arr) != len(y):
            raise ValueError("years and values must have equal length")
        if len(y) < 2:
            raise ValueError("need at least two observations to fit")
        self.years_, self.y_ = years_arr, y
        self._fit(years_arr, y)
        return self

    def _check_fitted(self) -> None:
        if self.y_ is None:
            raise RuntimeError("call fit() before predict()")

    @abstractmethod
    def _fit(self, years: np.ndarray, y: np.ndarray) -> None:
        """Estimate model parameters."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` yearly values after the training data."""

    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values for the training years."""

    def residuals(self) -> np.ndarray:
        self._check_fitted()
        return self.y_ - self.fitted_values()


class LinearTrendForecaster(Forecaster):
    """(a) Straight-line trend y = a*t + b fitted with ``np.polyfit``."""

    name = "Linear trend"

    def _fit(self, years, y):
        self.coef_ = np.polyfit(years, y, deg=1)

    def predict(self, horizon):
        self._check_fitted()
        future = self.years_[-1] + np.arange(1, horizon + 1)
        return np.polyval(self.coef_, future)

    def fitted_values(self):
        self._check_fitted()
        return np.polyval(self.coef_, self.years_)


class CAGRForecaster(Forecaster):
    """(b) Exponential growth at the training-set CAGR."""

    name = "Exponential (CAGR)"

    def _fit(self, years, y):
        if y[0] <= 0:
            raise ValueError("CAGR model needs a positive first value")
        periods = years[-1] - years[0]
        self.g_ = (y[-1] / y[0]) ** (1.0 / periods) - 1.0

    def predict(self, horizon):
        self._check_fitted()
        return self.y_[-1] * (1.0 + self.g_) ** np.arange(1, horizon + 1)

    def fitted_values(self):
        self._check_fitted()
        return self.y_[0] * (1.0 + self.g_) ** (self.years_ - self.years_[0])


def fibonacci(n: int) -> List[int]:
    """First ``n`` Fibonacci numbers: 1, 1, 2, 3, 5, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq: List[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


class FibonacciRatioForecaster(Forecaster):
    """(c) Scale the last value by *successive Fibonacci ratios* F(k+1)/F(k).

    Two modes:

    ``mode="raw"``
        Literal reading of the old assignment:
        ``y[T+h] = y[T+h-1] * F(k+h+1)/F(k+h)``. The multiplier is about 1.6, i.e. 60 %
        growth per year, so the forecast explodes.

    ``mode="calibrated"`` (default)
        The ratio is normalised by the golden ratio (its limit):
        ``growth_h = g * [F(k+h+1)/F(k+h)] / phi`` where ``g`` is the training CAGR.
        The Fibonacci ratios then only add a small, damped oscillation around the
        historical growth rate.
    """

    name = "Fibonacci ratio"
    PHI = (1 + math.sqrt(5)) / 2

    def __init__(self, mode: str = "calibrated", start: int = 3) -> None:
        super().__init__()
        if mode not in {"raw", "calibrated"}:
            raise ValueError("mode must be 'raw' or 'calibrated'")
        if start < 1:
            raise ValueError("start must be >= 1")
        self.mode, self.start = mode, start

    def _ratio(self, h: int) -> float:
        fib = fibonacci(self.start + h + 2)
        k = self.start + h - 1          # 0-based index of F(k)
        return fib[k + 1] / fib[k]

    def _multiplier(self, h: int) -> float:
        if self.mode == "raw":
            return self._ratio(h)
        return 1.0 + self.g_ * self._ratio(h) / self.PHI

    def _fit(self, years, y):
        if y[0] <= 0:
            raise ValueError("Fibonacci model needs a positive first value")
        periods = years[-1] - years[0]
        self.g_ = (y[-1] / y[0]) ** (1.0 / periods) - 1.0

    def predict(self, horizon):
        self._check_fitted()
        out, level = [], self.y_[-1]
        for h in range(1, horizon + 1):
            level = level * self._multiplier(h)
            out.append(level)
        return np.array(out)

    def fitted_values(self):
        """One-step-ahead fitted values (first point equals the data)."""
        self._check_fitted()
        fit = [self.y_[0]]
        for t in range(1, len(self.y_)):
            fit.append(self.y_[t - 1] * self._multiplier(t))
        return np.array(fit)


# --------------------------------------------------------------------------
# Metrics and validation
# --------------------------------------------------------------------------
def _pair(actual, predicted) -> Tuple[np.ndarray, np.ndarray]:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("inputs must be non-empty and the same shape")
    return a, p


def mae(actual, predicted) -> float:
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted) -> float:
    """Mean absolute percentage error (in %). Undefined if any actual value is 0."""
    a, p = _pair(actual, predicted)
    if np.any(a == 0):
        raise ZeroDivisionError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def default_models() -> List[Forecaster]:
    return [LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()]


def evaluate_models(district: DistrictPopulation, last_train_year: int = 2021,
                    models: Sequence[Forecaster] | None = None) -> List[Dict[str, float]]:
    """Train on years <= ``last_train_year`` and score on the remaining years."""
    ty, tp, _, test_pop = district.split(last_train_year)
    rows = []
    for model in (models if models is not None else default_models()):
        m = copy.deepcopy(model).fit(ty, tp)
        pred = m.predict(len(test_pop))
        rows.append({"model": m.name, "MAE": mae(test_pop, pred),
                     "RMSE": rmse(test_pop, pred), "MAPE_%": mape(test_pop, pred)})
    return rows


def select_best(rows: List[Dict[str, float]], key: str = "RMSE") -> str:
    """Name of the model with the smallest ``key`` error."""
    if not rows:
        raise ValueError("no results to select from")
    return min(rows, key=lambda r: r[key])["model"]


def model_by_name(name: str) -> Forecaster:
    for m in default_models():
        if m.name == name:
            return m
    raise KeyError(name)


# --------------------------------------------------------------------------
# Bootstrap prediction intervals
# --------------------------------------------------------------------------
def bootstrap_intervals(model: Forecaster, years, values, horizon: int,
                        n_boot: int = 1000, seed: int = 42,
                        level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction interval.

    1. Fit the model and take its in-sample residuals.
    2. Resample residuals with replacement, add them to the fitted values to create a
       pseudo-series, refit, forecast, then add one more resampled residual for future noise.
    3. Take the (1-level)/2 and (1+level)/2 percentiles across ``n_boot`` runs.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    base = copy.deepcopy(model).fit(years, values)
    fitted, resid = base.fitted_values(), base.residuals()
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        y_star = fitted + rng.choice(resid, size=len(resid), replace=True)
        y_star = np.maximum(y_star, 1e-9)
        m = copy.deepcopy(model).fit(years, y_star)
        sims[b] = m.predict(horizon) + rng.choice(resid, size=horizon, replace=True)
    alpha = (1 - level) / 2
    return np.percentile(sims, 100 * alpha, axis=0), np.percentile(sims, 100 * (1 - alpha), axis=0)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def classrooms_needed(population_thousands: float, school_age_share: float = 0.18,
                      pupils_per_class: int = 53) -> int:
    """Classrooms required (rounded up) for a population given in thousands."""
    if population_thousands < 0:
        raise ValueError("population cannot be negative")
    if not 0 <= school_age_share <= 1:
        raise ValueError("school_age_share must be in [0, 1]")
    if pupils_per_class <= 0:
        raise ValueError("pupils_per_class must be positive")
    pupils = population_thousands * 1000 * school_age_share
    return int(math.ceil(pupils / pupils_per_class))


def additional_classrooms(current_pop_k: float, future_pop_k: float, **kw) -> int:
    """Extra classrooms needed between two population levels (never negative)."""
    return max(0, classrooms_needed(future_pop_k, **kw) - classrooms_needed(current_pop_k, **kw))


def load_districts() -> Dict[str, DistrictPopulation]:
    return {name: DistrictPopulation(name, YEARS, vals) for name, vals in DATA.items()}
