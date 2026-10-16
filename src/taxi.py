"""Project 5 - Taxi Route Revenue, Pricing & Fleet Planner."""
from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import linalg

ROUTES: Dict[str, Tuple[List[int], float]] = {
    "Kampala-Ntinda": ([35, 40, 42, 50, 55, 60, 48, 52, 47, 45], 2000.0),
    "Kampala-Entebbe": ([60, 58, 65, 70, 72, 80, 75, 68, 66, 64], 5000.0),
    "Kampala-Mukono": ([45, 47, 50, 49, 55, 62, 58, 53, 51, 50], 3000.0),
}


class Route:
    """Daily passenger counts and the fare (UGX) of one taxi route."""

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        p = np.asarray(passengers, dtype=float)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if np.any(np.isnan(p)) or np.any(p < 0):
            raise ValueError("passenger counts must be non-negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name, self.passengers, self.fare = name, p, float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare={self.fare:.0f})"

    def __len__(self) -> int:
        return len(self.passengers)

    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> Dict[str, float]:
        """Mean, variance and st.dev. of daily passengers (``statistics``, sample)."""
        if len(self) < 2:
            raise ValueError("need at least two days for variance")
        data = self.passengers.tolist()
        return {"mean": statistics.mean(data), "variance": statistics.variance(data),
                "stdev": statistics.stdev(data)}


# --------------------------------------------------------------------------
# Supply-demand equilibrium  Qd = a - b P,  Qs = c + d P
# --------------------------------------------------------------------------
def equilibrium(a: float = 120.0, b: float = 0.02, c: float = 10.0, d: float = 0.03) -> Tuple[float, float]:
    """Solve the 2x2 system  Q + bP = a  and  Q - dP = c  for ``(P, Q)``."""
    A = np.array([[b, 1.0], [-d, 1.0]])        # unknown vector [P, Q]
    rhs = np.array([a, c])
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("demand and supply curves are parallel: no unique equilibrium")
    P, Q = linalg.solve(A, rhs)
    return float(P), float(Q)


def market_position(fare: float, a: float = 120.0, b: float = 0.02, c: float = 10.0, d: float = 0.03) -> Dict[str, float | str]:
    """Compare a fare with equilibrium: excess demand > 0 means shortage (fare too low)."""
    p_star, q_star = equilibrium(a, b, c, d)
    qd, qs = a - b * fare, c + d * fare
    state = "below equilibrium (shortage)" if fare < p_star else (
        "above equilibrium (surplus)" if fare > p_star else "at equilibrium")
    return {"P_star": p_star, "Q_star": q_star, "Qd": qd, "Qs": qs, "excess_demand": qd - qs, "state": state}


# --------------------------------------------------------------------------
# Forecasters
# --------------------------------------------------------------------------
class Forecaster(ABC):
    """Base class: ``fit(history)`` then ``predict(horizon)``."""

    name = "Forecaster"

    def __init__(self) -> None:
        self.y_: np.ndarray | None = None

    def fit(self, history: Sequence[float]) -> "Forecaster":
        y = np.asarray(history, dtype=float)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("history must be a non-empty 1-D sequence")
        if np.any(np.isnan(y)):
            raise ValueError("history contains NaN")
        self.y_ = y
        self._fit(y)
        return self

    def _fit(self, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, horizon: int = 1) -> np.ndarray:
        """Forecast the next ``horizon`` days."""

    def _check(self, horizon: int) -> None:
        if self.y_ is None:
            raise RuntimeError("call fit() first")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")


class MovingAverageForecaster(Forecaster):
    """Mean of the last ``window`` observations (the original 3-day method)."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"{window}-day moving average"

    def predict(self, horizon: int = 1) -> np.ndarray:
        self._check(horizon)
        buf = list(self.y_)
        out = []
        for _ in range(horizon):
            f = float(np.mean(buf[-self.window:]))
            out.append(f)
            buf.append(f)
        return np.array(out)


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing: ``level = alpha*y + (1-alpha)*level``."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (alpha={alpha:g})"

    def _fit(self, y: np.ndarray) -> None:
        level = y[0]
        for v in y[1:]:
            level = self.alpha * v + (1 - self.alpha) * level
        self.level_ = float(level)

    def predict(self, horizon: int = 1) -> np.ndarray:
        self._check(horizon)
        return np.full(horizon, self.level_)


class LinearTrendForecaster(Forecaster):
    """Least-squares straight line through the history (needs >= 2 points)."""

    name = "Linear trend"

    def _fit(self, y: np.ndarray) -> None:
        if len(y) < 2:
            raise ValueError("linear trend needs at least two observations")
        self.coef_ = np.polyfit(np.arange(len(y)), y, 1)

    def predict(self, horizon: int = 1) -> np.ndarray:
        self._check(horizon)
        return np.polyval(self.coef_, len(self.y_) + np.arange(horizon))


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value from one season ago (default season = 7 days)."""

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"Seasonal naive (m={period})"

    def _fit(self, y: np.ndarray) -> None:
        if len(y) < self.period:
            raise ValueError("history shorter than one season")

    def predict(self, horizon: int = 1) -> np.ndarray:
        self._check(horizon)
        y = self.y_
        return np.array([y[len(y) - self.period + (h % self.period)] for h in range(horizon)])


# --------------------------------------------------------------------------
# Backtesting and tuning
# --------------------------------------------------------------------------
def rolling_origin_mae(model: Forecaster, series: Sequence[float], first_origin: int = 3) -> float:
    """Walk-forward one-step-ahead MAE.

    For each origin ``t`` from ``first_origin`` to ``n-1`` the model is fit on ``series[:t]``
    and forecasts ``series[t]``. With ``first_origin=3`` on 10 days this scores days 4-10.
    """
    y = np.asarray(series, dtype=float)
    if not 1 <= first_origin < len(y):
        raise ValueError("first_origin must leave at least one day to forecast")
    errors = []
    for t in range(first_origin, len(y)):
        m = copy.deepcopy(model).fit(y[:t])
        errors.append(abs(y[t] - m.predict(1)[0]))
    return float(np.mean(errors))


def tune_alpha(series: Sequence[float], grid: Sequence[float] | None = None,
               first_origin: int = 3) -> Tuple[float, float]:
    """Grid-search the SES alpha that minimises rolling-origin MAE. Returns ``(alpha, mae)``."""
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else grid
    scores = [(rolling_origin_mae(ExponentialSmoothingForecaster(a), series, first_origin), a) for a in grid]
    best_mae, best_alpha = min(scores)
    return float(best_alpha), float(best_mae)


def backtest_table(routes: Sequence[Route], first_origin: int = 3) -> Dict[str, Dict[str, float]]:
    """MAE (passengers) per model per route; SES alpha is tuned separately for each route."""
    table: Dict[str, Dict[str, float]] = {}
    for r in routes:
        alpha, _ = tune_alpha(r.passengers, first_origin=first_origin)
        models: List[Forecaster] = [MovingAverageForecaster(3), ExponentialSmoothingForecaster(alpha),
                                    LinearTrendForecaster()]
        table[r.name] = {m.name: rolling_origin_mae(m, r.passengers, first_origin) for m in models}
    return table


def best_model_for(route: Route, first_origin: int = 3) -> Forecaster:
    """Model with the lowest rolling-origin MAE, refit on the whole series."""
    alpha, _ = tune_alpha(route.passengers, first_origin=first_origin)
    models: List[Forecaster] = [MovingAverageForecaster(3), ExponentialSmoothingForecaster(alpha),
                                LinearTrendForecaster()]
    best = min(models, key=lambda m: rolling_origin_mae(m, route.passengers, first_origin))
    return best.fit(route.passengers)          # returned already fitted on the full history


# --------------------------------------------------------------------------
# Fleet planning
# --------------------------------------------------------------------------
def vehicles_needed(passengers: float, trips_per_day: int = 8, seats: int = 14, buffer: float = 0.15) -> int:
    """Vehicles required for a passenger demand, with a safety buffer, rounded *up*."""
    if passengers < 0:
        raise ValueError("passengers cannot be negative")
    if trips_per_day <= 0 or seats <= 0:
        raise ValueError("trips_per_day and seats must be positive")
    if buffer < 0:
        raise ValueError("buffer must be non-negative")
    capacity = trips_per_day * seats
    return int(math.ceil(passengers * (1 + buffer) / capacity - 1e-12))


# --------------------------------------------------------------------------
# Extension: 60-day seasonal simulation
# --------------------------------------------------------------------------
def simulate_weekly_series(days: int = 60, base: float = 50.0, noise: float = 3.0, seed: int = 42) -> np.ndarray:
    """Seeded daily passengers with a weekly pattern (day 0 = Monday): Fridays busy, Sundays quiet."""
    if days < 1:
        raise ValueError("days must be >= 1")
    rng = np.random.default_rng(seed)
    pattern = np.array([1.00, 0.95, 1.00, 1.05, 1.35, 1.10, 0.65])   # Mon..Sun
    y = base * pattern[np.arange(days) % 7] + rng.normal(0, noise, days)
    return np.clip(np.round(y), 0, None)
