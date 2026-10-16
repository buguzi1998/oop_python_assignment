"""Project 2 - Solar Micro-Grid Dispatch Planner (Kasese health centre).

System solved every day:  3x + 2y = D1 (daytime load),  4x + y = D2 (critical load)
where x = solar energy and y = battery energy (kWh).
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy.optimize import nnls

SOLAR_COST = 150.0     # UGX per kWh (illustrative)
BATTERY_COST = 450.0   # UGX per kWh (illustrative)


class SingularSystemError(ValueError):
    """Raised when the coefficient matrix is singular / numerically rank deficient."""


@dataclass
class DispatchResult:
    """Outcome of solving many days. Arrays have shape ``(n_unknowns, n_days)``."""
    raw: np.ndarray            # exact solution of A u = d (may contain negatives)
    final: np.ndarray          # solution after the chosen infeasibility strategy
    infeasible: np.ndarray     # bool mask per day: raw solution had a negative entry
    strategy: str


class MicroGrid:
    """Linear dispatch model ``A u = d`` for a micro-grid.

    Parameters
    ----------
    coefficients : array-like, optional
        Square matrix ``A``. Defaults to ``[[3, 2], [4, 1]]``.
    """

    DEFAULT_A = ((3.0, 2.0), (4.0, 1.0))
    SOURCES: Tuple[str, ...] = ("solar", "battery")

    def __init__(self, coefficients: Sequence[Sequence[float]] | None = None) -> None:
        a = np.array(self.DEFAULT_A if coefficients is None else coefficients, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("coefficient matrix must be square")
        self.A = a
        self.n = a.shape[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(A={self.A.tolist()})"

    # -- diagnostics ------------------------------------------------------
    def determinant(self) -> float:
        return float(np.linalg.det(self.A))

    def condition_number(self) -> float:
        return float(np.linalg.cond(self.A))

    def is_well_posed(self, cond_limit: float = 1e8) -> bool:
        """Non-singular and not numerically ill-conditioned."""
        return abs(self.determinant()) > 1e-12 and self.condition_number() < cond_limit

    def _require_well_posed(self) -> None:
        if not self.is_well_posed():
            raise SingularSystemError(
                f"system is singular or ill-conditioned (det={self.determinant():.3g}, "
                f"cond={self.condition_number():.3g})")

    # -- solving ----------------------------------------------------------
    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve one day. Example: ``grid.solve_day(d1, d2)`` returns ``[x, y]``."""
        if len(demands) != self.n:
            raise ValueError(f"expected {self.n} demand values, got {len(demands)}")
        d = np.asarray(demands, dtype=float)
        if np.any(~np.isfinite(d)) or np.any(d < 0):
            raise ValueError("demands must be finite and non-negative")
        self._require_well_posed()
        return linalg.solve(self.A, d)

    def solve_loop(self, demand: np.ndarray) -> np.ndarray:
        """Solve each column of ``demand`` (shape ``(n, days)``) in a Python loop."""
        demand = self._check_demand(demand)
        return np.column_stack([self.solve_day(*demand[:, j]) for j in range(demand.shape[1])])

    def solve_vectorised(self, demand: np.ndarray) -> np.ndarray:
        """Solve all days with one ``scipy.linalg.solve`` call (matrix right-hand side)."""
        demand = self._check_demand(demand)
        self._require_well_posed()
        return linalg.solve(self.A, demand)

    def _check_demand(self, demand: np.ndarray) -> np.ndarray:
        demand = np.asarray(demand, dtype=float)
        if demand.ndim != 2 or demand.shape[0] != self.n:
            raise ValueError(f"demand must have shape ({self.n}, n_days)")
        if demand.shape[1] == 0:
            raise ValueError("demand contains no days")
        if np.any(~np.isfinite(demand)) or np.any(demand < 0):
            raise ValueError("demands must be finite and non-negative")
        return demand

    # -- infeasibility handling ---------------------------------------------
    def dispatch(self, demand: np.ndarray, strategy: str = "nnls") -> DispatchResult:
        """Solve all days and repair physically infeasible ones (negative source use).

        strategy
            ``"clip"`` : clip negatives to zero (fast, but demand is then *not* met exactly)
            ``"nnls"`` : non-negative least squares - the closest feasible dispatch in the
                         least-squares sense (``scipy.optimize.nnls``)
            ``"none"`` : keep the raw solution
        """
        raw = self.solve_vectorised(demand)
        infeasible = np.any(raw < -1e-9, axis=0)
        if strategy == "none":
            final = raw.copy()
        elif strategy == "clip":
            final = np.clip(raw, 0.0, None)
        elif strategy == "nnls":
            final = raw.copy()
            for j in np.where(infeasible)[0]:
                final[:, j], _ = nnls(self.A, np.asarray(demand)[:, j])
        else:
            raise ValueError("strategy must be 'none', 'clip' or 'nnls'")
        return DispatchResult(raw=raw, final=final, infeasible=infeasible, strategy=strategy)

    def unmet_demand(self, usage: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Residual ``d - A u`` per day (0 when demand is met exactly)."""
        return np.asarray(demand) - self.A @ np.asarray(usage)

    # -- statistics & cost -------------------------------------------------
    @staticmethod
    def describe(values: Sequence[float]) -> Dict[str, float]:
        v = list(map(float, values))
        if len(v) < 2:
            raise ValueError("need at least two values")
        mean = statistics.mean(v)
        sd = statistics.stdev(v)
        return {"mean": mean, "variance": statistics.variance(v), "stdev": sd,
                "cv": sd / mean if mean else float("nan")}

    @staticmethod
    def daily_cost(solar: np.ndarray, battery: np.ndarray,
                   solar_price: float = SOLAR_COST, battery_price: float = BATTERY_COST) -> np.ndarray:
        if solar_price < 0 or battery_price < 0:
            raise ValueError("prices must be non-negative")
        return np.asarray(solar) * solar_price + np.asarray(battery) * battery_price

    # -- sensitivity ---------------------------------------------------------
    def sensitivity(self, demands: Sequence[float], pct: float = 0.05, n_draws: int = 1000,
                    seed: int = 42) -> Dict[str, np.ndarray | float]:
        """Monte Carlo: perturb every demand uniformly by +-``pct`` and re-solve.

        Returns the perturbed solutions plus the observed and theoretical amplification of
        relative error. Theory: ``||du||/||u|| <= cond(A) * ||dd||/||d||``.
        """
        rng = np.random.default_rng(seed)
        d = np.asarray(demands, dtype=float)
        base = self.solve_day(*d)
        factors = rng.uniform(1 - pct, 1 + pct, size=(self.n, n_draws))
        sols = linalg.solve(self.A, d[:, None] * factors)
        rel_in = np.linalg.norm(d[:, None] * factors - d[:, None], axis=0) / np.linalg.norm(d)
        rel_out = np.linalg.norm(sols - base[:, None], axis=0) / np.linalg.norm(base)
        return {"base": base, "solutions": sols, "rel_in": rel_in, "rel_out": rel_out,
                "max_amplification": float(np.max(rel_out / rel_in)),
                "cond": self.condition_number()}


class HybridMicroGrid(MicroGrid):
    """Solar (x) + battery (y) + diesel generator (z) with a 3x3 system.

    Default equations (kWh):
        3x + 2y +  z = D1   (daytime load, diesel also contributes)
        4x +  y      = D2   (critical load: diesel is not wired to the critical circuit)
         x +  y +  z = D3   (total dispatch requirement / capacity balance)
    """

    DEFAULT_A = ((3.0, 2.0, 1.0), (4.0, 1.0, 0.0), (1.0, 1.0, 1.0))
    SOURCES = ("solar", "battery", "diesel")

    def __init__(self, coefficients: Sequence[Sequence[float]] | None = None) -> None:
        super().__init__(self.DEFAULT_A if coefficients is None else coefficients)
        if self.n != 3:
            raise ValueError("HybridMicroGrid needs a 3x3 matrix")

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))

    def solve_day_safe(self, *demands: float) -> Tuple[np.ndarray, str]:
        """Like ``solve_day`` but falls back to the minimum-norm least-squares solution
        (``np.linalg.lstsq``) if the equations are linearly dependent."""
        try:
            return self.solve_day(*demands), "exact"
        except SingularSystemError:
            sol, *_ = np.linalg.lstsq(self.A, np.asarray(demands, dtype=float), rcond=None)
            return sol, f"least-squares (rank {self.rank()} < {self.n}: infinitely many or no solutions)"


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def parse_demand(text: str) -> float:
    """Parse one demand value; reject empty, non-numeric, negative or non-finite input."""
    text = (text or "").strip()
    if not text:
        raise ValueError("value is empty")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    if value < 0:
        raise ValueError("value must not be negative")
    return value


def prompt_demand(label: str, input_fn: Callable[[str], str] = input,
                  max_tries: int | None = None) -> float:
    """Ask until a valid demand is entered (``input_fn`` can be mocked in tests)."""
    tries = 0
    while True:
        try:
            return parse_demand(input_fn(f"Enter {label} (kWh): "))
        except ValueError as err:
            print(f"  Invalid input: {err}. Please try again.")
            tries += 1
            if max_tries is not None and tries >= max_tries:
                raise


def interactive_day(grid: MicroGrid, input_fn: Callable[[str], str] = input) -> np.ndarray:
    """Interactive mode: ask for D1 and D2 with re-prompting, then solve."""
    d1 = prompt_demand("daytime demand D1", input_fn)
    d2 = prompt_demand("critical demand D2", input_fn)
    return grid.solve_day(d1, d2)


def generate_demand_csv(path: str | Path, days: int = 30, seed: int = 42) -> Path:
    """Write a seeded synthetic 30-day demand file with a weekly pattern plus noise."""
    if days < 1:
        raise ValueError("days must be >= 1")
    rng = np.random.default_rng(seed)
    weekly = np.array([1.05, 1.08, 1.06, 1.04, 1.00, 0.82, 0.75])   # Mon..Sun clinic activity
    d = np.arange(days)
    f = weekly[d % 7]
    d1 = 40.0 * f + rng.normal(0, 2.5, days)
    d2 = 45.0 * (0.9 + 0.1 * f) + rng.normal(0, 4.0, days)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "d1_kwh", "d2_kwh"])
        for i in range(days):
            w.writerow([i + 1, round(max(d1[i], 0), 2), round(max(d2[i], 0), 2)])
    return path


def load_demand_csv(path: str | Path) -> np.ndarray:
    """Read the CSV and return a ``(2, days)`` array, validating every row."""
    rows: List[Tuple[float, float]] = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append((parse_demand(row["d1_kwh"]), parse_demand(row["d2_kwh"])))
            except (ValueError, KeyError) as err:
                raise ValueError(f"line {line_no}: {err}") from None
    if not rows:
        raise ValueError("CSV contains no data rows")
    return np.array(rows, dtype=float).T
