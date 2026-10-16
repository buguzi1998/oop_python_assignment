"""Project 4 - Rainfall Pattern & Crop Suitability Analyser."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine as scipy_cosine_distance

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

RAINFALL: Dict[str, List[float]] = {   # mm per month (illustrative)
    "Kampala": [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130],
    "Gulu": [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15],
    "Mbarara": [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90],
}


class Region:
    """Monthly rainfall (mm, Jan-Dec) of one region."""

    def __init__(self, name: str, rainfall: Sequence[float]) -> None:
        arr = np.asarray(rainfall, dtype=float)
        if arr.shape != (12,):
            raise ValueError("rainfall must contain exactly 12 monthly values")
        if np.any(np.isnan(arr)) or np.any(arr < 0):
            raise ValueError("rainfall must be non-negative and not NaN")
        self.name = name
        self.rainfall = arr

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():.0f} mm)"

    def __len__(self) -> int:
        return 12

    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        """Coefficient of variation of the monthly totals (seasonality index)."""
        m = self.rainfall.mean()
        if m == 0:
            raise ZeroDivisionError("CV undefined for a region with zero mean rainfall")
        return float(self.rainfall.std(ddof=1) / m)

    def summary(self) -> Dict[str, object]:
        return {"region": self.name, "annual_mm": self.annual_total(), "mean_mm": self.mean(),
                "wettest": self.wettest_month(), "driest": self.driest_month(), "cv": self.cv()}


@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall range ``[min_mm, max_mm]`` for a crop."""
    crop: str
    min_mm: float
    max_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.min_mm < 0 or self.max_mm <= self.min_mm:
            raise ValueError("need 0 <= min_mm < max_mm")

    def classify(self, mm: float) -> str:
        if mm < 0:
            raise ValueError("rainfall cannot be negative")
        if mm < self.min_mm:
            return "Drought risk"
        if mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def code(self, mm: float) -> int:
        """0 = drought risk, 1 = good, 2 = waterlogging risk (for heatmaps)."""
        label = self.classify(mm)
        return 0 if label == "Drought risk" else (2 if label == "Waterlogging risk" else 1)

    def classify_region(self, region: Region) -> List[str]:
        return [self.classify(x) for x in region.rainfall]


# Thresholds are *approximate monthly equivalents* of published seasonal water needs; see README.
CROP_RULES: List[CropRule] = [
    CropRule("maize", 90, 200,
             "FAO Irrigation & Drainage Paper 33 (Doorenbos & Kassam 1979): 500-800 mm over a "
             "~4-month cycle = 125-200 mm/month; lower bound relaxed for soil-water storage."),
    CropRule("beans", 80, 140,
             "FAO Paper 33: 300-500 mm over a ~3-month cycle = 100-170 mm/month; upper bound "
             "lowered because beans are sensitive to waterlogging and fungal disease."),
    CropRule("coffee", 100, 170,
             "Arabica 1,200-1,500 mm/yr and robusta 1,500-2,000 mm/yr (Coste 1992; Antonio 2008, "
             "cited in Agroclimatic zoning for coffee, Angola) = 100-170 mm/month."),
]


def suitability_table(regions: Sequence[Region], rules: Sequence[CropRule]) -> Dict[Tuple[str, str], List[str]]:
    """Label every month for every (region, crop)."""
    return {(r.name, c.crop): c.classify_region(r) for r in regions for c in rules}


def suitability_codes(region: Region, rule: CropRule) -> np.ndarray:
    return np.array([rule.code(x) for x in region.rainfall])


def planting_windows(region: Region, rule: CropRule, length: int = 3) -> List[List[str]]:
    """All runs of ``length`` consecutive (cyclic) months that are all 'Good' for ``rule``."""
    if not 1 <= length <= 12:
        raise ValueError("length must be between 1 and 12")
    good = suitability_codes(region, rule) == 1
    out = []
    for s in range(12):
        idx = [(s + k) % 12 for k in range(length)]
        if all(good[i] for i in idx):
            out.append([MONTHS[i] for i in idx])
    return out


# --------------------------------------------------------------------------
# Similarity measures
# --------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """cos(theta) = (a . b) / (|a| |b|). (The old method used math.cos(), which is wrong.)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("vectors must be 1-D and the same length")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ZeroDivisionError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / (na * nb))


def verify_cosine(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, bool]:
    """Own implementation vs ``1 - scipy.spatial.distance.cosine``."""
    mine = cosine_similarity(a, b)
    ref = 1.0 - float(scipy_cosine_distance(a, b))
    return mine, ref, bool(np.isclose(mine, ref))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        raise ZeroDivisionError("Pearson correlation undefined for a constant series")
    return float(np.corrcoef(a, b)[0, 1])


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("vectors must be the same length")
    return float(np.linalg.norm(a - b))


def similarity_matrices(regions: Sequence[Region]) -> Dict[str, np.ndarray]:
    """Pairwise cosine, Pearson and Euclidean matrices (symmetric, region order preserved)."""
    n = len(regions)
    out = {k: np.zeros((n, n)) for k in ("cosine", "pearson", "euclidean")}
    for i, ri in enumerate(regions):
        for j, rj in enumerate(regions):
            out["cosine"][i, j] = cosine_similarity(ri.rainfall, rj.rainfall)
            out["pearson"][i, j] = pearson(ri.rainfall, rj.rainfall)
            out["euclidean"][i, j] = euclidean(ri.rainfall, rj.rainfall)
    return out


# --------------------------------------------------------------------------
# Rainy-season detection
# --------------------------------------------------------------------------
def detect_peaks(region: Region, min_prominence_frac: float = 0.25) -> List[str]:
    """Rainy-season peaks found with ``scipy.signal.find_peaks``.

    Rainfall is *cyclic* (December is followed by January), so the series is tiled three
    times and only peaks in the middle copy are kept; otherwise peaks near the ends would
    be missed. A peak counts as a season only if its prominence is at least
    ``min_prominence_frac`` x (max - min) of the annual cycle.
    """
    if not 0 <= min_prominence_frac <= 1:
        raise ValueError("min_prominence_frac must be in [0, 1]")
    r = region.rainfall
    amp = r.max() - r.min()
    if amp == 0:
        return []
    tiled = np.tile(r, 3)
    peaks, _ = find_peaks(tiled, prominence=max(min_prominence_frac * amp, 1e-9))
    mid = [int(p) - 12 for p in peaks if 12 <= p < 24]
    return [MONTHS[p] for p in mid]


def classify_modality(region: Region, min_prominence_frac: float = 0.25) -> str:
    n = len(detect_peaks(region, min_prominence_frac))
    return {0: "no clear season", 1: "unimodal"}.get(n, "bimodal" if n == 2 else f"multimodal ({n})")


def modality_sensitivity(regions: Sequence[Region],
                         fracs: Sequence[float] = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40)) -> Dict[str, Dict[float, str]]:
    """How the unimodal/bimodal call changes with the prominence threshold."""
    return {r.name: {f: classify_modality(r, f) for f in fracs} for r in regions}


def load_regions() -> List[Region]:
    return [Region(k, v) for k, v in RAINFALL.items()]
