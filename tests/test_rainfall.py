import numpy as np
import pytest

from src.rainfall import (CROP_RULES, CropRule, Region, classify_modality, cosine_similarity,
                          detect_peaks, euclidean, load_regions, pearson, planting_windows,
                          similarity_matrices, verify_cosine)


def test_region_metrics():
    r = Region("T", [10] * 6 + [20] * 6)
    assert r.annual_total() == 180 and r.mean() == 15
    assert r.wettest_month() == "Jul" and r.driest_month() == "Jan"
    assert r.cv() > 0 and len(r) == 12


@pytest.mark.parametrize("bad", [[1] * 11, [1] * 13, [-1] + [1] * 11, [np.nan] + [1] * 11])
def test_region_validation(bad):
    with pytest.raises(ValueError):
        Region("bad", bad)


def test_region_zero_rainfall_cv_edge_case():
    with pytest.raises(ZeroDivisionError):
        Region("desert", [0] * 12).cv()


def test_crop_rule_classification():
    rule = CropRule("maize", 90, 200)
    assert rule.classify(50) == "Drought risk"
    assert rule.classify(120) == "Good for maize"
    assert rule.classify(250) == "Waterlogging risk"
    assert rule.classify(90) == "Good for maize" and rule.classify(200) == "Good for maize"   # boundaries
    with pytest.raises(ValueError):
        CropRule("x", 100, 50)
    with pytest.raises(ValueError):
        rule.classify(-1)


def test_cosine_hand_calculation_and_scipy_agreement():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)      # same direction, different size
    mine, ref, ok = verify_cosine([120, 140, 180], [8, 25, 75])
    assert ok and mine == pytest.approx(ref)


def test_cosine_zero_vector_edge_case():
    with pytest.raises(ZeroDivisionError):
        cosine_similarity([0, 0, 0], [1, 2, 3])


def test_cosine_high_even_when_one_region_much_wetter():
    a = np.array([10.0, 20, 30, 20])
    assert cosine_similarity(a, 10 * a) == pytest.approx(1.0)
    assert euclidean(a, 10 * a) > 100


def test_pearson_and_constant_series():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    with pytest.raises(ZeroDivisionError):
        pearson([1, 1, 1], [1, 2, 3])


def test_similarity_matrices_symmetric_with_unit_diagonal():
    m = similarity_matrices(load_regions())
    for k in ("cosine", "pearson", "euclidean"):
        assert np.allclose(m[k], m[k].T)
    assert np.allclose(np.diag(m["cosine"]), 1) and np.allclose(np.diag(m["euclidean"]), 0)


def test_peak_detection_on_synthetic_regimes():
    months = np.arange(12)
    uni = Region("uni", 100 + 80 * np.cos(2 * np.pi * (months - 6) / 12))
    bi = Region("bi", 100 + 80 * np.cos(4 * np.pi * (months - 3) / 12))
    assert classify_modality(uni) == "unimodal"
    assert classify_modality(bi) == "bimodal"
    assert classify_modality(Region("flat", [50] * 12)) == "no clear season"


def test_peaks_wrap_around_year_end():
    r = Region("wrap", [90, 20, 10, 10, 10, 10, 10, 10, 10, 10, 20, 100])   # peak straddles Dec/Jan
    assert len(detect_peaks(r)) == 1


def test_mbarara_is_bimodal_and_windows():
    regions = {r.name: r for r in load_regions()}
    assert classify_modality(regions["Mbarara"]) == "bimodal"
    beans = [c for c in CROP_RULES if c.crop == "beans"][0]
    wins = planting_windows(regions["Mbarara"], beans, 2)
    assert all(len(w) == 2 for w in wins)
    with pytest.raises(ValueError):
        planting_windows(regions["Mbarara"], beans, 0)
