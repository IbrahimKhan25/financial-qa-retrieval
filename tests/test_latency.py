import pytest

from finqa.eval.latency import compare


def test_compare_uses_the_median_round_and_applies_the_ratio():
    ok = compare([10.0, 10.0, 50.0], [12.0, 11.0, 12.0], max_ratio=1.3)  # one noisy base round is ignored
    assert ok["base_p95_ms"] == 10.0 and ok["head_p95_ms"] == 12.0
    assert ok["ratio"] == pytest.approx(1.2) and ok["passed"]
    slow = compare([10.0, 10.0, 10.0], [14.0, 13.5, 14.0], max_ratio=1.3)
    assert slow["ratio"] == pytest.approx(1.4) and not slow["passed"]


def test_compare_rejects_empty_or_zero_baselines():
    with pytest.raises(ValueError):
        compare([], [1.0], 1.3)
    with pytest.raises(ValueError):
        compare([0.0], [1.0], 1.3)
