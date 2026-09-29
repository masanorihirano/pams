import math
from typing import List

import numpy as np
import pytest
from scipy import stats

from pams.utils import autocorrelation
from pams.utils import excess_kurtosis
from pams.utils import hill_tail_index
from pams.utils import log_returns


class TestLogReturns:
    def test_every_step(self) -> None:
        prices: List[float] = [100.0, 110.0, 99.0]
        expected: List[float] = [math.log(1.1), math.log(0.9)]
        assert log_returns(prices).tolist() == pytest.approx(expected)

    def test_interval(self) -> None:
        prices: List[float] = [100.0, 110.0, 99.0, 108.9, 100.0, 120.0]
        # prices 100.0, 99.0 and 100.0 are sampled; the trailing 120.0 is not
        returns: np.ndarray = log_returns(prices, interval=2)
        assert returns.tolist() == pytest.approx(
            [math.log(0.99), math.log(100.0 / 99.0)]
        )
        assert len(log_returns(list(range(1, 22)), interval=10)) == 2
        assert len(log_returns([100.0], interval=1)) == 0
        assert len(log_returns([], interval=2)) == 0

    def test_drop_unchanged(self) -> None:
        prices: List[float] = [100.0, 100.0, 110.0, 110.0, 110.0, 99.0, 99.0]
        assert len(log_returns(prices)) == 6
        assert log_returns(prices, drop_unchanged=True).tolist() == pytest.approx(
            [math.log(1.1), math.log(0.9)]
        )
        # the prices are sampled before the unchanged periods are removed
        assert log_returns(
            prices, interval=2, drop_unchanged=True
        ).tolist() == pytest.approx([math.log(1.1), math.log(0.9)])
        assert len(log_returns([100.0, 100.0, 100.0], drop_unchanged=True)) == 0

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            log_returns([100.0, 101.0], interval=0)
        with pytest.raises(ValueError):
            log_returns([100.0, 0.0, 101.0])
        with pytest.raises(ValueError):
            log_returns([100.0, -1.0, 101.0])


class TestExcessKurtosis:
    def test_values(self) -> None:
        assert excess_kurtosis([1.0, -1.0, 1.0, -1.0]) == pytest.approx(-2.0)
        # uniform distribution on 0, 1, ..., 9: -6 (n^2 + 1) / (5 (n^2 - 1))
        assert excess_kurtosis(list(range(10))) == pytest.approx(-606.0 / 495.0)

    def test_matches_scipy(self) -> None:
        rng = np.random.default_rng(0)
        for values in [rng.standard_normal(1000), rng.standard_t(5, 1000)]:
            assert excess_kurtosis(values) == pytest.approx(stats.kurtosis(values))

    def test_heavier_tails_give_larger_values(self) -> None:
        rng = np.random.default_rng(0)
        gaussian: float = excess_kurtosis(rng.standard_normal(20000))
        student_t: float = excess_kurtosis(rng.standard_t(5, 20000))
        assert abs(gaussian) < 0.2
        assert student_t > 2.0

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            excess_kurtosis([])
        with pytest.raises(ValueError):
            excess_kurtosis([1.0, 1.0, 1.0])
        # the rounded means of these constant series differ from their values
        with pytest.raises(ValueError):
            excess_kurtosis([0.1] * 3)
        with pytest.raises(ValueError):
            excess_kurtosis([math.log(1.1)] * 7)


class TestAutocorrelation:
    def test_values(self) -> None:
        assert autocorrelation([1.0, 2.0, 3.0, 4.0, 5.0], lag=1) == pytest.approx(0.4)
        assert autocorrelation([1.0, 2.0, 3.0, 4.0, 5.0], lag=2) == pytest.approx(-0.1)
        assert autocorrelation([1.0, -1.0, 1.0, -1.0], lag=1) == pytest.approx(-0.75)
        assert autocorrelation([1.0, -1.0, 1.0, -1.0], lag=2) == pytest.approx(0.5)

    def test_ar1_process(self) -> None:
        rng = np.random.default_rng(0)
        noise: np.ndarray = rng.standard_normal(20000)
        series: np.ndarray = np.zeros(len(noise))
        for t in range(1, len(noise)):
            series[t] = 0.5 * series[t - 1] + noise[t]
        assert autocorrelation(series, lag=1) == pytest.approx(0.5, abs=0.03)
        assert autocorrelation(series, lag=2) == pytest.approx(0.25, abs=0.03)
        assert autocorrelation(noise, lag=1) == pytest.approx(0.0, abs=0.03)

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation([1.0, 2.0, 3.0], lag=0)
        with pytest.raises(ValueError):
            autocorrelation([1.0, 2.0, 3.0], lag=3)
        with pytest.raises(ValueError):
            autocorrelation([1.0, 1.0, 1.0], lag=1)
        with pytest.raises(ValueError):
            autocorrelation([0.1] * 3, lag=1)


class TestHillTailIndex:
    def test_pareto_quantiles(self) -> None:
        # deterministic quantiles of a Pareto distribution with tail index alpha
        for alpha in [1.5, 2.0, 3.0]:
            values: List[float] = [
                (i / 10000) ** (-1.0 / alpha) for i in range(1, 10001)
            ]
            estimate: float = hill_tail_index(values, tail_fraction=0.05)
            assert estimate == pytest.approx(alpha, rel=0.02)

    def test_uses_both_tails(self) -> None:
        values: List[float] = [(i / 1000) ** -0.5 for i in range(1, 1001)]
        mirrored: List[float] = [-v if i % 2 else v for i, v in enumerate(values)]
        assert hill_tail_index(mirrored) == pytest.approx(hill_tail_index(values))

    def test_heavier_tails_give_smaller_values(self) -> None:
        rng = np.random.default_rng(0)
        gaussian: float = hill_tail_index(rng.standard_normal(20000))
        student_t: float = hill_tail_index(rng.standard_t(3, 20000))
        assert student_t < 4.0 < gaussian

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            hill_tail_index([1.0, 2.0, 3.0], tail_fraction=0.0)
        with pytest.raises(ValueError):
            hill_tail_index([1.0, 2.0, 3.0], tail_fraction=1.0)
        with pytest.raises(ValueError):
            hill_tail_index([1.0, 2.0, 3.0], tail_fraction=0.05)
        with pytest.raises(ValueError):
            hill_tail_index([0.0] * 90 + [1.0] * 10, tail_fraction=0.5)
        with pytest.raises(ValueError):
            hill_tail_index([1.0] * 100, tail_fraction=0.05)
