import math
import random
from statistics import mean
from statistics import stdev

import pytest

from pams.utils import JsonRandom


class TestJsonRandom:
    def test__init__(self) -> None:
        prng = random.Random(42)
        jr = JsonRandom(prng=prng)
        assert jr.prng == prng

    def test_next_uniform(self) -> None:
        jr = JsonRandom(prng=random.Random(42))
        n_sample = 100
        results = [
            jr._next_uniform(min_value=10.9, max_value=12.2) for _ in range(n_sample)
        ]
        assert sum([x < 10.9 for x in results]) == 0
        assert sum([x >= 12.2 for x in results]) == 0

    def test_next_normal(self) -> None:
        jr = JsonRandom(prng=random.Random(42))
        n_samples = 1000
        results = [jr._next_normal(mu=1.0, sigma=2.0) for _ in range(n_samples)]
        assert abs(mean(results) - 1.0) < 2.0 * 2.0 / math.sqrt(n_samples)
        assert abs(stdev(results) - 2.0) < 2.0 * 2.0 / math.sqrt(n_samples)

    def test_next_exponential(self) -> None:
        jr = JsonRandom(prng=random.Random(42))
        n_samples = 1000
        results = [jr._next_exponential(lam=2.0) for _ in range(n_samples)]
        assert abs(mean(results) - 2.0) < 2.0 * 2.0 / math.sqrt(n_samples)
        assert abs(stdev(results) - 2.0) < 2.0 * 2.0 / math.sqrt(n_samples)

    def test_random(self) -> None:
        jr = JsonRandom(prng=random.Random(42))
        n_samples = 1000
        results = [jr.random([10.9, 12.2]) for _ in range(n_samples)]
        assert sum([x < 10.9 for x in results]) == 0
        assert sum([x >= 12.2 for x in results]) == 0
        results = [jr.random({"const": [10.1]}) for _ in range(n_samples)]
        assert sum([x != 10.1 for x in results]) == 0
        results = [jr.random(10.1) for _ in range(n_samples)]
        assert sum([x != 10.1 for x in results]) == 0
        results = [jr.random({"uniform": [10.9, 12.2]}) for _ in range(n_samples)]
        assert sum([x < 10.9 for x in results]) == 0
        assert sum([x >= 12.2 for x in results]) == 0
        results = [jr.random({"normal": [1.0, 2.0]}) for _ in range(n_samples)]
        assert abs(mean(results) - 1.0) < 2.0 * 2.0 / math.sqrt(n_samples)
        assert abs(stdev(results) - 2.0) < 2.0 * 2.0 / math.sqrt(n_samples)
        results = [jr.random({"expon": [2.0]}) for _ in range(n_samples)]
        assert abs(mean(results) - 2.0) < 2.0 * 2.0 / math.sqrt(n_samples)
        assert abs(stdev(results) - 2.0) < 2.0 * 2.0 / math.sqrt(n_samples)

        with pytest.raises(ValueError):
            jr.random([1])
        with pytest.raises(ValueError):
            jr.random({"normal": [1.0, 2.0], "expon": [2.0]})
        with pytest.raises(ValueError):
            jr.random({"const": 1.0})
        with pytest.raises(ValueError):
            jr.random({"const": [1.0, 2.0]})
        with pytest.raises(ValueError):
            jr.random({"uniform": 1.0})
        with pytest.raises(ValueError):
            jr.random({"uniform": [1.0]})
        with pytest.raises(ValueError):
            jr.random({"normal": 1.0})
        with pytest.raises(ValueError):
            jr.random({"normal": [1.0]})
        with pytest.raises(ValueError):
            jr.random({"expon": 1.0})
        with pytest.raises(ValueError):
            jr.random({"expon": [1.0, 2.0]})
        with pytest.raises(ValueError):
            jr.random({"unknown": [1.0, 2.0]})
