"""Check the stylized facts of returns in a seeded FCN market.

The setup, the statistics, the thresholds and their calibration are described in
docs/source/user_guide/stylized_facts.rst. Keep that page in sync with this file.
"""

import copy
import random
import warnings
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pytest

from pams.runners import SequentialRunner
from pams.utils import autocorrelation
from pams.utils import excess_kurtosis
from pams.utils import hill_tail_index
from pams.utils import log_returns

WARM_UP_STEPS: int = 1000
SIMULATION_STEPS: int = 20000

# samples/CI2002/config.json with shorter time windows, narrower order margins and
# slower mean reversion, and with longer sessions; withPrint is not used by PAMS, and
# the deprecated hifreqSubmitRate is left out because its default is the sample value
CONFIG: Dict[str, Any] = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents"],
        "sessions": [
            {
                "sessionName": 0,
                "iterationSteps": WARM_UP_STEPS,
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": False,
            },
            {
                "sessionName": 1,
                "iterationSteps": SIMULATION_STEPS,
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": False,
            },
        ],
    },
    "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents": 100,
        "markets": ["Market"],
        "assetVolume": 50,
        "cashAmount": 10000,
        "fundamentalWeight": {"expon": [1.0]},
        "chartWeight": {"expon": [0.0]},
        "noiseWeight": {"expon": [1.0]},
        "meanReversionTime": {"uniform": [400, 800]},
        "noiseScale": 0.001,
        "timeWindowSize": [10, 50],
        "orderMargin": [0.0, 0.01],
    },
}

SEEDS: List[int] = [0, 1]

MIN_PRICE_CHANGES: int = 5000
MIN_EXCESS_KURTOSIS: float = 3.0
TAIL_FRACTION: float = 0.05
MIN_TAIL_INDEX: float = 1.5
MAX_TAIL_INDEX: float = 4.5
MAX_RETURN_LAG: int = 20
MAX_RETURN_AUTOCORRELATION: float = 0.1
MIN_VOLATILITY_AUTOCORRELATION: float = 0.1
MAX_VOLATILITY_LAG: int = 10
MIN_MEAN_VOLATILITY_AUTOCORRELATION: float = 0.02


@pytest.fixture(
    name="returns", scope="module", params=SEEDS, ids=lambda seed: f"seed{seed}"
)
def fixture_returns(request: pytest.FixtureRequest) -> np.ndarray:
    """Log returns in event time after the warm-up session of a seeded run."""
    runner = SequentialRunner(
        settings=copy.deepcopy(CONFIG), prng=random.Random(request.param)
    )
    with warnings.catch_warnings():
        # FCN agents do not round their order prices to the tick size
        warnings.filterwarnings(
            "ignore", message="order price does not accord", category=UserWarning
        )
        runner.main()
    prices: List[float] = runner.simulator.markets[0].get_market_prices()
    return log_returns(prices[WARM_UP_STEPS:], drop_unchanged=True)


def test_enough_price_changes(returns: np.ndarray) -> None:
    assert len(returns) > MIN_PRICE_CHANGES


def test_heavy_tails(returns: np.ndarray) -> None:
    assert excess_kurtosis(returns) > MIN_EXCESS_KURTOSIS


def test_tail_index(returns: np.ndarray) -> None:
    tail_index: float = hill_tail_index(returns, tail_fraction=TAIL_FRACTION)
    assert MIN_TAIL_INDEX < tail_index < MAX_TAIL_INDEX


def test_absence_of_autocorrelation(returns: np.ndarray) -> None:
    for lag in range(1, MAX_RETURN_LAG + 1):
        value: float = autocorrelation(returns, lag)
        assert abs(value) < MAX_RETURN_AUTOCORRELATION, f"lag {lag}: {value}"


def test_volatility_clustering(returns: np.ndarray) -> None:
    abs_returns: np.ndarray = np.abs(returns)
    assert autocorrelation(abs_returns, 1) > MIN_VOLATILITY_AUTOCORRELATION
    mean_autocorrelation: float = float(
        np.mean(
            [
                autocorrelation(abs_returns, lag)
                for lag in range(1, MAX_VOLATILITY_LAG + 1)
            ]
        )
    )
    assert mean_autocorrelation > MIN_MEAN_VOLATILITY_AUTOCORRELATION
