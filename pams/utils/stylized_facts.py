"""Statistics for checking the stylized facts of simulated price series."""

from typing import Sequence
from typing import Union

import numpy as np


def log_returns(
    prices: Union[Sequence[float], np.ndarray],
    interval: int = 1,
    drop_unchanged: bool = False,
) -> np.ndarray:
    r"""Compute the log returns of a price series.

    The prices are sampled every ``interval`` steps, starting from the first price,
    and the log returns :math:`\log p_{t + \Delta} - \log p_t` of the sampled prices
    are returned, where :math:`\Delta` is ``interval``.

    If ``drop_unchanged`` is True, the returns of the periods in which the price
    does not change are removed. With ``interval=1``, this gives the returns in event
    (tick) time, i.e. the returns between successive price changes.

    Args:
        prices (Union[Sequence[float], np.ndarray]): price series, e.g. the output of
            :meth:`pams.Market.get_market_prices`. All prices must be positive.
        interval (int): sampling interval :math:`\Delta` in steps. It must be
            positive. Defaults to 1.
        drop_unchanged (bool): whether to remove the zero returns. Defaults to False.

    Returns:
        np.ndarray: log returns. Without ``drop_unchanged``, the length is
        ``max(len(prices) - 1, 0) // interval``.

    Examples:
        >>> from pams.utils import log_returns
        >>> log_returns([100.0, 110.0, 99.0]).round(4).tolist()
        [0.0953, -0.1054]
        >>> prices = [100.0, 110.0, 99.0, 108.9, 100.0]
        >>> log_returns(prices, interval=2).round(4).tolist()
        [-0.0101, 0.0101]
        >>> prices = [100.0, 100.0, 110.0, 110.0, 110.0, 99.0]
        >>> log_returns(prices, drop_unchanged=True).round(4).tolist()
        [0.0953, -0.1054]
    """
    if interval < 1:
        raise ValueError("interval must be positive")
    prices_array: np.ndarray = np.asarray(prices, dtype=float)
    if np.any(prices_array <= 0):
        raise ValueError("prices must be positive")
    returns: np.ndarray = np.diff(np.log(prices_array[::interval]))
    if drop_unchanged:
        returns = returns[returns != 0.0]
    return returns


def excess_kurtosis(values: Union[Sequence[float], np.ndarray]) -> float:
    r"""Compute the sample excess kurtosis of a series.

    The excess kurtosis is :math:`m_4 / m_2^2 - 3`, where
    :math:`m_j = \frac{1}{n} \sum_{t=1}^{n} (x_t - \bar{x})^j` is the :math:`j`-th
    sample central moment. It is 0 for a Gaussian distribution and positive for a
    distribution with heavier tails. This is the same estimator as the default of
    :func:`scipy.stats.kurtosis`.

    Args:
        values (Union[Sequence[float], np.ndarray]): series, e.g. log returns.

    Returns:
        float: excess kurtosis.

    Examples:
        >>> from pams.utils import excess_kurtosis
        >>> excess_kurtosis([1.0, -1.0, 1.0, -1.0])
        -2.0
        >>> round(excess_kurtosis([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]), 2)
        3.14
    """
    values_array: np.ndarray = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        raise ValueError("values must not be empty")
    # the raw values are compared because the deviations of a constant series from
    # its mean may not be exactly zero after rounding
    if np.all(values_array == values_array[0]):
        raise ValueError("values must not be constant")
    deviations: np.ndarray = values_array - values_array.mean()
    second_moment: float = float(np.mean(deviations**2))
    fourth_moment: float = float(np.mean(deviations**4))
    return fourth_moment / second_moment**2 - 3.0


def autocorrelation(values: Union[Sequence[float], np.ndarray], lag: int) -> float:
    r"""Compute the sample autocorrelation of a series at a lag.

    The estimator is the one used for correlograms,
    :math:`\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})
    / \sum_{t=1}^{n} (x_t - \bar{x})^2`,
    where :math:`n` is the length of the series and :math:`k` is the lag.

    Args:
        values (Union[Sequence[float], np.ndarray]): series, e.g. log returns or
            absolute log returns.
        lag (int): lag :math:`k`. It must be positive and smaller than the length of
            ``values``.

    Returns:
        float: autocorrelation at ``lag``.

    Examples:
        >>> from pams.utils import autocorrelation
        >>> autocorrelation([1.0, 2.0, 3.0, 4.0, 5.0], lag=1)
        0.4
        >>> autocorrelation([1.0, -1.0, 1.0, -1.0], lag=1)
        -0.75
    """
    values_array: np.ndarray = np.asarray(values, dtype=float)
    if lag < 1 or lag >= len(values_array):
        raise ValueError("lag must be positive and smaller than the length of values")
    # the raw values are compared for the same reason as in excess_kurtosis
    if np.all(values_array == values_array[0]):
        raise ValueError("values must not be constant")
    deviations: np.ndarray = values_array - values_array.mean()
    denominator: float = float(np.dot(deviations, deviations))
    return float(np.dot(deviations[:-lag], deviations[lag:])) / denominator


def hill_tail_index(
    values: Union[Sequence[float], np.ndarray], tail_fraction: float = 0.05
) -> float:
    r"""Estimate the tail index of a distribution with the Hill estimator.

    The absolute values are sorted in descending order,
    :math:`x_{(1)} \ge x_{(2)} \ge \dots \ge x_{(n)}`, and the largest
    :math:`k = \lfloor q n \rfloor` of them are used, where :math:`q` is
    ``tail_fraction``:
    :math:`\hat{\alpha} = k / \sum_{i=1}^{k} \log (x_{(i)} / x_{(k+1)})` (Hill, 1975).
    Both tails are pooled because the absolute values are used.

    A smaller value means a heavier tail. For a power-law tail
    :math:`P(|X| > x) \propto x^{-\alpha}`, the estimate approaches :math:`\alpha`.
    For a thin tail, the estimate is finite but large: for 8,000 Gaussian samples
    and ``tail_fraction=0.05``, it is about 6.

    Args:
        values (Union[Sequence[float], np.ndarray]): samples, e.g. log returns.
        tail_fraction (float): fraction :math:`q` of the samples used as the tail.
            It must be in (0, 1). Defaults to 0.05.

    Returns:
        float: estimated tail index.

    Examples:
        >>> from pams.utils import hill_tail_index
        >>> pareto_quantiles = [(i / 1000) ** -0.5 for i in range(1, 1001)]
        >>> round(hill_tail_index(pareto_quantiles, tail_fraction=0.05), 2)
        2.08
    """
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must be in (0, 1)")
    sorted_values: np.ndarray = np.sort(np.abs(np.asarray(values, dtype=float)))[::-1]
    k: int = int(tail_fraction * len(sorted_values))
    if k < 1:
        raise ValueError("values are too short for tail_fraction")
    threshold: float = float(sorted_values[k])
    if threshold == 0.0:
        raise ValueError("values have too many zeros for tail_fraction")
    log_excess: float = float(np.sum(np.log(sorted_values[:k] / threshold)))
    if log_excess == 0.0:
        raise ValueError("the tail of values must not be constant")
    return k / log_excess
