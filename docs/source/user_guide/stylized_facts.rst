.. _stylized-facts:

Stylized facts
==============

Stylized facts are statistical properties that the returns of many assets share across markets
and periods (Cont, 2001). An artificial market is expected to reproduce at least some of them.
PAMS checks a few of them automatically: the tests in ``tests/validation`` run seeded simulations
of a market with FCN agents and assert each fact with a fixed threshold, so that a change that
breaks a fact makes the test suite fail.

This page describes which facts are checked, how they are measured, on which simulation, why the
thresholds are set as they are, how to run the checks, and what is not checked.

.. _stylized-facts-simulation:

Simulation
----------

The checks use one market with 100 FCN agents (:class:`pams.agents.FCNAgent`), the setup of
Chiarella and Iori (2002) that ``samples/CI2002`` also follows. The config is the one in
``samples/CI2002/config.json`` except for the keys below. Apart from them, the tests only set
``withPrint`` to ``false`` and leave out the deprecated ``hifreqSubmitRate``, and neither changes
the simulation.

.. list-table::
   :header-rows: 1

   * - Key
     - ``samples/CI2002``
     - Validation
   * - ``FCNAgents.timeWindowSize``
     - ``[100, 200]``
     - ``[10, 50]``
   * - ``FCNAgents.orderMargin``
     - ``[0.0, 0.1]``
     - ``[0.0, 0.01]``
   * - ``FCNAgents.meanReversionTime``
     - ``{"uniform": [50, 100]}``
     - ``{"uniform": [400, 800]}``
   * - Steps of session 0 (without order execution)
     - 100
     - 1,000
   * - Steps of session 1 (with order execution)
     - 500
     - 20,000

With the parameters of the sample, the order margins of up to 10% and the noise of the expected
price, which grows with the time window, spread the orders over a wide range of prices. The price
then mostly jumps between the two sides of a wide spread: successive price changes have a lag-1
autocorrelation of about -0.44 (bid-ask bounce). The returns are also close to Gaussian, with an
excess kurtosis of about 0.2 to 1.7 and a Hill estimate of about 5 to 7 (40 runs with seeds 0 to
39). So neither the absence of linear autocorrelation nor heavy tails holds, although the absolute
returns are about as autocorrelated as in the validation setup. The shorter time windows and the
narrower margins keep the orders close to the current price, and the slower mean reversion
weakens the pull towards the constant fundamental price, which otherwise shows up as a negative
autocorrelation of returns.

Chart following stays off as in the sample. Even with a ``chartWeight`` of ``{"expon": [0.1]}``,
the returns follow trends, with a lag-1 autocorrelation of up to about 0.4, and in some runs the
price drifts by orders of magnitude. With ``{"expon": [0.3]}`` or more, the price often runs away
and the simulation fails.

.. _stylized-facts-returns:

Returns
-------

The checks use the log returns of the market price (:meth:`pams.Market.get_market_prices`) in
session 1, in event (tick) time: the steps in which the price does not change are skipped, so
each return is the change between two successive different prices. In this setup, the price does
not change in about 60% of the steps, and in calendar time the zero returns would dominate the
statistics. A run gives 7,500 to 8,500 returns. They are computed by
:func:`pams.utils.log_returns` with ``drop_unchanged=True``.

.. _stylized-facts-checks:

Checked facts
-------------

In the table, :math:`r` is the series of returns, :math:`n` is its length, and
:math:`\mathrm{ACF}(x, k)` is the sample autocorrelation of a series :math:`x` at lag :math:`k`.
The calibration column gives the range over the 240 runs described in
:ref:`stylized-facts-thresholds`. The reference column gives the values for 8,000 independent
Gaussian returns (200 samples) or for the returns of each run in a random order, which keeps
their distribution but removes any dependence in time.

.. list-table::
   :header-rows: 1

   * - Fact
     - Statistic
     - Threshold
     - Calibration
     - Reference
   * - Heavy tails
     - Excess kurtosis of :math:`r` (:func:`pams.utils.excess_kurtosis`)
     - Above 3
     - 5.8 to 30.3 (median 12.5)
     - -0.13 to 0.20 for Gaussian returns
   * - Tail exponent
     - Hill estimate of the tail index from the largest 5% of :math:`|r|`
       (:func:`pams.utils.hill_tail_index`)
     - Between 1.5 and 4.5
     - 1.72 to 3.25 (median 2.51)
     - 5.4 to 6.8 for Gaussian returns; 2 to 5 in real markets (Cont, 2001)
   * - Absence of linear autocorrelation
     - Largest :math:`|\mathrm{ACF}(r, k)|` over :math:`k = 1, \dots, 20`
       (:func:`pams.utils.autocorrelation`)
     - Below 0.1
     - 0.024 to 0.067
     - :math:`1.96 / \sqrt{n} \approx 0.022` bounds 95% of the values for white noise
   * - Volatility clustering
     - :math:`\mathrm{ACF}(|r|, 1)`
     - Above 0.1
     - 0.144 to 0.240 (median 0.185)
     - -0.028 to 0.034 for shuffled returns
   * - Volatility clustering
     - Mean of :math:`\mathrm{ACF}(|r|, k)` over :math:`k = 1, \dots, 10`
     - Above 0.02
     - 0.034 to 0.077 (median 0.053)
     - -0.009 to 0.010 for shuffled returns

The tests also require more than 5,000 returns, so that a run in which the price hardly moves
fails instead of giving meaningless statistics.

.. _stylized-facts-thresholds:

Thresholds
----------

The tests run seeds 0 and 1. A seed fixes the run, so the tests are deterministic, but the
thresholds are not tuned to these two runs. The results of the floating-point math library can
differ in the last digit between platforms, and such a tiny difference can change which orders
match and then the whole path of the price. The thresholds must therefore hold for any typical
run, not only for the two seeds. They were calibrated on 240 runs with seeds 0 to 239:

- every threshold holds in all 240 runs, with a clear margin (see the table above);
- every threshold lies far from the values of returns without the fact, i.e. of Gaussian returns
  for the tails and of shuffled returns for volatility clustering, so the tests fail if a fact
  disappears.

The bounds are loose on purpose: they check that the facts are present, not their exact size.
For example, the tail index is not compared with a value estimated from real data. It only has
to be in the range of heavy tails rather than that of thin tails. The lower bound of 1.5 catches
runs that a few extreme jumps dominate.

If a change to PAMS makes a check fail, first decide whether the change is meant to alter the
market dynamics. If it is, calibrate again on many seeds, and update both the thresholds in
``tests/validation/test_stylized_facts.py`` and this page.

.. _stylized-facts-running:

Running the checks
------------------

The checks are part of the test suite, so they run in CI on every supported operating system and
Python version. To run only them:

.. code-block:: bash

   python -m pytest tests/validation

The two runs take a few seconds in total. The statistics are public functions of
:mod:`pams.utils`, so they also work on the prices of your own simulations:

.. code-block:: python

   import numpy as np

   from pams.utils import autocorrelation
   from pams.utils import excess_kurtosis
   from pams.utils import hill_tail_index
   from pams.utils import log_returns

   prices = runner.simulator.markets[0].get_market_prices()
   returns = log_returns(prices, drop_unchanged=True)
   print(excess_kurtosis(returns))
   print(hill_tail_index(returns, tail_fraction=0.05))
   print([autocorrelation(returns, lag) for lag in range(1, 21)])
   print([autocorrelation(np.abs(returns), lag) for lag in range(1, 11)])

.. _stylized-facts-out-of-scope:

Out of scope
------------

- **Aggregational Gaussianity.** The excess kurtosis of the returns over 100 steps is below that
  of the returns over 10 steps in all 240 runs (median ratio 0.09), but the ratio reaches 0.95 in
  the worst run: a run has only 200 returns over 100 steps, and a few large moves dominate their
  kurtosis. This fact is therefore not asserted.
- **Long memory of volatility.** The autocorrelation of absolute returns decays quickly and falls
  below :math:`1.96 / \sqrt{n}` after 4 to 14 lags (median 7). It does not decay slowly like a
  power law, as it does in real markets.
- **Other facts** listed by Cont (2001), such as the leverage effect, the gain/loss asymmetry and
  the correlation between volume and volatility, are not checked.
- **Other setups.** Only the market above is checked. The samples and the other agent classes are
  not validated, and no parameter is calibrated to real data.

.. _stylized-facts-references:

References
----------

- Chiarella, C., & Iori, G. (2002). A simulation analysis of the microstructure of double auction
  markets. Quantitative Finance, 2(5), 346–353. https://doi.org/10.1088/1469-7688/2/5/303
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues.
  Quantitative Finance, 1(2), 223–236. https://doi.org/10.1080/713665670
- Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution.
  The Annals of Statistics, 3(5), 1163–1174. https://doi.org/10.1214/aos/1176343247
