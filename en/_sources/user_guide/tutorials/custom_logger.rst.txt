.. _tutorial-custom-logger:

Writing a logger
================

A logger receives the logs of a simulation: the orders, cancels and executions, and the beginning and end of the
simulation, of each session and of each market step. The built-in :class:`~pams.logs.MarketStepPrintLogger` and
:class:`~pams.logs.MarketStepSaver` only record the market prices at each step (see :doc:`first_simulation`).
This tutorial writes a logger that prints a summary of each session and saves every trade to a CSV file.
The complete script is :download:`tutorial_custom_logger.py <code/tutorial_custom_logger.py>`.


The logger
----------

A logger inherits from :class:`~pams.logs.Logger` and overrides the methods of the logs it needs. The others do
nothing:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Log
   * - ``process_simulation_begin_log``, ``process_simulation_end_log``
     - :class:`~pams.logs.SimulationBeginLog`, :class:`~pams.logs.SimulationEndLog`
   * - ``process_session_begin_log``, ``process_session_end_log``
     - :class:`~pams.logs.SessionBeginLog`, :class:`~pams.logs.SessionEndLog` (``log.session``)
   * - ``process_market_step_begin_log``, ``process_market_step_end_log``
     - :class:`~pams.logs.MarketStepBeginLog`, :class:`~pams.logs.MarketStepEndLog` (``log.session``,
       ``log.market``), for each market at each step
   * - ``process_order_log``
     - :class:`~pams.logs.OrderLog`, for each order a market has received
   * - ``process_cancel_log``
     - :class:`~pams.logs.CancelLog`, for each cancel order a market has received
   * - ``process_expiration_log``
     - ``ExpirationLog``, for each order removed from the order book because of its ``ttl``
   * - ``process_execution_log``
     - :class:`~pams.logs.ExecutionLog`, for each execution

The order, cancel, expiration and execution logs give the IDs of the market and of the agents. The other logs
give the simulator (``log.simulator``).

.. literalinclude:: code/tutorial_custom_logger.py
   :language: python
   :start-after: # [logger-start]
   :end-before: # [logger-end]

- ``__init__`` must call ``super().__init__()``.
- The runner sets ``self.simulator`` when the runner is created, before the simulation starts. The logger uses it
  to find the names of the market and of the agents from their IDs.
- An :class:`~pams.logs.ExecutionLog` gives the step (``time``), the market, the buyer and the seller, their
  order IDs, the price and the volume of one execution.
- The logger keeps the executions in memory and writes the file at the end of the simulation. This also keeps
  it usable with the process runner, which cannot copy a logger that holds an open file (see
  :ref:`config-parallel`).


When the logs are processed
---------------------------

The market step logs are processed at once, during the step. So, in ``process_market_step_end_log``, the market
is at the end of that step, which is how :class:`~pams.logs.MarketStepSaver` reads its prices.

The order, cancel, expiration and execution logs are kept by the logger until it receives the next session or
simulation log, at the beginning or at the end of a session or of the simulation. They are then processed in
the order they happened, before that log. In ``process_execution_log``, the market and the agents are therefore
already at the end of the session: use the fields of the log, not the current state of the market or the
agents.

In this logger, the orders and executions of a session are counted when the session ends, just before
``process_session_end_log`` prints them.


Using the logger
----------------

A runner takes one logger:

.. literalinclude:: code/tutorial_custom_logger.py
   :language: python
   :start-after: # [run-start]
   :end-before: # [run-end]

To record several things, handle them all in one logger. ``run_simulation(seed=42, path="trades.csv")`` prints
the summaries of the two sessions, then the two time lines of ``main()``:

.. code-block:: text

   warmup: 100 orders, 0 shares traded
   main: 500 orders, 205 shares traded
   # INITIALIZATION TIME 0.0019999
   # EXECUTION TIME 0.0909915

In each step, one FCN agent places one order (``maxNormalOrders`` is 1), and orders are only executed in the
``main`` session. ``trades.csv`` starts with:

.. code-block:: text

   time,market,price,volume,buyer,seller
   100,Market,298.11165,1,FCNAgents-16,FCNAgents-66
   100,Market,298.11165,1,FCNAgents-91,FCNAgents-66
   100,Market,298.11165,1,FCNAgents-87,FCNAgents-55

At step 100, the first step of ``main``, the orders placed during ``warmup`` start to be executed.
