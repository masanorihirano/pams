.. _config:

JSON configuration
==================

A PAMS simulation is described by a single JSON file: which markets exist, which agents trade on them,
how the simulation is split into sessions, and which events (shocks, circuit breakers, ...) happen.
Changing the market setup, the population of agents or the scenario usually needs no Python code at all:
you only edit the config file.

This page explains how the file is organized and documents every key understood by the built-in classes.

.. |required| replace:: :bdg-danger:`required`
.. |optional| replace:: :bdg-secondary:`optional`
.. |conditional| replace:: :bdg-warning:`conditional`
.. |jsonrandom| replace:: :bdg-info:`JsonRandom`
.. |deprecated| replace:: :bdg-dark:`deprecated`

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Quick start
      :link: config-quickstart
      :link-type: ref

      Complete, runnable configs for common setups.

   .. grid-item-card:: File structure
      :link: config-structure
      :link-type: ref

      The ``simulation`` block, named blocks and how they refer to each other.

   .. grid-item-card:: Sessions
      :link: config-sessions
      :link-type: ref

      Steps, order placement and execution, the order flow of one step.

   .. grid-item-card:: Groups and naming
      :link: config-groups
      :link-type: ref

      ``numMarkets`` / ``numAgents``, ``from`` / ``to``, ``prefix`` and ``extends``.

   .. grid-item-card:: Markets
      :link: config-markets
      :link-type: ref

      ``Market``, ``IndexMarket``, fundamental prices and correlations.

   .. grid-item-card:: Agents
      :link: config-agents
      :link-type: ref

      ``FCNAgent``, ``ArbitrageAgent``, ``MarketMakerAgent`` and more.

   .. grid-item-card:: Events
      :link: config-events
      :link-type: ref

      Price shocks, price limits, trading halts and fat-finger orders.

   .. grid-item-card:: JsonRandom
      :link: config-jsonrandom
      :link-type: ref

      Random agent parameters such as ``[100, 200]`` or ``{"expon": [1.0]}``.

   .. grid-item-card:: Troubleshooting
      :link: config-troubleshooting
      :link-type: ref

      Common mistakes, silent pitfalls and error messages.


.. _config-quickstart:

Quick start
-----------

Save one of the configs below as ``config.json`` and run it with a runner:

.. code-block:: python

   import random

   from pams.logs import MarketStepPrintLogger
   from pams.runners import SequentialRunner

   with open("config.json", encoding="utf-8") as fp:
       runner = SequentialRunner(
           settings=fp,  # a dict or a file path also works
           prng=random.Random(42),  # fix the seed for reproducible results
           logger=MarketStepPrintLogger(),  # prints the market price at every step
       )
   runner.main()

.. tab-set::

   .. tab-item:: Minimal

      One market and 100 :class:`~pams.agents.FCNAgent` agents.
      The first session only builds the order book (no execution), the second one trades.
      With the default session settings, only one agent, chosen at random, places orders in each step
      (see ``maxNormalOrders`` in :ref:`config-sessions`).
      It follows the Chiarella and Iori (2002) setup of ``samples/CI2002``, which additionally sets
      ``meanReversionTime``.

      .. literalinclude:: config_examples/minimal.json
         :language: json

   .. tab-item:: Events

      The minimal setup plus two events in the ``main`` session:
      the fundamental price drops by 10% 100 steps into the session,
      and a circuit breaker halts trading for 50 steps when the market price moves 5% away from its value at
      step 0 (10% for the second halt, 15% for the third, and so on).

      .. literalinclude:: config_examples/events.json
         :language: json
         :emphasize-lines: 19,23-34

   .. tab-item:: Index and arbitrage

      Two correlated spot markets, an index market made of them, FCN agents on each market,
      and :class:`~pams.agents.ArbitrageAgent` agents that trade the gap between the index market price
      and the index value. The FCN groups share their settings through ``extends``.
      The spot markets set ``outstandingShares`` (required for every component of an index) and
      ``fundamentalVolatility`` (required for the correlation). The index and the correlation refer to the spot
      markets by their instance names ``SpotMarkets-0`` and ``SpotMarkets-1``, while the agents use the group name
      ``SpotMarkets`` (see :ref:`config-names`). The sessions let three FCN agents place orders per step
      (``maxNormalOrders: 3``) and keep the arbitrage agents off during the warm-up
      (``maxHighFrequencyOrders: 0``).

      .. literalinclude:: config_examples/index_arbitrage.json
         :language: json
         :emphasize-lines: 25-27,34-35,41

   .. tab-item:: Market maker

      FCN agents plus two ``MarketMakerAgent`` agents that quote both sides of the market.
      Market makers are high-frequency agents, so the session keys ``maxHighFrequencyOrders`` and
      ``highFrequencySubmitRate`` control how often they act. Here they are switched off during the warm-up
      (``maxHighFrequencyOrders: 0``). In ``main``, in about half of the steps (``highFrequencySubmitRate: 0.5``),
      one of the two market makers, chosen at random, quotes after the normal agent's orders.

      .. literalinclude:: config_examples/market_maker.json
         :language: json
         :emphasize-lines: 12,20-21,43-52

   .. tab-item:: Parallel runner

      The same format works with :class:`~pams.runners.MultiThreadAgentParallelRunner` and
      :class:`~pams.runners.MultiProcessAgentParallelRunner` (both experimental).
      ``simulation.numParallel`` sets the number of workers. At most ``maxNormalOrders`` agents are asked in
      parallel, so ``numParallel`` should not exceed it (both are 4 here). ``maxNormalOrders`` is part of the
      model, because it sets how many agents place orders per step: choose it first, then set ``numParallel``.
      To run this config, replace ``SequentialRunner`` in the snippet above with one of these classes
      (``SequentialRunner`` ignores ``numParallel``), and see :ref:`config-parallel` before using the process
      runner.

      .. literalinclude:: config_examples/parallel.json
         :language: json
         :emphasize-lines: 12,20,23

More complete examples are in the ``samples`` directory of the repository (see :ref:`config-samples`).


.. _config-structure:

File structure
--------------

The root of the file is a JSON object with two kinds of entries:

- the ``simulation`` block, which lists what is created and in which order, and
- **named blocks**: every other top-level key. The key is a name you choose, and the block holds the settings
  of a market group, an agent group or an event.

A named block does nothing by itself. It is used only when its name is referenced:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Where the block name is listed
     - Result
   * - ``simulation.markets``
     - The block becomes a **market group** (one or more markets).
   * - ``simulation.agents``
     - The block becomes an **agent group** (one or more agents).
   * - ``events`` of a session
     - The block becomes an **event** attached to that session.
   * - ``"extends"`` in another block
     - The other block inherits this block's keys (see :ref:`config-extends`).
       A block that is only used through ``extends`` is a template.

The block name and the class are independent: ``"class"`` selects the class, and the name is only a label.
``"FCNAgents": {"class": "FCNAgent", ...}`` and ``"Noise traders": {"class": "FCNAgent", ...}`` are equally valid.

.. _config-names:

Group names and instance names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A market group can create several markets (``"numMarkets": 2``). The group then has a **group name**
(the block name, e.g. ``SpotMarkets``) and each market has an **instance name** (e.g. ``SpotMarkets-0`` and
``SpotMarkets-1``, see :ref:`config-naming`). Different keys expect different kinds of names:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Key
     - Expects
   * - ``simulation.markets``, ``simulation.agents``, ``sessions[].events``, ``extends``
     - block names
   * - ``markets`` of an agent group
     - market **group** names (all markets of the group become tradable)
   * - ``markets`` of an ``IndexMarket``, ``fundamentalCorrelations.pairwise``,
       ``targetMarket`` of ``MarketMakerAgent``, ``target`` and ``targetMarkets`` of events
     - market **instance** names

When a group creates a single market and uses neither ``prefix`` nor ``from`` / ``to``, both names are the same,
so the distinction mostly matters for groups with several markets. With those keys the names differ even for a
single market: ``"prefix": "Stock"`` gives the instance name ``Stock``.

.. _config-notation:

Notation used on this page
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75

   * - |required|
     - The key must be present. Otherwise setting up the simulation fails with an error.
   * - |optional|
     - The key may be omitted. The default is used.
   * - |conditional|
     - Required only in some situations, described next to the key.
   * - |jsonrandom|
     - The value may be a random distribution drawn separately for each agent (see :ref:`config-jsonrandom`).
       Keys without this badge take a plain value.
   * - |deprecated|
     - Old key. Depending on the key, it still works with a warning, is ignored with a warning, or is an error
       (see its description). Do not use it in new configs.

.. important::

   - The file must be **strict JSON**: no comments and no trailing commas. Use ``true`` / ``false``, not
     ``True`` / ``False``. To leave a note, add a key that PAMS does not read, such as ``"MEMO"``; the samples do
     this.
   - **Unknown keys are silently ignored.** A typo such as ``"numMarket"`` instead of ``"numMarkets"`` does not
     raise an error; the default is used instead. Check spelling and case carefully.
   - Some keys check the JSON number type. ``triggerChangeRate`` (``PriceLimitRule``, ``TradingHaltRule``) and
     ``priceChangeRate`` of ``OrderMistakeShock`` must be written as floats (``1.0``, not ``1``).
     ``iterationSteps``, ``outstandingShares``, ``numParallel``, ``triggerTime``, ``shockTimeLength``,
     ``haltingTimeLength``, and ``orderVolume`` / ``orderTimeLength`` of ``ArbitrageAgent`` and
     ``OrderMistakeShock`` must be written as integers (``100``, not ``100.0``).


.. _config-simulation:

The ``simulation`` block
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Value
     - Description
   * - ``markets`` |required|
     - list of block names
     - The market groups to create, in this order. Market IDs are assigned in this order, and an
       ``IndexMarket`` must come after the markets it is made of.
   * - ``agents`` |required|
     - list of block names
     - The agent groups to create, in this order. Agent IDs are assigned in this order.
   * - ``sessions`` |required|
     - list of session objects
     - The sessions, run one after another in list order. See :ref:`config-sessions`.
   * - ``fundamentalCorrelations`` |optional|
     - object
     - Correlations between the fundamental prices of markets. See :ref:`config-correlations`.
       Default: all fundamental prices move independently.
   * - ``numParallel`` |optional|
     - int ≥ 1
     - Number of workers for the agent-parallel runners (default: the number of CPUs minus 1, at least 1).
       Ignored by
       :class:`~pams.runners.SequentialRunner`. See :ref:`config-parallel`.
   * - ``startMethod`` |optional|
     - ``"spawn"``, ``"fork"`` or ``"forkserver"``
     - How :class:`~pams.runners.MultiProcessAgentParallelRunner` starts its worker processes (default: the
       platform's default). Only the start methods available on the platform are accepted: Windows has only
       ``"spawn"``. Ignored by the other runners. See :ref:`config-parallel`.
   * - ``torchNumThreads`` |optional|
     - int ≥ 1
     - Number of threads of PyTorch on each worker process of :class:`~pams.runners.TorchAgentParallelRunner`
       (default: the number of threads of PyTorch on the main process divided by ``numParallel``, at least 1).
       Ignored by the other runners. See :doc:`platform`.

Other keys in ``simulation`` are ignored. In particular, events are not listed here but in each session.


.. _config-sessions:

Sessions
--------

A simulation runs in discrete **steps** on one clock that starts at 0. The steps are divided into
**sessions**, which run one after another. Each session decides whether agents may place orders and whether
orders are executed. A typical setup has a warm-up session that only fills the order book, followed by
the main trading session:

.. code-block:: json

   "sessions": [
     {"sessionName": "warmup", "iterationSteps": 100,
      "withOrderPlacement": true, "withOrderExecution": false, "withPrint": true},
     {"sessionName": "main", "iterationSteps": 500,
      "withOrderPlacement": true, "withOrderExecution": true, "withPrint": true}
   ]

Here ``warmup`` covers steps 0 to 99 and ``main`` covers steps 100 to 599. In general, a session starts at
the sum of the ``iterationSteps`` of all sessions before it.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Value
     - Description
   * - ``sessionName`` |required|
     - string
     - Name of the session. Any JSON value is converted to a string, and names must be unique after the
       conversion (``0`` and ``"0"`` collide). The built-in loggers show the session's position in the list
       (``0``, ``1``, ...), not this name.
   * - ``iterationSteps`` |required|
     - int
     - Number of steps in the session. Must be a JSON integer (``100.0`` is rejected).
   * - ``withOrderPlacement`` |required|
     - bool
     - If ``false``, no agent is asked for orders during the session. Time still advances, the fundamental
       price still moves and orders in the book can still expire.
   * - ``withOrderExecution`` |required|
     - bool
     - If ``false``, orders are accepted into the order books but never executed, and the market price stays
       at its previous value. Orders still in the book (not expired) can be matched once a later session allows
       execution.
   * - ``withPrint`` |required|
     - bool
     - Required, but currently not used by PAMS (kept for compatibility with plham configs).
       What is printed is decided by the logger passed to the runner.
   * - ``maxNormalOrders`` |optional|
     - int ≥ 0
     - Maximum number of normal (not high-frequency) agents that submit orders in one step. Default: ``1``.
       It counts **agents**, not orders: an agent may submit several orders at once.
   * - ``maxHighFrequencyOrders`` |optional|
     - int ≥ 0
     - Maximum number of high-frequency agents that submit orders in each high-frequency round.
       Default: ``1``. It also counts agents, not orders.
   * - ``highFrequencySubmitRate`` |optional|
     - float in [0, 1]
     - Probability that a high-frequency round follows each normal agent's orders. Default: ``1.0``.
   * - ``events`` |optional|
     - list of block names
     - Events attached to this session. See :ref:`config-events`.
   * - ``maxHifreqOrders`` |deprecated|
     - int
     - Old name of ``maxHighFrequencyOrders``. Works with a warning; giving both is an error.
   * - ``hifreqSubmitRate`` |deprecated|
     - float
     - Old name of ``highFrequencySubmitRate``. Works with a warning; giving both is an error.

None of the session keys accepts :ref:`JsonRandom <config-jsonrandom>` notation, and session objects do not support
``extends`` (it is ignored like any unknown key).

.. dropdown:: What happens in one step
   :icon: list-ordered

   For every step of a session:

   #. Events hooked before the step run (e.g. :class:`~pams.events.FundamentalPriceShock`).
   #. If ``withOrderPlacement`` is ``true``, the normal agents are shuffled and asked one by one for orders
      (all markets are passed to each agent). Asking stops when ``maxNormalOrders`` agents have returned at
      least one order, or when every agent has been asked. An agent that returns no order does not count.
   #. The collected order lists are processed in random order. Each order (or cancel) goes to its market and,
      if ``withOrderExecution`` is ``true`` and the market is not halted by a
      :class:`~pams.events.TradingHaltRule`, the market matches orders immediately (continuous double auction).
   #. After each normal agent's orders, a **high-frequency round** takes place with probability
      ``highFrequencySubmitRate``: the high-frequency agents are shuffled and asked until
      ``maxHighFrequencyOrders`` of them have returned orders. Each one's orders are processed (and matched) right
      away, before the next one is asked, so high-frequency agents react to the orders placed just before them.
   #. The end-of-step log is written (this is what :class:`~pams.logs.MarketStepPrintLogger` prints), then the
      events hooked after the step run.
   #. Time advances by one step: expired orders leave the books and the fundamental prices move.

.. important::

   - With the default ``maxNormalOrders: 1``, only **one** normal agent places orders per step, however many
     agents the config creates. Raise it to let more agents act in the same step. They all decide on the same
     market state: none of them sees the orders the others submit in that step.
   - High-frequency agents (:class:`~pams.agents.ArbitrageAgent` and ``MarketMakerAgent``) act **only** in the
     high-frequency rounds that can follow each normal agent's orders (step 4 above). They never act in a step
     where no normal agent submitted an order, and ``highFrequencySubmitRate: 0.0`` switches them off.


.. _config-groups:

Market and agent groups
-----------------------

Market blocks and agent blocks share a set of keys that decide which class is used, how many objects are
created and how they are named.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Value
     - Description
   * - ``class`` |required|
     - string
     - Class name, e.g. ``"Market"`` or ``"FCNAgent"``. Use the plain class name (case-sensitive), not a dotted
       path. Built-in classes are found automatically; your own classes must be registered
       (see :ref:`config-user-classes`).
   * - ``extends`` |optional|
     - block name
     - Inherit the keys of another block. See :ref:`config-extends`.
   * - ``numMarkets`` / ``numAgents`` |optional|
     - int
     - Number of markets (in a market block) or agents (in an agent block) created from this block.
       Default: ``1``. Cannot be combined with ``from`` / ``to``.
   * - ``from`` and ``to`` |optional|
     - int
     - Create one object for each index from ``from`` to ``to`` (**both included**, so ``to`` must be at least
       ``from``). Give both or neither. The indices only affect names, not IDs.
   * - ``prefix`` |optional|
     - string
     - Base of the instance names. Default: the block name, plus ``-`` when the group creates more than one
       object. An explicit ``prefix`` is used as is (no ``-`` is added). See :ref:`config-naming`.

.. _config-naming:

Instance names
~~~~~~~~~~~~~~

Names are what other keys use to refer to a single market (see :ref:`config-names`), so it helps to know
how they are built:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Block ``"Market": {...}`` with
     - Resulting names
   * - no count, or ``"numMarkets": 1``
     - ``Market``
   * - ``"numMarkets": 3``
     - ``Market-0``, ``Market-1``, ``Market-2``
   * - ``"numMarkets": 3, "prefix": "Stock"``
     - ``Stock0``, ``Stock1``, ``Stock2``
   * - ``"numMarkets": 3, "prefix": "Stock-"``
     - ``Stock-0``, ``Stock-1``, ``Stock-2``
   * - ``"from": 3, "to": 5``
     - ``Market-3``, ``Market-4``, ``Market-5``
   * - ``"from": 5, "to": 5``
     - ``Market5``

The rule is:

- Without ``prefix``, the prefix is the block name, followed by ``-`` when the group has more than one object.
  An explicit ``prefix`` is used as is (no ``-`` is added).
- When the group has exactly one object and no ``from`` / ``to``, the name is the prefix alone. Otherwise an
  index is appended: ``0, 1, 2, ...`` with a count, or ``from`` to ``to`` with a range.

Market names must be unique across all market groups, and agent names across all agent groups.
IDs (``market_id`` and ``agent_id``) are assigned from 0 in the order of ``simulation.markets`` and
``simulation.agents``, regardless of ``from`` / ``to``.

.. _config-extends:

Inheritance with ``extends``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"extends": "OtherBlock"`` copies every key of ``OtherBlock`` into this block, and keys written in this block
override the inherited ones. The parent can itself use ``extends``, so chains are resolved recursively.
A block has only one parent: ``extends`` takes a single block name, not a list.
Market groups, agent groups and event blocks all support ``extends``.

.. code-block:: json

   "FCNBase": {
     "class": "FCNAgent", "numAgents": 100, "assetVolume": 50, "cashAmount": 10000,
     "fundamentalWeight": {"expon": [1.0]}, "chartWeight": {"expon": [0.0]},
     "noiseWeight": {"expon": [1.0]}, "noiseScale": 0.001,
     "timeWindowSize": [100, 200], "orderMargin": [0.0, 0.1]
   },
   "SpotFCNAgents": {"extends": "FCNBase", "markets": ["SpotMarkets"]},
   "Chartists": {"extends": "FCNBase", "markets": ["SpotMarkets"], "chartWeight": {"expon": [1.0]}}

Here ``FCNBase`` is a template: it is not listed in ``simulation.agents``, so no agent is created from it
directly. Both children inherit ``"numAgents": 100``, so each creates 100 agents (``SpotFCNAgents-0`` to
``SpotFCNAgents-99`` and ``Chartists-0`` to ``Chartists-99``).

- The merge is shallow: an object value in the child (e.g. ``{"expon": [1.0]}``) replaces the parent's value as
  a whole.
- ``from`` and ``to`` are never inherited. ``numMarkets``, ``numAgents`` and ``prefix`` **are** inherited by
  market and agent groups. A child that sets ``from`` / ``to`` while its parent sets ``numAgents`` therefore
  fails, and two listed groups with the same inherited ``prefix`` (e.g. two children of one template, or a child
  and its listed parent) produce duplicate names unless their ``from`` / ``to`` ranges do not overlap. For
  example, two children of a template with ``"prefix": "Trader-"`` (and no ``numAgents``) that use
  ``"from": 0, "to": 49`` and ``"from": 50, "to": 99`` together create ``Trader-0`` to ``Trader-99``.
  Otherwise give each group its own ``prefix``.
- Event blocks do not inherit ``numMarkets``, ``from``, ``to`` or ``prefix``.
- A missing parent and a loop (``A`` extends ``B`` extends ``A``) are errors.


.. _config-markets:

Markets
-------

A market block creates one or more order-book markets. Besides the group keys (``class``, ``extends``,
``numMarkets``, ``from`` / ``to``, ``prefix``), a market block has the keys below. None of them accepts
:ref:`JsonRandom <config-jsonrandom>` notation.

Market
~~~~~~

``"class": "Market"`` (:class:`pams.Market`)

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``tickSize`` |required|
     - number > 0
     - Minimum price increment. Limit order prices that are not a multiple of it are rounded
       (buy orders down, sell orders up) with a warning.
   * - ``marketPrice`` |conditional|
     - number > 0
     - Initial market price. At least one of ``marketPrice`` and ``fundamentalPrice`` is required.
   * - ``fundamentalPrice`` |conditional|
     - number > 0
     - Initial fundamental price. At least one of ``marketPrice`` and ``fundamentalPrice`` is required.
   * - ``fundamentalDrift`` |optional|
     - number
     - Drift of the fundamental price per step (see below). Default: ``0.0``.
   * - ``fundamentalVolatility`` |optional|
     - number ≥ 0
     - Volatility of the fundamental price per step (see below). Default: ``0.0``, i.e. the fundamental price
       only follows the drift.
   * - ``outstandingShares`` |conditional|
     - int
     - Number of shares. Required for markets that are part of an ``IndexMarket``, where it is the weight of
       the market in the index. Must be a JSON integer. Not set by default.

If only one of ``marketPrice`` and ``fundamentalPrice`` is given, both prices start at that value.
If both are given, the market price starts at ``marketPrice`` and the fundamental price at
``fundamentalPrice``.

.. dropdown:: How the fundamental price moves
   :icon: graph

   The fundamental price :math:`P^f_t` of each market follows a geometric random walk with one step per
   simulation step:

   .. math::

      \ln P^f_{t+1} - \ln P^f_t = \mu + \sigma \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)

   where :math:`\mu` is ``fundamentalDrift`` and :math:`\sigma` is ``fundamentalVolatility``. Both are per
   step, not annualized. The shocks :math:`\varepsilon_t` of different markets are independent unless they are
   correlated with :ref:`fundamentalCorrelations <config-correlations>`. The fundamental price keeps moving in
   every session, including sessions without order execution.

.. dropdown:: How the market price is determined
   :icon: info

   Agents see the initial market price at the start of step 0. From then on, while the session executes orders,
   the market price is the last execution price, or the mid price if nothing has been executed yet.
   While orders are not executed (``withOrderExecution: false`` or during a trading halt), the market price
   stays at its previous value.

   An incoming order is executed at the price of the order it matches in the book (the one placed earlier);
   a market order takes the price of the limit order it matches. If an order fills against several price levels
   at once, all of these fills use the last (worst) of those prices.

.. _config-indexmarket:

IndexMarket
~~~~~~~~~~~

``"class": "IndexMarket"`` (:class:`pams.IndexMarket`) is a market whose fundamental price is the weighted
average of the fundamental prices of its component markets. It has its own order book and market price,
which can differ from the **index value**, the ``outstandingShares``-weighted average of the component markets'
**market** prices (not their fundamental prices). :class:`~pams.agents.ArbitrageAgent`
trades this gap.

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``markets`` |required|
     - list of market instance names
     - Component markets, e.g. ``["SpotMarkets-0", "SpotMarkets-1"]``. Every component needs
       ``outstandingShares``, and its group must be listed **before** the index in ``simulation.markets``.
   * - ``tickSize`` |required|
     - number > 0
     - As for ``Market``.
   * - ``marketPrice`` / ``fundamentalPrice`` |conditional|
     - number
     - At least one is required. They only set the initial market price of the index. The fundamental price
       of the index is always computed from the components.
   * - ``outstandingShares`` |conditional|
     - int
     - Only needed when this index is itself a component of another index.
   * - ``requires`` |deprecated|
     - any
     - Ignored with a warning.

The fundamental price of the index at each step is

.. math::

   P^f_{\mathrm{index}, t} = \frac{\sum_i S_i P^f_{i, t}}{\sum_i S_i}

where :math:`S_i` is the ``outstandingShares`` of component :math:`i`. ``fundamentalDrift`` and
``fundamentalVolatility`` of an ``IndexMarket`` are ignored. To avoid an artificial gap at the start, set its
``marketPrice`` to the weighted average of the component prices.

.. _config-correlations:

Correlated fundamental prices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``simulation.fundamentalCorrelations`` correlates the random shocks of the fundamental prices of pairs of
markets:

.. code-block:: json

   "fundamentalCorrelations": {
     "pairwise": [
       ["SpotMarkets-0", "SpotMarkets-1", 0.5],
       ["SpotMarkets-0", "SpotMarkets-2", -0.2]
     ]
   }

- ``pairwise`` is a list of ``[market name, market name, correlation]``. The names are market **instance**
  names, and the two markets must differ.
- The correlation must be strictly between -1 and 1, and both markets need ``fundamentalVolatility`` greater
  than 0. Index markets cannot be used, because their fundamental price is computed from their components.
- If the same pair appears twice, the last value is used. All the correlations together must form a valid
  (positive definite) correlation matrix; otherwise the simulation fails when it starts.
- ``pairwise`` is the only supported key.


.. _config-agents:

Agents
------

An agent block creates one or more agents of one class. Besides the group keys (``class``, ``extends``,
``numAgents``, ``from`` / ``to``, ``prefix``), every agent block has the keys below.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Value
     - Description
   * - ``markets`` |required|
     - list of market group names
     - The markets these agents can trade. Every market of each listed group becomes tradable.
   * - ``cashAmount`` |required| |jsonrandom|
     - number
     - Initial cash of each agent.
   * - ``assetVolume`` |required| |jsonrandom|
     - number
     - Initial number of shares held in **each** tradable market (drawn separately per market and truncated
       to an integer).

Cash and holdings are only bookkeeping: the built-in agents do not check them before ordering, so they can
become negative.

Each agent is either a **normal** agent or a **high-frequency** agent, which decides when it is asked for
orders (see *What happens in one step* in :ref:`config-sessions`):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Class
     - Kind
     - Behavior
   * - :class:`~pams.agents.FCNAgent`
     - normal
     - Fundamentalist, chartist and noise trader.
   * - :class:`~pams.agents.MarketShareFCNAgent`
     - normal
     - FCN trader that picks one market per decision, weighted by recent trading volume.
   * - :class:`~pams.agents.ArbitrageAgent`
     - high-frequency
     - Trades the gap between an index market and its components.
   * - ``MarketMakerAgent``
     - high-frequency
     - Quotes a buy and a sell order around the current price.
   * - ``TestAgent``
     - normal
     - Random orders for testing PAMS itself. Not intended for research.

FCNAgent
~~~~~~~~

``"class": "FCNAgent"`` (:class:`pams.agents.FCNAgent`) combines a fundamentalist, a chartist and a noise
component (Chiarella and Iori, 2002). All its numeric parameters (every key except ``marginType``) accept
:ref:`JsonRandom <config-jsonrandom>` notation, so every agent can get different values.

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``fundamentalWeight`` |required| |jsonrandom|
     - number ≥ 0
     - Weight :math:`w_F` of the fundamental component.
   * - ``chartWeight`` |required| |jsonrandom|
     - number ≥ 0
     - Weight :math:`w_C` of the chartist (trend-following) component.
   * - ``noiseWeight`` |required| |jsonrandom|
     - number ≥ 0
     - Weight :math:`w_N` of the noise component. The three weights must not all be 0.
   * - ``noiseScale`` |required| |jsonrandom|
     - number
     - Standard deviation :math:`\sigma_N` of the noise component.
   * - ``timeWindowSize`` |required| |jsonrandom|
     - int ≥ 1
     - Time window :math:`\tau` in steps: the look-back of the chartist component, the forecast horizon and the
       lifetime (TTL) of the orders. Truncated to an integer.
   * - ``orderMargin`` |required| |jsonrandom|
     - number ≥ 0
     - Price margin :math:`k`. With ``"fixed"`` margins it is relative and must be between 0 and 1. With
       ``"normal"`` margins it is the standard deviation of the order price, in price units; keep it well below
       the price, because an order price below 0 stops the simulation with an ``AssertionError``.
   * - ``marginType`` |optional|
     - ``"fixed"`` or ``"normal"``
     - How the order price is derived from the expected price. Default: ``"fixed"``.
   * - ``meanReversionTime`` |optional| |jsonrandom|
     - int
     - Time scale :math:`\tau_r` of the fundamental component, in steps. Truncated to an integer.
       Default: the agent's own ``timeWindowSize``.

.. dropdown:: How an FCN agent decides its orders
   :icon: light-bulb

   At step :math:`t`, for every tradable market with market price :math:`P_t` and fundamental price
   :math:`P^f_t`, the agent computes an expected log return

   .. math::

      \hat{r} = \frac{1}{w_F + w_C + w_N}
      \left(
      w_F \frac{1}{\tau_r} \ln\frac{P^f_t}{P_t}
      + w_C \frac{1}{\tau'} \ln\frac{P_t}{P_{t-\tau'}}
      + w_N \sigma_N \varepsilon
      \right),
      \qquad \varepsilon \sim \mathcal{N}(0, 1),

   with :math:`\tau' = \min(t, \tau)` (in the two denominators, values of :math:`\tau_r` and :math:`\tau'`
   below 1 are replaced by 1), and the expected price
   :math:`\hat{P} = P_t \exp(\hat{r} \tau)`. If :math:`\hat{P} > P_t` it places a buy limit order,
   if :math:`\hat{P} < P_t` a sell limit order, each of volume 1 and lifetime :math:`\tau`:

   - ``"fixed"``: buy at :math:`\hat{P}(1 - k)`, sell at :math:`\hat{P}(1 + k)`.
   - ``"normal"``: at :math:`\hat{P} + k \varepsilon'` with :math:`\varepsilon' \sim \mathcal{N}(0, 1)`.

   The chartist component always follows the trend; there is no key for contrarian behavior.

.. warning::

   The ranges of the FCN parameters are checked only when the agent first submits orders, not when the
   simulation is set up. A distribution that can produce invalid values, such as ``{"normal": [...]}`` for a
   weight, can therefore stop the simulation midway with an ``AssertionError`` (a ``ZeroDivisionError`` if all
   three weights are 0). A ``timeWindowSize`` that truncates to 0 (e.g. ``0.5``, or a range starting at 0) makes
   the agent silently never place orders, and a negative one stops the simulation with an ``AssertionError``.

MarketShareFCNAgent
~~~~~~~~~~~~~~~~~~~

``"class": "MarketShareFCNAgent"`` (:class:`pams.agents.MarketShareFCNAgent`) has exactly the same keys as
``FCNAgent``. Instead of ordering in every tradable market, it picks **one** market per decision, with a
probability proportional to the volume executed in that market in the current step and the ``timeWindowSize``
steps before it.
It needs at least one tradable market.

While any market has executions in the window, a market without executions gets almost zero probability (the
weight is the volume plus ``1e-10``), so once trading starts in one market the agents usually keep choosing it and
the other markets get no volume. If no market has executions, every market is equally likely. ``samples/market_share`` controls the
starting split with a user-defined market class that sets an initial traded volume (``tradeVolume``).

ArbitrageAgent
~~~~~~~~~~~~~~

``"class": "ArbitrageAgent"`` (:class:`pams.agents.ArbitrageAgent`) is a high-frequency agent that watches every
tradable ``IndexMarket``. When the index market price is lower than the index value (the weighted average of the
component market prices, see :ref:`IndexMarket <config-indexmarket>`) by more than ``orderThresholdPrice``, it buys the index and sells each component (limit orders
at each market's current market price); in the opposite case it does the reverse. None of its own keys accepts
JsonRandom notation (``cashAmount`` and ``assetVolume`` still do).

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``orderVolume`` |required|
     - int ≥ 1
     - Volume per component market. The index order has ``orderVolume`` × (number of components).
       Must be a JSON integer.
   * - ``orderThresholdPrice`` |required|
     - number ≥ 0
     - Minimum gap, in price units, between the index market price and the index value.
   * - ``orderTimeLength`` |optional|
     - int ≥ 1
     - Lifetime (TTL) of its orders in steps. Default: ``1``. Must be a JSON integer.

- ``markets`` must include the index market group **and** the groups of all its component markets.
  Otherwise the component orders are still sent, and the simulation stops with a ``KeyError`` (a market ID)
  when the first one is executed.
- All component markets must have the same ``outstandingShares``.
- It only trades while the index and all its components are executing orders.

MarketMakerAgent
~~~~~~~~~~~~~~~~

``"class": "MarketMakerAgent"`` is a high-frequency agent that places one buy and one sell limit order (volume 1)
in ``targetMarket`` every time it is asked, centered on the best prices of **all** its tradable markets (see
below). Old orders are not cancelled; they expire after ``orderTimeLength`` steps.

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``targetMarket`` |required|
     - market instance name
     - The market to quote in. It must also be one of the agent's tradable markets; otherwise the simulation
       stops with a ``KeyError`` (a market ID) when its first order is executed.
   * - ``netInterestSpread`` |required| |jsonrandom|
     - number
     - Full spread between the two orders, as a fraction of the fundamental price (``0.02`` = 2%).
   * - ``orderTimeLength`` |optional| |jsonrandom|
     - int ≥ 1
     - Lifetime (TTL) of its orders in steps. Truncated to an integer. Default: ``2``.

.. dropdown:: How a market maker sets its prices
   :icon: light-bulb

   The two orders are centered on the midpoint :math:`c` between the highest best buy price and the lowest best
   sell price over **all** tradable markets of the agent, not only ``targetMarket``. If no tradable market has a
   buy order, or none has a sell order, :math:`c` is the market price of ``targetMarket``. With :math:`s` =
   ``netInterestSpread`` and :math:`P^f` the fundamental price of ``targetMarket``, the agent buys at
   :math:`c - P^f s / 2` and sells at :math:`c + P^f s / 2`.

TestAgent
~~~~~~~~~

``"class": "TestAgent"`` places random orders and has no keys of its own besides ``markets``, ``cashAmount``
and ``assetVolume``. It exists to test PAMS itself and should not be used for research.


.. _config-events:

Events
------

Events change the course of the simulation: they shock prices, restrict order prices, halt trading or inject
erroneous orders. An event is defined by a named block, whose ``"class"`` (required) names the event class, and
is attached to a session by listing the block name in the session's ``events``:

.. code-block:: json

   "sessions": [
     {"sessionName": "warmup", "iterationSteps": 100, "...": "..."},
     {"sessionName": "main", "iterationSteps": 500, "...": "...", "events": ["PriceDrop"]}
   ],
   "PriceDrop": {
     "class": "FundamentalPriceShock",
     "target": "Market",
     "triggerTime": 100,
     "priceChangeRate": -0.1
   }

Timing rules:

- ``triggerTime`` counts steps **from the start of the session that lists the event**. Above, ``main`` starts
  at step 100, so the shock happens at step 200. A ``triggerTime`` beyond the end of the session fires in a
  later session, or never if the simulation has already ended.
- Rules without a trigger time (:class:`~pams.events.PriceLimitRule` and :class:`~pams.events.TradingHaltRule`)
  are active during the **whole simulation**, from step 0, whichever session lists them.
- Listing the same block in two sessions creates two independent events. List a ``PriceLimitRule`` or
  ``TradingHaltRule`` in **one** session only: it already applies to every session, and a second copy of a
  ``TradingHaltRule`` watches the same markets twice (each copy with its own halt count), which is not supported.

Every event accepts ``"enabled": false``, which switches it off without deleting it (default: ``true``).
Use it for control runs: with the same seed, a run with the event disabled matches the run with the event up to
the step where the event acts, while removing the event from ``events`` changes the random numbers from the
start.

All rates are fractions: ``0.05`` means 5%. None of the event keys accepts JsonRandom notation.

FundamentalPriceShock
~~~~~~~~~~~~~~~~~~~~~

Multiplies the fundamental price of one market by :math:`1 + r` (:class:`pams.events.FundamentalPriceShock`).
The change is permanent: the fundamental price continues its random walk from the new level. From the shock
onwards, the fundamental prices of **all** markets continue with newly drawn random numbers, so even with the same
seed a run with the shock and one without it do not share the same fundamental noise afterwards (drift and
volatility are unchanged).

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``target`` |required|
     - market instance name
     - The market whose fundamental price is changed. Cannot be an ``IndexMarket``: this is not checked at
       setup, and the simulation stops with a ``KeyError`` when the shock fires. To move an index, shock its
       component markets.
   * - ``triggerTime`` |required|
     - int
     - Step of the shock, counted from the start of the session.
   * - ``priceChangeRate`` |required|
     - float
     - Rate :math:`r`. ``-0.1`` lowers the fundamental price by 10%.
   * - ``shockTimeLength`` |optional|
     - int
     - Number of consecutive steps at which the change is applied. The effect compounds to
       :math:`(1 + r)^n`. Default: ``1``.
   * - ``enabled`` |optional|
     - bool
     - Default: ``true``.
   * - ``triggerDays`` |deprecated|
     - any
     - Replaced by ``triggerTime``. Its presence is an error.

PriceLimitRule
~~~~~~~~~~~~~~

Restricts the prices of limit orders to a band around the initial price (:class:`pams.events.PriceLimitRule`).
A limit order priced outside :math:`[P_0 (1 - r), P_0 (1 + r)]` is moved to the nearest edge of the band,
where :math:`P_0` is the market price at step 0 and stays fixed for the rest of the simulation. :math:`P_0` is
the market's ``marketPrice`` when the first session has ``withOrderExecution: false`` (as in the examples);
otherwise trading during step 0 can move it. Market orders are not affected.

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``targetMarkets`` |required|
     - list of market instance names
     - The markets whose orders are restricted. It must include **every market that receives orders**;
       an order to a market that is not listed stops the simulation with an ``AssertionError``.
   * - ``triggerChangeRate`` |required|
     - float
     - Width :math:`r` of the band, greater than 0. Must be written as a float (``1.0``, not ``1``).
   * - ``enabled`` |optional|
     - bool
     - Default: ``true``.
   * - ``referenceMarket`` |deprecated|
     - any
     - Ignored with a warning.

TradingHaltRule
~~~~~~~~~~~~~~~

A circuit breaker (:class:`pams.events.TradingHaltRule`). After an execution in one of the target markets, if
the market price has moved away from its price at step 0 (:math:`P_0`, as for ``PriceLimitRule``) by at least
:math:`P_0 \times r \times (n + 1)`,
where :math:`n` is the number of halts so far, **all** target markets are halted: orders are still accepted,
but nothing is executed. Trading resumes ``haltingTimeLength`` + 1 steps after the halt started.

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``targetMarkets`` |required|
     - list of market instance names
     - The markets that are watched and halted together. To halt markets independently, define one event
       per market. A market must not be watched by two ``TradingHaltRule`` events.
   * - ``triggerChangeRate`` |required|
     - float
     - Threshold :math:`r`, greater than 0. Must be written as a float (``1.0``, not ``1``).
   * - ``haltingTimeLength`` |required|
     - int
     - Length of a halt in steps.
   * - ``enabled`` |optional|
     - bool
     - Default: ``true``.
   * - ``referenceMarket`` |deprecated|
     - any
     - Ignored with a warning.

A halt that is still running at the end of a session continues into the next session. When it ends, execution
resumes only if the current session has ``withOrderExecution: true``.

OrderMistakeShock
~~~~~~~~~~~~~~~~~

A fat-finger error (:class:`pams.events.OrderMistakeShock`). At the trigger step, the **first order submitted
to the target market** is replaced by a large limit order: a buy at :math:`P (1 + r)` if :math:`r > 0`,
otherwise a sell at :math:`P (1 + r)`, where :math:`P` is the current market price. No new order is created;
the agent that submitted the original order owns the mistaken one.

.. list-table::
   :header-rows: 1
   :widths: 27 13 60

   * - Key
     - Value
     - Description
   * - ``target`` |required|
     - market instance name
     - The market where the mistake happens.
   * - ``triggerTime`` |required|
     - int
     - Step of the mistake, counted from the start of the session. If no order reaches the target market at
       exactly this step, nothing happens.
   * - ``priceChangeRate`` |required|
     - float
     - Rate :math:`r` relative to the current market price. Must be written as a float. ``0.0`` does not mean
       "no mistake": it places a sell at :math:`P`.
   * - ``orderVolume`` |required|
     - int
     - Volume of the mistaken order.
   * - ``orderTimeLength`` |required|
     - int
     - Lifetime (TTL) of the mistaken order in steps.
   * - ``enabled`` |optional|
     - bool
     - Default: ``true``.
   * - ``agent`` |deprecated|
     - any
     - Ignored with a warning.

The mistaken order is not restricted by a ``PriceLimitRule``.


.. _config-jsonrandom:

Random parameters (JsonRandom)
------------------------------

Agent parameters marked |jsonrandom| can be a fixed number or a distribution. When a group creates many
agents, **each agent draws its own value once** (``assetVolume`` once for each tradable market), when the simulation is
set up; the value does not change
during the simulation. This is how a single block describes a heterogeneous population of agents.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notation
     - Value
   * - ``300`` or ``{"const": [300]}``
     - Always 300.
   * - ``[a, b]`` or ``{"uniform": [a, b]}``
     - Uniform distribution with :math:`a \le x < b`.
   * - ``{"normal": [mu, sigma]}``
     - Normal distribution with mean ``mu`` and standard deviation ``sigma``. Can be negative.
   * - ``{"expon": [lam]}``
     - Exponential distribution with **mean** ``lam`` (``lam`` is the scale, not the rate).
       ``{"expon": [0.0]}`` is always 0.

.. code-block:: json

   "FCNAgents": {
     "class": "FCNAgent",
     "numAgents": 100,
     "markets": ["Market"],
     "cashAmount": {"uniform": [5000, 15000]},
     "assetVolume": [40, 61],
     "fundamentalWeight": {"expon": [1.0]},
     "chartWeight": {"expon": [0.0]},
     "noiseWeight": {"const": [1.0]},
     "noiseScale": 0.001,
     "timeWindowSize": [100, 201],
     "orderMargin": [0.0, 0.1]
   }

.. tip::

   Integer parameters (``assetVolume``, ``timeWindowSize``, ``meanReversionTime`` and ``orderTimeLength`` of
   ``MarketMakerAgent``) are **truncated**, not rounded. ``[100, 200]`` therefore gives integers from 100 to
   199; write ``[100, 201]`` to include 200.

JsonRandom works only for the agent keys marked |jsonrandom| on this page. All other keys take plain values:
market, session, event and ``simulation`` keys, and agent keys without the badge, such as ``marginType``,
``targetMarket`` and the ``ArbitrageAgent`` keys ``orderVolume``, ``orderThresholdPrice`` and
``orderTimeLength``. A distribution there fails, in some cases only while the simulation is running.


.. _config-user-classes:

User-defined classes
--------------------

Your own markets, agents and events are configured like the built-in ones. Register each class with the runner
before calling ``main()``, then refer to it by its class name:

.. code-block:: python

   from pams.agents import FCNAgent
   from pams.runners import SequentialRunner


   class MyAgent(FCNAgent):
       def setup(self, settings, accessible_markets_ids, *args, **kwargs):
           super().setup(settings, accessible_markets_ids, *args, **kwargs)
           # read your own keys from the block
           self.my_parameter = settings.get("myParameter", 1.0)


   runner = SequentialRunner(settings="config.json")
   runner.class_register(cls=MyAgent)
   runner.main()

.. code-block:: json

   "MyAgents": {"extends": "FCNAgents", "class": "MyAgent", "myParameter": 2.0}

- The block (after ``extends`` is resolved, without ``numAgents`` / ``numMarkets``, ``from``, ``to`` and
  ``prefix``) is passed to ``setup(settings=...)``, so any extra key you add is available there. To accept JsonRandom notation, draw the value in ``setup`` with
  ``JsonRandom(prng=self.prng).random(settings["myParameter"])`` (``from pams.utils import JsonRandom``; see
  :class:`~pams.utils.JsonRandom`). It always returns a float, so apply ``int()`` yourself for integer
  parameters.
- Class names must be unique: a class with the same name as a built-in class (e.g. your own ``FCNAgent``)
  is ambiguous and fails. Register each class only once.
- Agent classes must inherit from :class:`~pams.agents.Agent` and market classes from :class:`~pams.Market`.
  An agent that inherits from :class:`~pams.agents.HighFrequencyAgent` is scheduled as a high-frequency agent.
- ``samples/user_class`` and ``samples/market_share`` show complete examples.


.. _config-parallel:

Parallel runners
----------------

:class:`~pams.runners.MultiThreadAgentParallelRunner` and :class:`~pams.runners.MultiProcessAgentParallelRunner`
(both experimental) read the same config and give the same results as :class:`~pams.runners.SequentialRunner` for
the same seed. They call ``submit_orders`` of normal agents in parallel; everything else stays sequential.

- **Speed**: because of Python's GIL, the thread runner only helps when ``submit_orders`` waits for I/O (for
  example a call to an external model); it does not speed up the built-in agents. The process runner is much
  slower, because the whole simulation is copied to the worker processes in every step (see below). See also
  :doc:`platform`.
- **User-defined agents** must not change shared objects (markets, other agents, the logger) in
  ``submit_orders``. With the process runner, changes an agent makes to its own attributes in ``submit_orders``
  are lost; update such state in callbacks like ``executed_order``, which run in the main process.

The number of worker threads or processes is set by ``simulation.numParallel`` (see :ref:`config-simulation`;
default: the number of CPUs minus 1, at least 1).

At most ``maxNormalOrders`` agents are asked at the same time, so workers beyond a session's ``maxNormalOrders``
stay idle. A warning is shown when ``numParallel`` is larger than the ``maxNormalOrders`` of every session (for
example with the defaults). ``maxNormalOrders`` is part of the model: raising it changes the simulation results.
Choose it for your model first, then set ``numParallel`` to at most that value.

With the process runner, put the code that creates and runs the runner under ``if __name__ == "__main__":``
(required on Windows and macOS, and on Linux from Python 3.14, where worker processes are no longer started by
``fork``; without it the run fails or hangs), and define user-defined classes in a ``.py``
file, not in a notebook or an interactive session, so that the worker processes can import them.

``simulation.startMethod`` sets how the process runner starts its worker processes: ``"spawn"`` (the default on
Windows and macOS), ``"fork"`` (the default on Linux up to Python 3.13) or ``"forkserver"`` (the default on Linux
from Python 3.14). Only the values that :func:`multiprocessing.get_all_start_methods` returns on the platform are
accepted; any other value is an error. Without the key, the platform's default is used, unless a subclass of the
runner sets another default. Use ``"spawn"`` when agents use a library that does not work in a forked process,
such as PyTorch with CUDA, TensorFlow or JAX. The start method does not change the simulation results.

The process runner pickles the agents and the markets for each task, together with everything they refer to (the
simulator, the other agents, the events and the logger). User-defined agents, markets, events and loggers must
therefore be picklable: for example, a logger that keeps an open file fails with
``TypeError: cannot pickle '_io.TextIOWrapper' object``. Keep such data in memory and write the file after the run.

Because of these references, each task copies the whole simulation. The copy is about 4 KB per agent, mostly the
state of each agent's random number generator (about 4 MB for 1000 agents), and it takes tens of milliseconds,
much longer than ``submit_orders`` of the built-in agents. To limit this cost, the agents asked at the same time
are split into at most ``numParallel`` tasks. An object held by an agent, such as a neural network model, is copied
in every task, even in the tasks of other agents. Exclude such an object from pickling (for example with
``__getstate__``), and if it is read-only, load it once per worker process instead (see :doc:`platform`).


.. _config-troubleshooting:

Troubleshooting
---------------

Silent pitfalls
~~~~~~~~~~~~~~~

These mistakes do not raise an error but change the simulation:

- **Misspelled or misplaced keys are ignored**: ``"numMarket"`` or ``"maxNormalOrder"`` falls back to the
  default, and ``events`` written in ``simulation`` instead of in a session creates no event.
- **Only one normal agent places orders per step** by default (``maxNormalOrders: 1``).
- **High-frequency agents never act** when no normal agent submits orders in a step, or when
  ``highFrequencySubmitRate`` is ``0.0``.
- ``withPrint`` **has no effect**; use a logger such as :class:`~pams.logs.MarketStepPrintLogger`.
- **Event rules are active in every session**: a ``PriceLimitRule`` or ``TradingHaltRule`` listed only in the
  second session also applies during the first one.
- ``triggerTime`` **counts from the start of the session** that lists the event, not from step 0. A time beyond
  the end of the session fires in a later session, or never.
- ``[a, b]`` **never produces** ``b``, and integer parameters are truncated.
- **Adding, removing or reordering entries changes the random numbers.** The runner draws a seed for every
  market, agent, session and event entry in list order, so such an edit usually changes the whole run even with
  the same seed. To switch an event off for a comparison run, keep it listed with ``"enabled": false``.

Common errors
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Cause and fix
   * - ``JSONDecodeError: Expecting ...``
     - The file is not strict JSON: a trailing comma, a comment, single quotes, or ``True`` / ``False`` /
       ``None`` instead of ``true`` / ``false`` / ``null``. The message gives the line and column.
   * - ``class for X is found 0 times``
     - Unknown class name, or a user-defined class that was not registered with ``runner.class_register``.
   * - ``class for X is found 2 times``
     - Two classes have the same name (e.g. your own class named like a built-in one), or the same class was
       registered twice. Rename the class or register it once.
   * - ``X setting is missing in config``
     - A name in ``simulation.markets`` or ``simulation.agents`` has no block. Check the spelling.
   * - ``KeyError: 'X'``
     - A name was not found. Typical causes:

       - a session ``events`` entry without a block;
       - a market **instance** name (``Market-0``) in an agent's ``markets`` (use the group name);
       - a group name in ``FundamentalPriceShock.target``, ``MarketMakerAgent.targetMarket``,
         ``IndexMarket.markets`` or ``fundamentalCorrelations`` (use the instance name);
       - a missing required ``FCNAgent`` parameter.
   * - ``X does not exist`` / ``market X is not exists``
     - A name in ``targetMarkets`` of a ``PriceLimitRule`` or ``TradingHaltRule``, or the ``target`` of an
       ``OrderMistakeShock``, is not a market instance name. Use the instance name (``Market-0``), not the group
       name.
   * - ``KeyError`` with a number, e.g. ``KeyError: 0``, while the simulation runs
     - A market was used that the agent cannot trade: a ``MarketMakerAgent`` whose ``targetMarket`` is not in
       its ``markets``, or an ``ArbitrageAgent`` whose ``markets`` lack the component markets. A
       ``FundamentalPriceShock`` whose ``target`` is an ``IndexMarket`` fails the same way.
   * - ``market name X is duplicate`` / ``agent name X is duplicate``
     - Two groups produce the same instance name, e.g. a group listed twice or an inherited ``prefix``.
   * - ``X.numAgents and (X.from or X.to) cannot be used at the same time``
     - Both a count and a range are given, possibly because the count is inherited through ``extends``.
   * - ``triggerChangeRate have to be float`` (and similar)
     - Write the number as a float (``0.05``, ``1.0``), or as an integer for ``int`` keys (``100``, not
       ``100.0``).
   * - ``outstandingShares is required in component market setting``
     - A component of an ``IndexMarket`` has no ``outstandingShares``, or its group is listed after the index in
       ``simulation.markets``.
   * - ``For applying fundamental correlation fo X, fundamentalVolatility for X is required``
     - Markets in ``fundamentalCorrelations`` need ``fundamentalVolatility`` greater than 0.
   * - ``hifreqSubmitRate is replaced to highFrequencySubmitRate`` (warning)
     - Rename the deprecated key. The same applies to ``maxHifreqOrders``.
   * - ``order price does not accord to the tick size`` (warning)
     - An agent submitted a price that is not a multiple of ``tickSize``; it was rounded. This is normal for
       FCN agents.
   * - ``AssertionError`` while the simulation runs
     - Often an agent parameter out of range (see the FCNAgent warning) or an order to a market missing from a
       ``PriceLimitRule``'s ``targetMarkets``.


.. _config-samples:

Sample configurations
---------------------

The `samples <https://github.com/masanorihirano/pams/tree/main/samples>`_ directory of the repository contains
complete configs with a ``main.py`` to run them. The samples are not installed with the package, so clone the
repository and run them from its root, e.g.

.. code-block:: bash

   python -m samples.CI2002.main --config samples/CI2002/config.json --seed 1

The ``examples`` directory has Jupyter notebooks for most of these samples, with the config written as a Python
dict.

.. note::

   ``samples/CI2002``, ``samples/test`` and ``samples/user_class`` still use the deprecated session key
   ``hifreqSubmitRate``, and ``samples/market_share`` uses ``maxHifreqOrders``, so running them prints a
   deprecation warning. Use the new names in your own configs.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Sample
     - What it shows
   * - ``CI2002``
     - One market with FCN agents (Chiarella and Iori, 2002).
   * - ``shock_transfer``
     - Two spot markets built with ``extends``, an index market made of them, and ``ArbitrageAgent`` agents.
       A ``FundamentalPriceShock`` hits one spot market; compare how the three markets react.
   * - ``fat_finger``
     - ``OrderMistakeShock``.
   * - ``price_limit``
     - ``PriceLimitRule``.
   * - ``trading_halt``
     - ``FundamentalPriceShock`` and ``TradingHaltRule``.
   * - ``market_share``
     - ``MarketShareFCNAgent`` and a user-defined market class (``ExtendedMarket`` with a ``tradeVolume`` key).
       ``config-mm.json`` adds a ``MarketMakerAgent``; run it with
       ``--config samples/market_share/config-mm.json``.
   * - ``user_class``
     - A user-defined agent class registered with ``class_register``.
   * - ``test``
     - Three markets from one block (``numMarkets``, ``prefix``, ``extends``), ``fundamentalCorrelations`` and a
       ``FundamentalPriceShock`` with ``shockTimeLength``. It is a feature test: its ``fundamentalVolatility`` of
       ``0.1`` per step is far above realistic values.
