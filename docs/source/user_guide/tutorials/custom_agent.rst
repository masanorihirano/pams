.. _tutorial-custom-agent:

Writing an agent
================

This tutorial writes two agents and adds them to the simulation of :doc:`first_simulation`:

- ``MovingAverageAgent``, a new strategy written from :class:`~pams.agents.Agent`;
- ``ContrarianFCNAgent``, a small change to :class:`~pams.agents.FCNAgent`.

The complete script is :download:`tutorial_custom_agent.py <code/tutorial_custom_agent.py>`.


How the runner uses an agent
----------------------------

For each agent of the configuration, the runner:

1. creates the agent with an ``agent_id``, a ``name`` (such as ``"MovingAverageAgents-0"``), the
   ``simulator``, the ``logger``, and its own random number generator ``prng``. You rarely need to change
   ``__init__``;
2. calls ``setup(settings, accessible_markets_ids)`` once, after all the markets and agents are created.
   ``settings`` is the agent's block of the configuration, after ``extends`` is resolved and without
   ``numAgents``, ``from``, ``to`` and ``prefix``. All the agents of a group receive the same dict, so read it
   but do not change it. ``accessible_markets_ids`` are the IDs of the markets in the groups listed in the
   block's ``markets``;
3. calls ``submit_orders(markets)`` in some of the steps. The agent returns the orders it wants to place or
   cancel, possibly none;
4. calls ``submitted_order``, ``canceled_order`` and ``executed_order`` when a market has accepted one of its
   orders, has accepted one of its cancel orders, or has executed one of its orders.

In each step, the runner asks the agents in a random order until ``maxNormalOrders`` of them (``1`` by default,
see :ref:`config-sessions`) have returned at least one order. An agent that returns no order does not count, and
the runner asks the next one. So, in a simulation with many agents, each agent places orders only in a few
steps.

An agent must inherit from :class:`~pams.agents.Agent` and implement ``submit_orders``. The callbacks do nothing
by default.


Reading the parameters
----------------------

``MovingAverageAgent`` compares the market price with the average price of the last ``windowSize`` steps.
It reads its parameters in ``setup``:

.. literalinclude:: code/tutorial_custom_agent.py
   :language: python
   :start-after: # [setup-start]
   :end-before: # [setup-end]
   :dedent: 4

- ``super().setup(...)`` reads ``cashAmount`` and ``assetVolume``, which every agent requires, and gives the agent
  its cash and its shares in each accessible market.
- Any other key of the block can be read from ``settings``. :class:`~pams.utils.JsonRandom` accepts the same
  notation as the built-in agents, such as ``[10, 30]`` for a uniform random value (see
  :ref:`config-jsonrandom`), and returns a float. ``int()`` truncates it, so ``[10, 30]`` gives windows from 10
  to 29.
- Declare the attributes of the agent in the class body and set them in ``setup``.
- Use ``self.prng`` for every random choice of the agent, so that the simulation stays determined by its seed.


Placing orders
--------------

``submit_orders`` receives every market of the simulation and returns a list of orders:

.. literalinclude:: code/tutorial_custom_agent.py
   :language: python
   :start-after: # [submit-start]
   :end-before: # [submit-end]
   :dedent: 4

- Skip the markets the agent cannot trade, with ``is_market_accessible``.
- Only the current step and the steps before it can be read: ``market.get_market_price()`` is the price now,
  and ``get_market_prices(times=...)`` gives the prices of the earlier steps.
- Every order needs ``agent_id=self.agent_id``: an order with another agent's ID raises
  ``ValueError: spoofing order is not allowed. please check agent_id in order``.
- A :obj:`~pams.LIMIT_ORDER` needs a ``price``, and a :obj:`~pams.MARKET_ORDER` must not have one
  (``price=None``). ``volume`` must be positive.
- Round limit prices to the tick size with ``convert_to_tick_level`` and ``convert_to_price``. Otherwise the
  market rounds them for you (buy orders down, sell orders up) with a warning.
- Without ``ttl``, an order stays in the order book until it is executed or canceled. With ``ttl=n``, it is
  removed once more than ``n`` steps have passed since it was placed.
- To cancel an order, return ``Cancel(order=...)`` with the :class:`~pams.Order` you submitted earlier.
  The market reduces ``order.volume`` as the order is executed, so an order without ``ttl`` is still in the
  order book while its ``volume`` is positive. An order removed because of its ``ttl`` keeps its ``volume``.
  Canceling an order that is not in the order book any more is not an error.
- Create a new :class:`~pams.Order` for every order: submitting the same object again raises
  ``ValueError: the order is already submitted``.
- The orders of one agent are processed in the order of the list, so the cancel above is processed before the
  new order.
- The market does not check cash or shares: an agent can buy without cash and sell shares it does not have.
  Add such checks to ``submit_orders`` if your model needs them.


Following the results
---------------------

The callbacks tell the agent what happened to its orders:

.. literalinclude:: code/tutorial_custom_agent.py
   :language: python
   :start-after: # [callbacks-start]
   :end-before: # [callbacks-end]
   :dedent: 4

``executed_order`` is called for both the buyer and the seller of an execution, after their cash and shares are
updated. There is no callback for an order that expires because of its ``ttl``.


Changing an existing agent
--------------------------

To change a built-in agent, inherit from it and override only what differs.
:class:`~pams.agents.FCNAgent` combines a fundamental, a chart and a noise term. Its chart term follows the past
trend when ``is_chart_following`` is ``True``, which is the default. ``ContrarianFCNAgent`` reverses it:

.. literalinclude:: code/tutorial_custom_agent.py
   :language: python
   :start-after: # [fcn-start]
   :end-before: # [fcn-end]


Using the agents
----------------

The configuration adds a group of each new agent to the one of :doc:`first_simulation`:

.. literalinclude:: code/tutorial_custom_agent.py
   :language: python
   :start-after: # [config-start]
   :end-before: # [config-end]

``ContrarianAgents`` copies the block of ``FCNAgents`` with ``extends``, then changes its class, its number of
agents and its ``chartWeight`` (the chart term does nothing when its weight is 0).
``MovingAverageAgents`` gives the keys that ``Agent.setup`` requires, and the keys that ``MovingAverageAgent``
reads.

Register the new classes with the runner before calling ``main()``. The ``class`` of a block is the name of the
registered class:

.. literalinclude:: code/tutorial_custom_agent.py
   :language: python
   :start-after: # [run-start]
   :end-before: # [run-end]

``report(run_simulation(seed=42))`` prints the two time lines of ``main()``, then:

.. code-block:: text

   MovingAverageAgents-0: window 12, orders 2, cancels 0, trades 2, shares 50, cash 9989.21
   MovingAverageAgents-1: window 27, orders 1, cancels 0, trades 1, shares 51, cash 9697.14
   MovingAverageAgents-2: window 14, orders 4, cancels 0, trades 4, shares 52, cash 9394.82
   MovingAverageAgents-3: window 13, orders 1, cancels 0, trades 1, shares 49, cash 10295.09
   MovingAverageAgents-4: window 11, orders 4, cancels 0, trades 4, shares 48, cash 10588.81
   MovingAverageAgents-5: window 12, orders 4, cancels 1, trades 3, shares 51, cash 9685.83
   MovingAverageAgents-6: window 12, orders 2, cancels 0, trades 2, shares 50, cash 9994.57
   MovingAverageAgents-7: window 10, orders 6, cancels 0, trades 6, shares 46, cash 11195.17
   MovingAverageAgents-8: window 22, orders 5, cancels 2, trades 3, shares 49, cash 10293.29
   MovingAverageAgents-9: window 12, orders 5, cancels 0, trades 5, shares 51, cash 9684.50

The moving average agents place no order in ``warmup``: the market price does not move while orders are not
executed, so it equals its average. In ``main``, only one agent places orders in each step, as explained above,
and the agents share these steps with the 120 FCN agents. To make them trade more, raise ``maxNormalOrders`` in
the sessions, or add more of them.

Class names must be unique: do not name your class like a built-in class, such as ``FCNAgent``.
:ref:`config-user-classes` gives more details.


Notes
-----

- An agent that inherits from :class:`~pams.agents.HighFrequencyAgent` is a high-frequency agent. The runner asks
  high-frequency agents after the orders of each normal agent, as set by ``maxHighFrequencyOrders`` and
  ``highFrequencySubmitRate`` (see :ref:`config-sessions`).
- ``MovingAverageAgent`` keeps its last orders in an attribute set in ``submit_orders``. This works with
  :class:`~pams.runners.SequentialRunner` and the thread runner, but not with the process runner, which loses
  such changes (see :ref:`config-parallel`).

Next, :doc:`custom_event` changes the market during the simulation, and :doc:`custom_logger` records the trades.
