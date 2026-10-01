.. _tutorial-custom-event:

Writing an event or a market
============================

Events change the simulation while it runs: the built-in events shock the fundamental price, limit order prices,
halt trading or inject erroneous orders (see :ref:`config-events`).
This tutorial writes an event that moves the fundamental price at random at regular intervals, like news, and
then shows how to extend the market itself.
The complete script is :download:`tutorial_custom_event.py <code/tutorial_custom_event.py>`.


How the runner uses an event
----------------------------

An event is attached to a session by listing the name of its block in the session's ``events``. For each name
listed, the runner:

1. creates the event. Its ``name`` is the name of the block, its ``session`` is the session that lists it, and
   it has its own random number generator ``prng`` and the ``simulator``;
2. calls ``setup(settings)`` with the block (after ``extends`` is resolved), after all the markets and agents
   are set up;
3. calls ``hook_registration()`` right after ``setup``. It returns a list of :class:`~pams.events.EventHook`
   that tell the simulator which methods of the event to call, and when.

Each hook has a ``hook_type`` and ``is_before``, which select the method that is called:

.. list-table::
   :header-rows: 1
   :widths: 14 43 43

   * - ``hook_type``
     - ``is_before=True``
     - ``is_before=False``
   * - ``"order"``
     - ``hooked_before_order(simulator, order)``, before a market receives an order. The event can change the
       :class:`~pams.Order` here.
     - ``hooked_after_order(simulator, order_log)``, after a market has received an order.
   * - ``"cancel"``
     - ``hooked_before_cancel(simulator, cancel)``
     - ``hooked_after_cancel(simulator, cancel_log)``
   * - ``"execution"``
     - Not allowed.
     - ``hooked_after_execution(simulator, execution_log)``, after each execution.
   * - ``"session"``
     - ``hooked_before_session(simulator, session)``, before the first step of a session.
     - ``hooked_after_session(simulator, session)``, after the last step of a session.
   * - ``"market"``
     - ``hooked_before_step_for_market(simulator, market)``, before the agents are asked for orders in a step.
     - ``hooked_after_step_for_market(simulator, market)``, at the end of a step.

The hook's ``time`` is the list of steps at which it is active. For a session hook, this is the first step of
the session (before) or its last step (after).
The steps are counted from the start of the simulation, not from the start of the session.
With ``time=None`` (the default), the hook is active at every step of **every** session, not only in the session
that lists the event.

A ``"market"`` hook also accepts ``specific_instance`` (one :class:`~pams.Market`) or ``specific_class`` (a
market class) to be called only for these markets. Without them, it is called for every market.


The event
---------

``NewsShock`` multiplies the fundamental price of its ``target`` market by :math:`e^{x}` every ``interval``
steps of its session, where :math:`x` is drawn from a normal distribution with mean 0 and standard deviation
``jumpScale``:

.. literalinclude:: code/tutorial_custom_event.py
   :language: python
   :start-after: # [event-start]
   :end-before: # [event-end]

- ``setup`` checks and reads the keys of its block. The markets already exist, so it can look up its target
  with ``self.simulator.name2market``.
- ``hook_registration`` uses ``self.session`` to keep its hooks in its own session: the news hook is active at
  the steps ``start + interval``, ``start + 2 * interval``, ... before the end of the session, and only for the
  target market. The session hook is active at the last step of the session. With ``time=None``, the session hook
  would also be called after the ``warmup`` session.
- ``change_fundamental_price`` changes the fundamental price of the current step, before the agents are asked for
  orders. The change is permanent: the fundamental price stays at the new level, or continues its random walk
  from it when the market has a ``fundamentalVolatility``.
- The event draws its random numbers from ``self.prng``, so the simulation stays determined by its seed.


Using the event
---------------

The ``main`` session lists the event, and the block gives its class and its parameters. The rest is the
configuration of :doc:`first_simulation`:

.. literalinclude:: code/tutorial_custom_event.py
   :language: python
   :start-after: # [config-start]
   :end-before: # [config-end]

As with agents, register the class with the runner before calling ``main()``:

.. literalinclude:: code/tutorial_custom_event.py
   :language: python
   :start-after: # [run-start]
   :end-before: # [run-end]

After the run, ``runner.simulator.name2event["News"]`` is the event. Running the complete script prints the
summary of the event at the end of ``main``, the two time lines of ``main()``, then the fundamental price before
and after each news:

.. code-block:: text

   News: 9 news in main, fundamental price 291.61
   # INITIALIZATION TIME 0.005436
   # EXECUTION TIME 0.1829801
   step 150: 300.00 -> 295.17
   step 200: 295.17 -> 302.24
   step 250: 302.24 -> 301.98
   step 300: 301.98 -> 294.94
   step 350: 294.94 -> 298.95
   step 400: 298.95 -> 298.90
   step 450: 298.90 -> 292.69
   step 500: 292.69 -> 286.11
   step 550: 286.11 -> 291.61

The ``main`` session starts at step 100, so the first news is at step 150.


Writing a market
----------------

Markets are extended in the same way: inherit from :class:`~pams.Market`, call ``super().setup`` and read your
own keys from the block. ``samples/market_share`` uses this market:

.. literalinclude:: ../../../../samples/market_share/main.py
   :language: python
   :pyobject: ExtendedMarket

It sets the traded volume of step 0 from the key ``tradeVolume``. In this sample,
:class:`~pams.agents.MarketShareFCNAgent` chooses the market of each order at random, weighted by the volume
traded in each market since ``timeWindowSize`` steps ago, so ``tradeVolume`` sets the initial market shares.
The sample registers the class with ``runner.class_register(cls=ExtendedMarket)``, and its configuration uses
``"class": "ExtendedMarket"`` in the market blocks.

Next, :doc:`custom_logger` records the results of a simulation in your own format.
