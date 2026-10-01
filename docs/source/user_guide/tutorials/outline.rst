Outline
=====================

Guideline
~~~~~~~~~~~~~~~~~~~
Here is what you should take care for your implementations:

- **Be careful to change parameters in other classes.**
  Because of python system, all parameters can be changeable.
  However, it doesn't mean that you should change them.
  Only referring those parameters outside the class is somehow reasonable, but changing them is dangerous for some situations.
  The names of parameters that should not be changed outside are started from "_".
  Therefore, the parameter whose name is started from "_" is highly forbidden to access outside the class.
- **Publicly accessible parameters and methods don't mean you should access them.**
  For some convenience, we make some parameters and methods are public starting without "_".
  However, if you should access them is always depending on what you want to do.
  Therefore, some accessible parameters and methods could not be appropriate to be used correspondence to the actual markets.

The tutorials
~~~~~~~~~~~~~~~~~~~
The first four tutorials start from the same small simulation, one market and 100 FCN agents:

- :doc:`first_simulation`: run a simulation from Python, save and read its results, and plot the prices.
- :doc:`custom_agent`: write your own agent, and change a built-in one.
- :doc:`custom_event`: write an event that changes the simulation while it runs, and extend a market.
- :doc:`custom_logger`: write a logger that records the results in your own format.

The next tutorials follow the tutorials of `Plham <https://plham.github.io/>`_, the artificial market simulator
whose concept PAMS inherits, and run their research cases with PAMS. Each page links to the Plham tutorial it
follows, cites the papers of the case, and says where the results of PAMS differ:

- :doc:`ci2002`: how fundamentalist, chartist and noise components shape the price.
- :doc:`price_limit`: a price limit, and how long the price stays at it.
- :doc:`trading_halt`: trading halts after a fall of the fundamental price.
- :doc:`fat_finger`: a large sell order placed by mistake, and the order book after it.
- :doc:`shock_transfer`: a shock that spreads from one stock to another through an index and arbitrage agents.
- :doc:`market_share`: two markets competing for volume, with a smaller tick size or a market maker.

Plham also has tutorials on a dark pool, on options and on parallel runs of many assets. PAMS has no dark pool
market and no options, so these cases have no tutorial here; to run PAMS in parallel, see :ref:`config-parallel`.
The Plham tutorials on setting up a project and on the JSON format correspond to the first four tutorials and to
:ref:`config`.

Each tutorial comes with a complete script that you can download and run. Every key of the configuration is
explained in :ref:`config`.
