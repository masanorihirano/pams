import os
import random
from io import TextIOWrapper
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from ..agents.base import Agent
from ..events import EventABC
from ..events import EventHook
from ..index_market import IndexMarket
from ..logs.base import CancelLog
from ..logs.base import ExecutionLog
from ..logs.base import Log
from ..logs.base import Logger
from ..logs.base import MarketStepBeginLog
from ..logs.base import MarketStepEndLog
from ..logs.base import OrderLog
from ..logs.base import SessionBeginLog
from ..logs.base import SessionEndLog
from ..logs.base import SimulationBeginLog
from ..logs.base import SimulationEndLog
from ..market import Market
from ..order import Cancel
from ..order import Order
from ..session import Session
from ..simulator import Simulator
from ..utils.class_finder import find_class
from ..utils.json_extends import json_extends
from .base import Runner


class SequentialRunner(Runner):
    """Sequential runner class."""

    def __init__(
        self,
        settings: Union[Dict, TextIOWrapper, os.PathLike, str],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator,
    ):
        """Initialize.

        Args:
            settings (Union[Dict, TextIOWrapper, os.PathLike, str]): runner configuration.
            prng (random.Random, Optional): pseudo random number generator for this runner.
            logger (Logger, Optional): logger instance.
            simulator_class (Type[Simulator]): type of simulator.

        Returns:
            None

        """
        super().__init__(settings, prng, logger, simulator_class)
        self._pending_setups: List[Tuple[Callable, Dict]] = []

    def _parse_group(
        self, name: str, count_key: str
    ) -> Tuple[Dict, range, bool, str, Type]:
        """Parse the settings common to market and agent groups (internal method).

        The settings for ``name`` are extended by :func:`pams.utils.json_extends`. Then,
        ``count_key``, ``from``, ``to`` and ``prefix`` are read and removed from them, and
        the class specified by ``class`` is found.

        Args:
            name (str): name of the market or agent group in the config.
            count_key (str): key for the number of markets or agents in the group, i.e.,
                ``numMarkets`` or ``numAgents``.

        Returns:
            Tuple[Dict, range, bool, str, Type]: the extended settings, the range of the
            indices, whether the index is appended to the prefix in the names, the prefix
            of the names and the class.

        """
        if name not in self.settings:
            raise ValueError(f"{name} setting is missing in config")
        group_settings: Dict = self.settings[name]
        group_settings = json_extends(
            whole_json=self.settings,
            parent_name=name,
            target_json=group_settings,
            excludes_fields=["from", "to"],
        )
        # TODO: warn "from" and "to" is included in parent setting and not set to this setting.
        n_entities = 1
        id_from = 0
        id_to = 0
        uses_range = False
        if count_key in group_settings:
            n_entities = int(group_settings[count_key])
            id_to = n_entities - 1
        if "from" in group_settings or "to" in group_settings:
            if "from" not in group_settings or "to" not in group_settings:
                raise ValueError(
                    f"both {name}.from and {name}.to are required in json file if you use"
                )
            if count_key in group_settings:
                raise ValueError(
                    f"{name}.{count_key} and ({name}.from or {name}.to) cannot be used at the same time"
                )
            id_from = int(group_settings["from"])
            id_to = int(group_settings["to"])
            if id_to < id_from:
                raise ValueError(
                    f"{name}.to must be greater than or equal to {name}.from"
                )
            n_entities = id_to - id_from + 1
            uses_range = True
        for key in [count_key, "from", "to"]:
            if key in group_settings:
                del group_settings[key]
        prefix: str
        if "prefix" in group_settings:
            prefix = group_settings["prefix"]
            del group_settings["prefix"]
        else:
            prefix = name + ("-" if n_entities > 1 else "")
        if "class" not in group_settings:
            raise ValueError(f"class is not defined for {name}")
        if not isinstance(group_settings["class"], (str, type)):
            raise ValueError(
                f"class for {name} must be a class name (str) or a class, "
                f"but {group_settings['class']!r} is given"
            )
        group_class: Type = find_class(
            name=group_settings["class"], optional_class_list=self.registered_classes
        )
        return (
            group_settings,
            range(id_from, id_to + 1),
            uses_range or n_entities != 1,
            prefix,
            group_class,
        )

    def _generate_markets(self, market_type_names: List[str]) -> None:
        """Generate markets (internal method).

        Args:
            market_type_names (List[str]): name list of market type.

        Returns:
            None

        """
        i_market = 0
        for name in market_type_names:
            market_settings, ids, numbered, prefix, market_class = self._parse_group(
                name=name, count_key="numMarkets"
            )
            if not isinstance(market_class, type) or not issubclass(
                market_class, Market
            ):
                raise ValueError(
                    f"market class for {name} does not inherit Market class"
                )
            if "fundamentalPrice" in market_settings:
                fundamental_price = float(market_settings["fundamentalPrice"])
            elif "marketPrice" in market_settings:
                fundamental_price = float(market_settings["marketPrice"])
            else:
                raise ValueError(
                    f"fundamentalPrice or marketPrice is required for {name}"
                )
            fundamental_drift: float = 0.0
            if "fundamentalDrift" in market_settings:
                fundamental_drift = float(market_settings["fundamentalDrift"])
            fundamental_volatility: float = 0.0
            if "fundamentalVolatility" in market_settings:
                fundamental_volatility = float(market_settings["fundamentalVolatility"])

            for i in ids:
                market = market_class(
                    market_id=i_market,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if numbered else ""),
                )
                i_market += 1
                self.simulator._add_market(market=market, group_name=name)
                if not isinstance(market, IndexMarket):
                    self.simulator.fundamentals.add_market(
                        market_id=market.market_id,
                        initial=fundamental_price,
                        drift=fundamental_drift,
                        volatility=fundamental_volatility,
                    )
                self._pending_setups.append(
                    (market.setup, {"settings": market_settings})
                )

    def _generate_agents(self, agent_type_names: List[str]) -> None:
        """Generate agents (internal method).

        Args:
            agent_type_names (List[str]): name list of agent type.

        Returns:
            None

        """
        i_agent = 0
        for name in agent_type_names:
            agent_settings, ids, numbered, prefix, agent_class = self._parse_group(
                name=name, count_key="numAgents"
            )
            if not isinstance(agent_class, type) or not issubclass(agent_class, Agent):
                raise ValueError(f"agent class for {name} does not inherit Agent class")
            if "markets" not in agent_settings:
                raise ValueError(f"markets is required in {name}")
            accessible_market_names: List[str] = agent_settings["markets"]
            accessible_market_ids: List[int] = sum(
                [
                    [m.market_id for m in self.simulator.markets_group_name2market[x]]
                    for x in accessible_market_names
                ],
                [],
            )
            for i in ids:
                agent = agent_class(
                    agent_id=i_agent,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if numbered else ""),
                )
                i_agent += 1
                self.simulator._add_agent(agent=agent, group_name=name)
                self._pending_setups.append(
                    (
                        agent.setup,
                        {
                            "settings": agent_settings,
                            "accessible_markets_ids": accessible_market_ids,
                        },
                    )
                )

    def _set_fundamental_correlation(self) -> None:
        """Set fundamental correlation (internal method)."""
        # pylint: disable=too-many-nested-blocks
        if "fundamentalCorrelations" in self.settings["simulation"]:
            corr_settings: Dict = self.settings["simulation"]["fundamentalCorrelations"]
            for key, value in corr_settings.items():
                if key == "pairwise":
                    if (
                        not isinstance(value, list)
                        or sum(len(x) != 3 for x in value) > 0
                    ):
                        raise ValueError(
                            "simulation.fundamentalCorrelations.pairwise has invalid format data"
                        )
                    for market1_name, market2_name, corr in value:
                        market1 = self.simulator.name2market[market1_name]
                        market2 = self.simulator.name2market[market2_name]
                        for m in [market1, market2]:
                            if (
                                self.simulator.fundamentals.volatilities[m.market_id]
                                == 0.0
                            ):
                                raise ValueError(
                                    f"For applying fundamental correlation fo {m.name}, "
                                    f"fundamentalVolatility for {m.name} is required"
                                )
                        self.simulator.fundamentals.set_correlation(
                            market_id1=market1.market_id,
                            market_id2=market2.market_id,
                            corr=float(corr),
                        )
                else:
                    raise NotImplementedError(
                        f"{key} for simulation.fundamentalCorrelations is not supported"
                    )

    def _generate_sessions(self) -> None:
        """Generate sessions (internal method)."""
        if "sessions" not in self.settings["simulation"]:
            raise ValueError("sessions is missing under 'simulation' config")
        session_settings: Dict = self.settings["simulation"]["sessions"]
        if not isinstance(session_settings, list):
            raise ValueError("simulation.sessions must be list[dict]")
        i_session = 0
        i_event = 0
        session_start_time: int = 0
        for session_setting in session_settings:
            if "sessionName" not in session_setting:
                raise ValueError(
                    "for each element in simulation.sessions must have sessionName"
                )
            session = Session(
                session_id=i_session,
                prng=random.Random(self._prng.randint(0, 2**31)),
                session_start_time=session_start_time,
                simulator=self.simulator,
                name=str(session_setting["sessionName"]),
                logger=self.logger,
            )
            i_session += 1
            if "iterationSteps" not in session_setting:
                raise ValueError(
                    "iterationSteps is required in each element of simulation.sessions"
                )
            session_start_time += session_setting["iterationSteps"]
            self.simulator._add_session(session=session)
            self._pending_setups.append((session.setup, {"settings": session_setting}))
            if "events" in session_setting:
                event_names: List[str] = session_setting["events"]
                for event_name in event_names:
                    event_setting = self.settings[event_name]
                    event_setting = json_extends(
                        whole_json=self.settings,
                        parent_name=event_name,
                        target_json=event_setting,
                        excludes_fields=["numMarkets", "from", "to", "prefix"],
                    )
                    if "class" not in event_setting:
                        raise ValueError(f"class is required in {event_name}")
                    if not isinstance(event_setting["class"], (str, type)):
                        raise ValueError(
                            f"class for {event_name} must be a class name (str) "
                            f"or a class, but {event_setting['class']!r} is given"
                        )
                    event_class: Type[EventABC] = find_class(
                        name=event_setting["class"],
                        optional_class_list=self.registered_classes,
                    )
                    if not isinstance(event_class, type) or not issubclass(
                        event_class, EventABC
                    ):
                        raise ValueError(
                            f"event class for {event_name} does not inherit EventABC class"
                        )
                    event = event_class(
                        event_id=i_event,
                        prng=random.Random(self._prng.randint(0, 2**31)),
                        session=session,
                        simulator=self.simulator,
                        name=event_name,
                    )
                    i_event += 1
                    self._pending_setups.append(
                        (event.setup, {"settings": event_setting})
                    )

                    def event_hook_setup(_event: EventABC):
                        event_hooks: List[EventHook] = _event.hook_registration()
                        for event_hook in event_hooks:
                            self.simulator._add_event(event_hook)

                    self._pending_setups.append((event_hook_setup, {"_event": event}))

    def _setup(self) -> None:
        """Runner setup (internal method)."""
        if "simulation" not in self.settings:
            raise ValueError("simulation is required in json file")

        if "markets" not in self.settings["simulation"]:
            raise ValueError("simulation.markets is required in json file")
        market_type_names: List[str] = self.settings["simulation"]["markets"]
        if (
            not isinstance(market_type_names, list)
            or sum(not isinstance(m, str) for m in market_type_names) > 0
        ):
            raise ValueError("simulation.markets in json file have to be list[str]")
        self._generate_markets(market_type_names=market_type_names)
        self._set_fundamental_correlation()

        if "agents" not in self.settings["simulation"]:
            raise ValueError("simulation.agents is required in json file")
        agent_type_names: List[str] = self.settings["simulation"]["agents"]
        if (
            not isinstance(agent_type_names, list)
            or sum(not isinstance(m, str) for m in agent_type_names) > 0
        ):
            raise ValueError("simulation.agents in json file have to be list[str]")
        self._generate_agents(agent_type_names=agent_type_names)

        if "sessions" not in self.settings["simulation"]:
            raise ValueError("simulation.sessions is required in json file")
        session_settings: List[Dict[str, Any]] = self.settings["simulation"]["sessions"]
        if (
            not isinstance(session_settings, list)
            or sum(not isinstance(m, dict) for m in session_settings) > 0
        ):
            raise ValueError("simulation.sessions in json file have to be List[Dict]")
        self._generate_sessions()

        _ = [func(**kwargs) for func, kwargs in self._pending_setups]

    def _check_submitted_orders(
        self, agent: Agent, orders: List[Union[Order, Cancel]]
    ) -> None:
        """Check the orders submitted by an agent (internal method).

        Every order has to be submitted by the agent itself and be for an existing market that the agent can
        access. For a cancel order, the order to be canceled is checked. Orders created by events are not checked
        because they are not submitted by agents.

        Args:
            agent (Agent): agent that submitted the orders.
            orders (List[Union[Order, Cancel]]): orders submitted by the agent.

        Returns:
            None

        """
        if sum(order.agent_id != agent.agent_id for order in orders) > 0:
            raise ValueError(
                "spoofing order is not allowed. please check agent_id in order"
            )
        for order in orders:
            order_kind: str = "cancel order" if isinstance(order, Cancel) else "order"
            if order.market_id not in self.simulator.id2market:
                raise ValueError(
                    f"{order_kind} for a nonexistent market is not allowed. "
                    f"{agent.name} submitted it for market_id {order.market_id}. "
                    "please check market_id in order"
                )
            if not agent.is_market_accessible(market_id=order.market_id):
                market: Market = self.simulator.id2market[order.market_id]
                # the markets setting of agents takes the group names of markets
                market_groups = self.simulator.markets_group_name2market
                market_group_name: str = next(
                    (name for name, group in market_groups.items() if market in group),
                    market.name,
                )
                agent_groups = self.simulator.agents_group_name2agent
                agent_group_name: str = next(
                    (name for name, group in agent_groups.items() if agent in group),
                    agent.name,
                )
                raise ValueError(
                    f"{order_kind} for an inaccessible market is not allowed. "
                    f"{agent.name} cannot access {market.name}. "
                    f"please add {market_group_name} to markets of {agent_group_name} "
                    "or check market_id in order"
                )

    def _collect_orders_from_normal_agents(
        self, session: Session
    ) -> List[List[Union[Order, Cancel]]]:
        """Collect orders from normal_agents (internal method).

        orders are collected until the total number of orders reaches max_normal_orders

        Args:
            session (Session): session.

        Returns:
            List[List[Union[Order, Cancel]]]: orders lists.

        """
        agents = self.simulator.normal_frequency_agents
        agents = self._prng.sample(agents, len(agents))
        n_orders = 0
        all_orders: List[List[Union[Order, Cancel]]] = []
        for agent in agents:
            if n_orders >= session.max_normal_orders:
                break
            orders = agent.submit_orders(markets=self.simulator.markets)
            if len(orders) > 0:
                if not session.with_order_placement:
                    raise AssertionError("currently order is not accepted")
                self._check_submitted_orders(agent=agent, orders=orders)
                all_orders.append(orders)
                # TODO: currently the original impl is used
                # n_orders += len(orders)
                n_orders += 1
        return all_orders

    def _process_order(self, session: Session, order: Union[Order, Cancel]) -> None:
        """Process one order or cancel order and the executions caused by it (internal method).

        Args:
            session (Session): session.
            order (Union[Order, Cancel]): order or cancel order.

        Returns:
            None

        """
        if not session.with_order_placement:
            raise AssertionError("currently order is not accepted")
        market: Market = self.simulator.id2market[order.market_id]
        agent: Agent
        if isinstance(order, Order):
            self.simulator._trigger_event_before_order(order=order)
            log: OrderLog = market._add_order(order=order)
            agent = self.simulator.id2agent[order.agent_id]
            agent.submitted_order(log=log)
            self.simulator._trigger_event_after_order(order_log=log)
        elif isinstance(order, Cancel):
            self.simulator._trigger_event_before_cancel(cancel=order)
            log_: CancelLog = market._cancel_order(cancel=order)
            agent = self.simulator.id2agent[order.order.agent_id]
            agent.canceled_order(log=log_)
            self.simulator._trigger_event_after_cancel(cancel_log=log_)
        else:
            raise NotImplementedError
        if session.with_order_execution and market.is_running:
            logs: List[ExecutionLog] = market._execution()
            self.simulator._update_agents_for_execution(execution_logs=logs)
            for execution_log in logs:
                agent = self.simulator.id2agent[execution_log.buy_agent_id]
                agent.executed_order(log=execution_log)
                agent = self.simulator.id2agent[execution_log.sell_agent_id]
                agent.executed_order(log=execution_log)
                self.simulator._trigger_event_after_execution(
                    execution_log=execution_log
                )

    def _handle_high_frequency_orders(
        self, session: Session
    ) -> List[List[Union[Order, Cancel]]]:
        """Collect and process the orders from high frequency agents (internal method).

        High frequency agents are asked in random order until the number of agents submitting orders
        reaches max_high_frequency_orders. The orders are processed immediately.

        Args:
            session (Session): session.

        Returns:
            List[List[Union[Order, Cancel]]]: order lists submitted by high frequency agents.

        """
        all_orders: List[List[Union[Order, Cancel]]] = []
        n_high_freq_orders = 0
        agents = self.simulator.high_frequency_agents
        agents = self._prng.sample(agents, len(agents))
        for agent in agents:
            if n_high_freq_orders >= session.max_high_frequency_orders:
                break
            high_freq_orders: List[Union[Order, Cancel]] = agent.submit_orders(
                markets=self.simulator.markets
            )
            if len(high_freq_orders) == 0:
                continue
            if not session.with_order_placement:
                raise AssertionError("currently order is not accepted")
            self._check_submitted_orders(agent=agent, orders=high_freq_orders)
            all_orders.append(high_freq_orders)
            # TODO: currently the original impl is used
            n_high_freq_orders += 1
            # n_high_freq_orders += len(high_freq_orders)
            for order in high_freq_orders:
                self._process_order(session=session, order=order)
        return all_orders

    def _handle_orders(
        self, session: Session, local_orders: List[List[Union[Order, Cancel]]]
    ) -> List[List[Union[Order, Cancel]]]:
        """Handle orders (internal method).

        processing local orders and correct and process the orders from high frequency agents.

        Args:
            session (Session): session.
            local_orders (List[List[Union[Order, Cancel]]]): local orders.

        Returns:
            List[List[Union[Order, Cancel]]]: order lists.

        """
        sequential_orders = self._prng.sample(local_orders, len(local_orders))
        all_orders: List[List[Union[Order, Cancel]]] = [*sequential_orders]
        for orders in sequential_orders:
            for order in orders:
                self._process_order(session=session, order=order)

            if session.high_frequency_submission_rate < self._prng.random():
                continue
            all_orders.extend(self._handle_high_frequency_orders(session=session))
        return all_orders

    def _update_markets(self, session: Session) -> None:
        """Update markets (internal method).

        Args:
            session (Session): session.

        Returns:
            None

        """
        local_orders: List[
            List[Union[Order, Cancel]]
        ] = self._collect_orders_from_normal_agents(session=session)
        self._handle_orders(session=session, local_orders=local_orders)

    def _iterate_market_updates(self, session: Session) -> None:
        """Iterate market updates (internal method).

        Args:
            session (Session): session.

        Returns:
            None

        """
        markets: List[Market] = self.simulator.markets
        for market in markets:
            market._is_running = session.with_order_execution

        for _ in range(session.iteration_steps):
            for market in markets:
                self.simulator._trigger_event_before_step_for_market(market=market)
                if self.logger is not None:
                    log: Log = MarketStepBeginLog(
                        session=session, market=market, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
            if session.with_order_placement:
                self._update_markets(session=session)
            for market in markets:
                if self.logger is not None:
                    log = MarketStepEndLog(
                        session=session, market=market, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
                self.simulator._trigger_event_after_step_for_market(market=market)
            self.simulator._update_times_on_markets(self.simulator.markets)  # t++

    def _run(self) -> None:
        """Main process (internal method)."""
        if self.logger is not None:
            log: Log = SimulationBeginLog(simulator=self.simulator)  # must be blocking
            log.read_and_write(logger=self.logger)
            self.logger._process()
        self.simulator._update_times_on_markets(self.simulator.markets)  # t: -1 -> 0

        for session in self.simulator.sessions:
            self.simulator.current_session = session
            self.simulator._trigger_event_before_session(session=session)
            if self.logger is not None:
                log = SessionBeginLog(
                    session=session, simulator=self.simulator
                )  # must be blocking
                log.read_and_write(logger=self.logger)
                self.logger._process()
            self._iterate_market_updates(session=session)
            self.simulator._trigger_event_after_session(session=session)
            if self.logger is not None:
                log = SessionEndLog(
                    session=session, simulator=self.simulator
                )  # must be blocking
                log.read_and_write(logger=self.logger)
                self.logger._process()
        if self.logger is not None:
            log = SimulationEndLog(simulator=self.simulator)  # must be blocking
            log.read_and_write(logger=self.logger)
            self.logger._process()
