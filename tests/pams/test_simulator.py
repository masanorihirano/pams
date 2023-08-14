import random
from typing import List

import pytest

from pams import LIMIT_ORDER
from pams import Cancel
from pams import IndexMarket
from pams import Market
from pams import Order
from pams import Session
from pams import Simulator
from pams.agents import Agent
from pams.agents import ArbitrageAgent
from pams.agents import FCNAgent
from pams.events import EventABC
from pams.events import EventHook
from pams.events import FundamentalPriceShock
from pams.logs import CancelLog
from pams.logs import ExecutionLog
from pams.logs import Logger
from pams.logs import OrderLog


class TestSimulator:
    def test__init(self) -> None:
        prng = random.Random(42)
        logger = Logger()
        sim = Simulator(prng=prng, logger=logger)
        assert sim._prng == prng
        assert sim.logger == logger

    def test_add(self) -> None:
        prng = random.Random(42)
        logger = Logger()
        sim = Simulator(prng=prng, logger=logger)
        session = Session(
            session_id=0,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        sim._add_session(session=session)
        with pytest.raises(ValueError):
            sim._add_session(session=session)
        session = Session(
            session_id=0,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        with pytest.raises(ValueError):
            sim._add_session(session=session)
        session = Session(
            session_id=1,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        with pytest.raises(ValueError):
            sim._add_session(session=session)
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        sim._add_market(market=market, group_name="market_group")
        with pytest.raises(ValueError):
            sim._add_market(market=market)
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        with pytest.raises(ValueError):
            sim._add_market(market=market)
        market = Market(
            market_id=1, prng=random.Random(42), simulator=sim, name="market"
        )
        with pytest.raises(ValueError):
            sim._add_market(market=market)
        event = FundamentalPriceShock(
            event_id=0,
            prng=random.Random(),
            session=session,
            simulator=sim,
            name="event1",
        )
        event.setup(
            settings={"target": "market", "triggerTime": 0, "priceChangeRate": -0.1}
        )
        event_hooks = event.hook_registration()
        for event_hook in event_hooks:
            sim._add_event(event_hook)
        with pytest.raises(ValueError):
            sim._add_event(event_hooks[0])
        agent = FCNAgent(
            agent_id=0, prng=random.Random(42), simulator=sim, name="agent"
        )
        sim._add_agent(agent=agent, group_name="agent_group")
        agent2 = ArbitrageAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="arbitrage"
        )
        sim._add_agent(agent=agent2, group_name="agent_group")
        with pytest.raises(ValueError):
            sim._add_agent(agent=agent)
        agent = FCNAgent(
            agent_id=0, prng=random.Random(42), simulator=sim, name="agent"
        )
        with pytest.raises(ValueError):
            sim._add_agent(agent=agent)
        agent = FCNAgent(
            agent_id=2, prng=random.Random(42), simulator=sim, name="agent"
        )
        with pytest.raises(ValueError):
            sim._add_agent(agent=agent)

    def test_update_time_on_market(self) -> None:
        sim = Simulator(prng=random.Random(32))
        im = IndexMarket(market_id=2, prng=random.Random(42), simulator=sim, name="im")
        m1 = Market(market_id=0, prng=random.Random(42), simulator=sim, name="market1")
        m1.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 100,
                "marketPrice": 300.0,
                "fundamentalPrice": 500.0,
            }
        )
        im._add_market(market=m1)
        m2 = Market(market_id=1, prng=random.Random(42), simulator=sim, name="market2")
        m2.setup(
            settings={
                "tickSize": 0.001,
                "outstandingShares": 300,
                "marketPrice": 400.0,
                "fundamentalPrice": 400.0,
            }
        )
        im._add_market(market=m2)
        sim._add_market(market=m1)
        sim._add_market(market=m2)
        sim._add_market(market=im)
        sim.fundamentals.add_market(
            market_id=m1.market_id, initial=500.0, drift=0.0, volatility=0.0
        )
        sim.fundamentals.add_market(
            market_id=m2.market_id, initial=400.0, drift=0.0, volatility=0.0
        )
        sim._update_times_on_markets(markets=[m1, m2, im])
        assert im.compute_fundamental_index() == 425.0

    def test_update_agents_for_execution(self) -> None:
        sim = Simulator(prng=random.Random(32))
        agent_buy = FCNAgent(
            agent_id=0, prng=random.Random(42), simulator=sim, name="buy_agent"
        )
        agent_sell = FCNAgent(
            agent_id=1, prng=random.Random(42), simulator=sim, name="sell_agent"
        )
        agent_buy.cash_amount = 1000
        agent_sell.cash_amount = 1000
        agent_buy.asset_volumes = {0: 100}
        agent_sell.asset_volumes = {0: 100}
        sim._add_agent(agent_buy)
        sim._add_agent(agent_sell)
        execution_log = ExecutionLog(
            market_id=0,
            time=1,
            buy_agent_id=0,
            sell_agent_id=1,
            buy_order_id=110,
            sell_order_id=111,
            price=90.9,
            volume=2,
        )
        sim._update_agents_for_execution(execution_logs=[execution_log])
        assert agent_buy.cash_amount == 1000 - 90.9 * 2
        assert agent_sell.cash_amount == 1000 + 90.9 * 2
        assert agent_buy.asset_volumes == {0: 102}
        assert agent_sell.asset_volumes == {0: 98}

    def test_check_event_class_and_instance(self) -> None:
        sim = Simulator(prng=random.Random(42))
        market = IndexMarket(
            market_id=0, prng=random.Random(42), simulator=sim, name="index_market"
        )
        assert sim._check_event_class_and_instance(
            check_object=market, class_requirement=Market
        )
        assert sim._check_event_class_and_instance(
            check_object=market, instance_requirement=market
        )
        assert sim._check_event_class_and_instance(
            check_object=market, class_requirement=Market, instance_requirement=market
        )
        assert not sim._check_event_class_and_instance(
            check_object=market, class_requirement=Agent
        )
        assert not sim._check_event_class_and_instance(
            check_object=market, class_requirement=Market, instance_requirement=sim
        )

    def test_triggers(self) -> None:
        class DummyEvent(EventABC):
            def __init__(
                self,
                event_id: int,
                prng: random.Random,
                session: Session,
                simulator: Simulator,
                name: str,
            ) -> None:
                super().__init__(
                    event_id=event_id,
                    prng=prng,
                    session=session,
                    simulator=simulator,
                    name=name,
                )
                self.n_hooked_before_order = 0
                self.n_hooked_after_order = 0
                self.n_hooked_before_cancel = 0
                self.n_hooked_after_cancel = 0
                self.n_hooked_after_execution = 0
                self.n_hooked_before_session = 0
                self.n_hooked_after_session = 0
                self.n_hooked_before_step_for_market = 0
                self.n_hooked_after_step_for_market = 0

            def hook_registration(self) -> List[EventHook]:
                event_hooks = []
                for hook_type in ["order", "cancel", "execution", "session", "market"]:
                    for is_before in [True, False]:
                        if hook_type == "execution" and is_before:
                            continue
                        for time in [None, [0, 1, 2]]:
                            event_hook = EventHook(
                                event=self,
                                hook_type=hook_type,
                                is_before=is_before,
                                time=time,
                            )
                            event_hooks.append(event_hook)
                return event_hooks

            def hooked_before_order(self, simulator: Simulator, order: Order) -> None:
                self.n_hooked_before_order += 1

            def hooked_after_order(
                self, simulator: Simulator, order_log: OrderLog
            ) -> None:
                self.n_hooked_after_order += 1

            def hooked_before_cancel(
                self, simulator: Simulator, cancel: Cancel
            ) -> None:
                self.n_hooked_before_cancel += 1

            def hooked_after_cancel(
                self, simulator: Simulator, cancel_log: CancelLog
            ) -> None:
                self.n_hooked_after_cancel += 1

            def hooked_after_execution(
                self, simulator: Simulator, execution_log: ExecutionLog
            ) -> None:
                self.n_hooked_after_execution += 1

            def hooked_before_session(
                self, simulator: Simulator, session: Session
            ) -> None:
                self.n_hooked_before_session += 1

            def hooked_after_session(
                self, simulator: Simulator, session: Session
            ) -> None:
                self.n_hooked_after_session += 1

            def hooked_before_step_for_market(
                self, simulator: Simulator, market: Market
            ) -> None:
                self.n_hooked_before_step_for_market += 1

            def hooked_after_step_for_market(
                self, simulator: Simulator, market: Market
            ) -> None:
                self.n_hooked_after_step_for_market += 1

        prng = random.Random(42)
        logger = Logger()
        sim = Simulator(prng=prng, logger=logger)
        session = Session(
            session_id=0,
            prng=random.Random(42),
            session_start_time=0,
            simulator=sim,
            name="session1",
        )
        session.iteration_steps = 1
        sim._add_session(session=session)
        market = Market(
            market_id=0, prng=random.Random(42), simulator=sim, name="market"
        )
        sim._add_market(market=market, group_name="market_group")
        event = DummyEvent(
            event_id=0,
            prng=random.Random(42),
            session=session,
            simulator=sim,
            name="dummy",
        )
        event_hooks = event.hook_registration()
        for event_hook in event_hooks:
            sim._add_event(event_hook=event_hook)
        market._update_time(next_fundamental_price=300)

        order = Order(
            agent_id=1,
            market_id=0,
            is_buy=True,
            kind=LIMIT_ORDER,
            volume=1,
            placed_at=1,
            price=100.0,
            order_id=1,
        )
        sim._trigger_event_before_order(order=order)
        order_log = OrderLog(
            order_id=order.order_id,  # type: ignore
            market_id=order.market_id,
            time=order.placed_at,  # type: ignore
            agent_id=order.agent_id,
            is_buy=order.is_buy,
            kind=order.kind,
            volume=order.volume,
            price=order.price,
            ttl=order.ttl,
        )
        sim._trigger_event_after_order(order_log=order_log)
        cancel = Cancel(order=order, placed_at=1)
        sim._trigger_event_before_cancel(cancel=cancel)
        cancel_log = CancelLog(
            order_id=cancel.order.order_id,  # type: ignore
            market_id=cancel.order.market_id,
            cancel_time=1,
            order_time=cancel.order.placed_at,  # type: ignore
            agent_id=cancel.agent_id,
            is_buy=cancel.order.is_buy,
            kind=cancel.order.kind,
            volume=cancel.order.volume,
            price=cancel.order.price,
            ttl=cancel.order.ttl,
        )
        sim._trigger_event_after_cancel(cancel_log=cancel_log)
        execution_log = ExecutionLog(
            market_id=0,
            time=1,
            buy_agent_id=0,
            sell_agent_id=1,
            buy_order_id=110,
            sell_order_id=111,
            price=90.9,
            volume=2,
        )
        sim._trigger_event_after_execution(execution_log=execution_log)
        sim._trigger_event_before_session(session=session)
        sim._trigger_event_after_session(session=session)
        market._update_time(next_fundamental_price=300.0)
        sim._trigger_event_before_step_for_market(market=market)
        sim._trigger_event_after_step_for_market(market=market)

        assert event.n_hooked_before_order == 2
        assert event.n_hooked_after_order == 2
        assert event.n_hooked_before_cancel == 2
        assert event.n_hooked_after_cancel == 2
        assert event.n_hooked_after_execution == 2
        assert event.n_hooked_before_session == 2
        assert event.n_hooked_after_session == 2
        assert event.n_hooked_before_step_for_market == 2
        assert event.n_hooked_after_step_for_market == 2
