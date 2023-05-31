import pytest

from pams.agents import FCNAgent
from pams.utils import find_class


def test_find_class() -> None:
    find_class(name="Runner")
    find_class(name="SequentialRunner")
    find_class(name="Simulator")
    find_class(name="Session")
    find_class(name="Market")
    find_class(name="IndexMarket")
    find_class(name="Fundamentals")
    find_class(name="Order")
    find_class(name="Cancel")
    find_class(name="OrderKind")
    find_class(name="MARKET_ORDER")
    find_class(name="LIMIT_ORDER")
    find_class(name="EventABC")
    find_class(name="EventHook")
    find_class(name="FundamentalPriceShock")
    find_class(name="PriceLimitRule")
    find_class(name="TradingHaltRule")
    find_class(name="Log")
    find_class(name="OrderLog")
    find_class(name="CancelLog")
    find_class(name="ExecutionLog")
    find_class(name="SimulationBeginLog")
    find_class(name="SimulationEndLog")
    find_class(name="SessionBeginLog")
    find_class(name="SessionEndLog")
    find_class(name="MarketStepBeginLog")
    find_class(name="MarketStepEndLog")
    find_class(name="Logger")
    find_class(name="MarketStepPrintLogger")
    find_class(name="MarketStepSaver")
    find_class(name="Agent")
    find_class(name="HighFrequencyAgent")
    find_class(name="FCNAgent")
    find_class(name="ArbitrageAgent")
    find_class(name="JsonRandom")

    class DummyAgent(FCNAgent):
        pass

    find_class(name="DummyAgent", optional_class_list=[DummyAgent])

    with pytest.raises(AttributeError):
        find_class(name="DummyAgent2", optional_class_list=[DummyAgent])
