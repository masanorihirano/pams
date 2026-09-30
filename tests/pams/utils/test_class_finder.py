from typing import Any

import pytest

from pams import Market
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
    find_class(name="OrderMistakeShock")
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
    find_class(name="MarketShareFCNAgent")
    find_class(name="MarketMakerAgent")
    find_class(name="JsonRandom")

    class DummyAgent(FCNAgent):
        pass

    find_class(name="DummyAgent", optional_class_list=[DummyAgent])

    with pytest.raises(AttributeError):
        find_class(name="DummyAgent2", optional_class_list=[DummyAgent])


def test_find_class_with_class() -> None:
    assert find_class(name=FCNAgent) is FCNAgent
    assert find_class(name=Market) is Market

    class DummyAgent(FCNAgent):
        pass

    # a class is returned as is without registration
    assert find_class(name=DummyAgent) is DummyAgent
    assert find_class(name=DummyAgent, optional_class_list=[DummyAgent]) is DummyAgent

    class DummyFCNAgent(FCNAgent):
        pass

    # the name of a given class is not searched, so it does not have to be unique
    DummyFCNAgent.__name__ = "FCNAgent"
    assert find_class(name=DummyFCNAgent) is DummyFCNAgent
    with pytest.raises(AttributeError):
        find_class(name="FCNAgent", optional_class_list=[DummyFCNAgent])


@pytest.mark.parametrize("name", [None, 1, 1.0, ["FCNAgent"], {"class": "FCNAgent"}])
def test_find_class_invalid(name: Any) -> None:
    with pytest.raises(ValueError, match="class must be a class name"):
        find_class(name=name)
