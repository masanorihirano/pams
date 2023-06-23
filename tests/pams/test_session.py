import random

import pytest

from pams import Session
from pams import Simulator
from pams.logs import Logger


class TestSession:
    def test__init(self) -> None:
        sim = Simulator(prng=random.Random(42))
        prng = random.Random(42)
        logger = Logger()
        s = Session(
            session_id=1,
            prng=prng,
            session_start_time=10,
            simulator=sim,
            name="session_test",
            logger=logger,
        )
        assert s.session_id == 1
        assert s.name == "session_test"
        assert s.prng == prng
        assert s.simulator == sim
        assert s.logger == logger
        assert s.session_start_time == 10

    def test_setup(self) -> None:
        sim = Simulator(prng=random.Random(42))
        prng = random.Random(42)
        logger = Logger()
        s = Session(
            session_id=1,
            prng=prng,
            session_start_time=10,
            simulator=sim,
            name="session_test",
            logger=logger,
        )
        setting = {
            "sessionName": "session_test2",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        s.setup(settings=setting)
        assert s.name == "session_test"
        assert s.iteration_steps == 10
        assert not s.with_order_placement
        assert not s.with_order_execution
        assert not s.with_print
        assert s.high_frequency_submission_rate == 0.8
        assert s.max_normal_orders == 2
        assert s.max_high_frequency_orders == 3
        assert (
            str(s)
            == "<pams.session.Session | id=1, name=session_test, iteration_steps=10, session_start_time=10, "
            "max_normal_orders=2, max_high_frequency_orders=3, with_order_placement=False, with_order_execution=False, "
            f"high_frequency_submission_rate=0.8, with_print=False, logger={logger}>"
        )

        setting = {
            "sessionName": "session_test",
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10.9,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": 10,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": 10,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": 20,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
            "maxHifreqOrders": 2,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHifreqOrders": 3,
        }
        with pytest.warns(Warning):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "highFrequencySubmitRate": 0.8,
            "hifreqSubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.raises(ValueError):
            s.setup(settings=setting)
        setting = {
            "sessionName": "session_test",
            "iterationSteps": 10,
            "withOrderPlacement": False,
            "withOrderExecution": False,
            "withPrint": False,
            "hifreqSubmitRate": 0.8,
            "maxNormalOrders": 2,
            "maxHighFrequencyOrders": 3,
        }
        with pytest.warns(Warning):
            s.setup(settings=setting)
