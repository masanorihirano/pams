import random
from contextlib import redirect_stdout
from io import StringIO

from pams import Market
from pams import Session
from pams import Simulator
from pams.logs import MarketStepEndLog
from pams.logs import MarketStepPrintLogger
from pams.logs import MarketStepSaver


class TestMarketStepPrintLogger:
    def test_process_market_step_end_log(self) -> None:
        logger = MarketStepPrintLogger()
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        market = Market(
            market_id=2, prng=random.Random(22), simulator=sim, name="test_market"
        )
        settings_market = {
            "tickSize": 0.01,
            "marketPrice": 300.0,
            "outstandingShares": 2000,
        }
        market.setup(settings=settings_market)
        market._update_time(next_fundamental_price=400.0)
        log = MarketStepEndLog(session=session, market=market, simulator=sim)
        io = StringIO()
        with redirect_stdout(io):
            logger.write_and_direct_process(log=log)
        assert io.getvalue() == "0 0 2 test_market 300.0 400.0\n"


class TestMarketStepSaver:
    def test__init__(self) -> None:
        logger = MarketStepSaver()
        assert logger.market_step_logs == []

    def test_process_market_step_end_log(self) -> None:
        logger = MarketStepSaver()
        sim = Simulator(prng=random.Random(42))
        session = Session(
            session_id=0,
            prng=random.Random(31),
            session_start_time=0,
            simulator=sim,
            name="test_session",
        )
        market = Market(
            market_id=2, prng=random.Random(22), simulator=sim, name="test_market"
        )
        settings_market = {
            "tickSize": 0.01,
            "marketPrice": 300.0,
            "outstandingShares": 2000,
        }
        market.setup(settings=settings_market)
        market._update_time(next_fundamental_price=400.0)
        log = MarketStepEndLog(session=session, market=market, simulator=sim)
        logger.write_and_direct_process(log=log)
        assert logger.market_step_logs == [
            {
                "session_id": 0,
                "market_time": 0,
                "market_id": 2,
                "market_name": "test_market",
                "market_price": 300.0,
                "fundamental_price": 400.0,
            }
        ]
