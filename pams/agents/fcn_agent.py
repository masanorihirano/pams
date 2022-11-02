import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .. import LIMIT_ORDER
from .. import Logger
from .. import Market
from .. import Order
from ..agent import Agent
from ..simulator import Simulator
from ..utils.json_random import JsonRandom

MARGIN_FIXED = 0
MARGIN_NORMAL = 1


class FCNAgent(Agent):
    fundamental_weight: float
    chart_weight: float
    is_chart_following: bool = True
    margin_type: int
    mean_reversion_time: int
    noise_scale: float
    noise_weight: float
    order_margin: float
    time_window_size: int

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger] = None,
    ):
        super().__init__(agent_id, prng, simulator, name, logger)

    def is_finite(self, x: float) -> bool:
        return not math.isnan(x) and not math.isinf(x)

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.fundamental_weight = json_random.random(
            json_value=settings["fundamentalWeight"]
        )
        self.chart_weight = json_random.random(json_value=settings["chartWeight"])
        self.noise_weight = json_random.random(json_value=settings["noiseWeight"])
        self.noise_scale = json_random.random(json_value=settings["noiseScale"])
        self.time_window_size = int(
            json_random.random(json_value=settings["timeWindowSize"])
        )
        self.order_margin = json_random.random(json_value=settings["orderMargin"])
        if settings.get("marginType") in [None, "fixed"]:
            self.margin_type = MARGIN_FIXED
        else:
            self.margin_type = MARGIN_NORMAL
        if "meanReversionTime" in settings:
            self.mean_reversion_time = int(
                json_random.random(json_value=settings["meanReversionTime"])
            )
        else:
            self.mean_reversion_time = self.time_window_size

    def submit_orders(self, markets: List[Market]) -> List[Order]:
        orders: List[Order] = sum(
            [self.submit_orders_by_market(market=market) for market in markets], []
        )
        return orders

    def submit_orders_by_market(self, market: Market) -> List[Order]:
        orders: List[Order] = []
        if not self.is_market_accessible(market_id=market.market_id):
            return orders

        time: int = market.get_time()
        time_window_size: int = min(time, self.time_window_size)
        assert time_window_size >= 0
        assert self.fundamental_weight >= 0.0
        assert self.chart_weight >= 0.0
        assert self.noise_weight >= 0.0

        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return = fundamental_scale * math.log(
            market.get_fundamental_price() / market.get_market_price()
        )
        assert self.is_finite(fundamental_log_return)

        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_mean_log_return = chart_scale * math.log(
            market.get_market_price() / market.get_market_price(time - time_window_size)
        )
        assert self.is_finite(chart_mean_log_return)

        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        assert self.is_finite(noise_log_return)

        expected_log_return: float = (
            1.0 / (self.fundamental_weight + self.chart_weight + self.noise_weight)
        ) * (
            self.fundamental_weight * fundamental_log_return
            + self.chart_weight
            * chart_mean_log_return
            * (1 if self.is_chart_following else -1)
            + self.noise_weight * noise_log_return
        )
        assert self.is_finite(expected_log_return)

        expected_future_price: float = market.get_market_price() * math.exp(
            expected_log_return * self.time_window_size
        )
        assert self.is_finite(expected_future_price)

        if self.margin_type == MARGIN_FIXED:
            assert 0.0 <= self.order_margin <= 1.0

            order_volume: int = 1

            if expected_future_price > market.get_market_price():
                order_price = expected_future_price * (1 - self.order_margin)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
            if expected_future_price < market.get_market_price():
                order_price = expected_future_price * (1 + self.order_margin)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )

        if self.margin_type == MARGIN_NORMAL:
            assert self.order_margin >= 0.0
            order_price = (
                expected_future_price
                + self.prng.gauss(mu=0.0, sigma=1.0) * self.order_margin
            )
            order_volume = 1
            assert order_price >= 0.0
            assert order_volume > 0
            if expected_future_price > market.get_market_price():
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
            if expected_future_price < market.get_market_price():
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
        return orders

    def __str__(self) -> str:
        return (
            f"Agent:{self.agent_id}[rnd:{self.prng},cw:{self.chart_weight},fw:{self.fundamental_weight},"
            + f"following:{self.is_chart_following},mtypr:{self.margin_type},mrt:{self.mean_reversion_time},"
            + f"ns:{self.noise_scale},nw:{self.noise_weight},om:{self.order_margin},tws:{self.time_window_size},"
            + f"cash:{self.cash_amount}]"
        )
