import random
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
from scipy.linalg import cholesky


class Fundamentals:
    def __init__(self, prng: random.Random) -> None:
        self._prng = prng
        self._np_prng: np.random.Generator = np.random.default_rng(
            self._prng.randint(0, 2**31)
        )
        self.correlation: Dict[Tuple[int, int], float] = {}
        self.drifts: Dict[int, float] = {}
        self.volatilities: Dict[int, float] = {}
        self.prices: Dict[int, List[float]] = {}
        self.market_ids: List[int] = []
        self.initials: Dict[int, float] = {}
        self.start_at: Dict[int, int] = {}
        self._generated_until: int = 0
        self._generate_chunk_size = 100

    def add_market(
        self,
        market_id: int,
        initial: float,
        drift: float,
        volatility: float,
        start_at: int = 0,
    ) -> None:
        if market_id in self.market_ids:
            raise ValueError(f"market {market_id} is already registered")
        self.market_ids.append(market_id)
        self.drifts[market_id] = drift
        self.volatilities[market_id] = volatility
        self.initials[market_id] = initial
        self.start_at[market_id] = start_at
        self.prices[market_id] = [initial for _ in range(start_at + 1)]
        self._generated_until = min(start_at, self._generated_until)

    def remove_market(self, market_id: int) -> None:
        self.market_ids.remove(market_id)

    def change_volatility(self, market_id: int, volatility: float) -> None:
        self.volatilities[market_id] = volatility

    def change_drift(self, market_id: int, drift: float) -> None:
        self.drifts[market_id] = drift

    def set_correlation(self, market_id1: int, market_id2: int, corr: float) -> None:
        if not (-1.0 <= corr <= 1.0):
            raise ValueError("corr must be between 0.0 and 1.0")
        self.correlation[(market_id1, market_id2)] = corr

    def remove_correlation(self, market_id1: int, market_id2: int) -> None:
        self.correlation.pop((market_id1, market_id2))

    def _generate_log_return(
        self, generate_target_ids: List[int], length: int
    ) -> np.ndarray:
        generate_target_ids_cholesky = list(
            filter(lambda x: self.volatilities[x] != 0.0, generate_target_ids)
        )
        generate_target_ids_other = list(
            filter(lambda x: self.volatilities[x] == 0.0, generate_target_ids)
        )
        corr_matrix = np.eye(len(generate_target_ids_cholesky))
        for (id1, id2), corr in self.correlation.items():
            if id1 not in generate_target_ids_cholesky:
                continue
            if id2 not in generate_target_ids_cholesky:
                continue
            if id1 == id2:
                raise AssertionError
            corr_matrix[id1, id2] = corr
            corr_matrix[id2, id1] = corr
        vol = np.asarray([self.volatilities[x] for x in generate_target_ids_cholesky])
        cov_matrix = vol * corr_matrix * vol
        cholesky_matrix = cholesky(cov_matrix, lower=True)

        dw_cholesky = self._np_prng.standard_normal(
            size=(len(generate_target_ids_cholesky), length)
        )
        drifts_cholesky = np.asarray(
            [self.drifts[x] for x in generate_target_ids_cholesky]
        )
        result_cholesky = np.dot(
            cholesky_matrix, dw_cholesky
        ) + drifts_cholesky.T.reshape(-1, 1)

        drifts_others = np.asarray(
            [[self.drifts[x] for _ in range(length)] for x in generate_target_ids_other]
        )

        return np.stack(
            [
                result_cholesky[generate_target_ids_cholesky.index(x)]
                if x in generate_target_ids_cholesky
                else drifts_others[generate_target_ids_other.index(x)]
                for x in generate_target_ids
            ]
        )

    def _generate_next(self) -> None:
        setting_change_points: List[int] = [
            x for x in self.start_at.values() if x > self._generated_until
        ]
        if len(setting_change_points) == 0:
            length = self._generate_chunk_size
        else:
            length = min(setting_change_points) - self._generated_until
        next_until = self._generated_until + length
        target_market_ids: List[int] = [
            key for key, value in self.start_at.items() if value < next_until
        ]
        log_return = self._generate_log_return(
            generate_target_ids=target_market_ids, length=length
        )
        current_prices = np.asarray(
            [self.prices[x][self._generated_until] for x in target_market_ids]
        )
        prices = current_prices.T.reshape(-1, 1) * np.exp(
            np.cumsum(log_return, axis=-1)
        )
        for market_id, price_seq in zip(target_market_ids, prices):
            self.prices[market_id] = (
                self.prices[market_id][: self._generated_until + 1] + price_seq.tolist()
            )
        self._generated_until += length

    def get_fundamental_price(self, market_id: int, time: int) -> float:
        while time >= self._generated_until:
            self._generate_next()
        return self.prices[market_id][time]

    def get_fundamental_prices(
        self, market_id: int, times: Iterable[int]
    ) -> List[float]:
        while max([x for x in times]) >= self._generated_until:
            self._generate_next()
        return [self.prices[market_id][x] for x in times]
