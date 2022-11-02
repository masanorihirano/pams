import json
import math
import random
from typing import Dict
from typing import List
from typing import Union

JsonValue = Union[Dict, List, float, int]


class JsonRandom:
    def __init__(self, prng: random.Random) -> None:
        self.prng: random.Random = prng

    def _next_uniform(self, min_value: float, max_value: float) -> float:
        return self.prng.random() * (max_value - min_value) + min_value

    def _next_normal(self, mu: float, sigma: float) -> float:
        return self.prng.gauss(mu=mu, sigma=sigma)

    def _next_exponential(self, lam: float) -> float:
        return lam * -math.log(self.prng.random())

    def random(self, json_value: JsonValue) -> float:
        if isinstance(json_value, list):
            if len(json_value) != 2:
                raise ValueError(
                    "Uniform distribution must be [min, max] but "
                    + json.dumps(json_value)
                )
            min_value: float = float(json_value[0])
            max_value: float = float(json_value[1])
            return self._next_uniform(min_value=min_value, max_value=max_value)
        if isinstance(json_value, dict):
            if len(json_value) != 1:
                raise ValueError(
                    "Multiple speficiation of distribution type: "
                    + json.dumps(json_value)
                )
            if "const" in json_value:
                args = json_value["const"]
                if len(args) != 1:
                    raise ValueError(
                        "Constant must be [value] but " + json.dumps(json_value)
                    )
                value = float(args[0])
                return value
            if "uniform" in json_value:
                args = json_value["uniform"]
                if len(args) != 2:
                    raise ValueError(
                        "Uniform distribution must be [min, max] but "
                        + json.dumps(json_value)
                    )
                min_value = float(args[0])
                max_value = float(args[1])
                return self._next_uniform(min_value=min_value, max_value=max_value)
            if "normal" in json_value:
                args = json_value["normal"]
                if len(args) != 2:
                    raise ValueError(
                        "Normal distribution must be [mu, sigma] but "
                        + json.dumps(json_value)
                    )
                mu = float(args[0])
                sigma = float(args[1])
                return self._next_normal(mu=mu, sigma=sigma)
            if "expon" in json_value:
                args = json_value["expon"]
                if len(args) != 1:
                    raise ValueError(
                        "Exponential distribution must be [lambda] but "
                        + json.dumps(json_value)
                    )
                lam = float(args[0])
                return self._next_exponential(lam=lam)
            raise ValueError("Unknown distribution type: " + json.dumps(json_value))
        return float(json_value)
