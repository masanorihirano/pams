import json
import math
import random
from typing import Dict
from typing import List
from typing import Union

JsonValue = Union[Dict, List, float, int]


class JsonRandom:
    """random generator from json.

    The following direction can be used for config as randomized values:
     - :code:`[a, b]`: uniform distribution started from a and ended to b. Not that the value should be int, this automatically converted into int. This always satisfy :math:`a \leq x < b`
     - :code:`{"const": [a]}`: constant value. Always set to a.
     - :code:`{"uniform": [a, b]}`: same as [a, b]
     - :code:`{"normal": [u, s]}`: normal distribution whose mean and deviation is u and s.
     - :code:`{"expon": [lam]}`: exponential distribution whose mean and deviation is lam.

    Examples:
        >>> from pams.utils.json_random import JsonRandom
        >>> import random
        >>> jr = JsonRandom(prng=random.Random(42))
        >>> [jr.random([10, 20]) for x in range(10)]
        [16.39426798457884, 10.25010755222667, 12.750293183691193, 12.232107381488227, 17.364712141640126, 16.766994874229113, 18.921795677048454, 10.869388326294162, 14.219218196852704, 10.297972194380703]
        >>> [jr.random({"const": [10]}) for x in range(10)]
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        >>> [jr.random({"uniform": [10, 20]}) for x in range(10)]
        [12.186379748036034, 15.053552881033625, 10.265359696838637, 11.988376506866485, 16.49884437779523, 15.449414806032166, 12.204406220406966, 15.892656838759088, 18.094304566778266, 10.064987596780611]
        >>> [jr.random({"normal": [0, 1]}) for x in range(10)]
        [0.5317762204008692, -1.453545298008678, -0.3122773171445598, 0.49036253259352475, 0.8734043853794468, -0.2406296726551354, 0.3765998586879102, 0.24821344932841446, 0.7823268087036421, -1.1132222142481727]
        >>> [jr.random({"expon": [3]}) for x in range(10)]
        [0.642818017709456, 0.9452346835236866, 1.869586994011895, 0.08175668259806873, 2.9143451561160503, 1.7824008841046926, 0.5611413226153803, 1.4412784552296345, 0.4465202669419299, 1.6479086846872075]
    """  # NOQA

    def __init__(self, prng: random.Random) -> None:
        """initialization.

        Args:
            prng (random.Random): pseudo random number generator for this event.

        Returns:
            None
        """
        self.prng: random.Random = prng

    def _next_uniform(self, min_value: float, max_value: float) -> float:
        """get next uniform.

        Its probability density function is
        :math:`p(x) = \frac{1}{max - min}`
        anywhere within the interval :math:`[min, max)`, and 0 elsewhere.

        Args:
            min_value (float): min value.
            max_value (float): max value.

        Returns:
            float: uniform.
        """
        return self.prng.random() * (max_value - min_value) + min_value

    def _next_normal(self, mu: float, sigma: float) -> float:
        """get next normal.

        Its probability density function is
        :math:`p(x) = \frac{1}{\\sqrt{2 \\pi \\sigma^2}e^{- \frac{(x - \\mu)^2}{2 \\sigma^2}}}`
        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.

        Args:
            mu (float): mu.
            sigma (float): sigma.

        Returns:
            float: normal.
        """
        return self.prng.gauss(mu=mu, sigma=sigma)

    def _next_exponential(self, lam: float) -> float:
        """get next exponential.

        Its probability density function is
        :math:`p(x) = \\lambda exp(- \\lambda x)`
        for :math:`x \\gt 0` and 0 elsewhere. Where :math:`\\lambda` is the scale parameter.

        Args:
            lam (float): lambda.

        Returns:
            float: exponential.
        """
        return lam * -math.log(self.prng.random())

    def random(self, json_value: JsonValue) -> float:
        """get a random value.

        Args:
            json_value (JsonValue): random type. This can include the parameter "const", "uniform", "normal", and "expon".

        Returns:
            float: random value.
        """
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
                    "Multiple specification of distribution type: "
                    + json.dumps(json_value)
                )
            if "const" in json_value:
                args = json_value["const"]
                if not isinstance(args, list):
                    raise ValueError(
                        "Constant must be [value] (list) but " + json.dumps(json_value)
                    )
                if len(args) != 1:
                    raise ValueError(
                        "Constant must be [value] but " + json.dumps(json_value)
                    )
                value = float(args[0])
                return value
            if "uniform" in json_value:
                args = json_value["uniform"]
                if not isinstance(args, list):
                    raise ValueError(
                        "Uniform distribution must be [min, max] (list) but "
                        + json.dumps(json_value)
                    )
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
                if not isinstance(args, list):
                    raise ValueError(
                        "Normal distribution must be [mu, sigma] (list) but "
                        + json.dumps(json_value)
                    )
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
                if not isinstance(args, list):
                    raise ValueError(
                        "Exponential distribution must be [lambda] (list) but "
                        + json.dumps(json_value)
                    )
                if len(args) != 1:
                    raise ValueError(
                        "Exponential distribution must be [lambda] but "
                        + json.dumps(json_value)
                    )
                lam = float(args[0])
                return self._next_exponential(lam=lam)
            raise ValueError("Unknown distribution type: " + json.dumps(json_value))
        return float(json_value)
