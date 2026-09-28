import json
import math
import random
from typing import Dict
from typing import List
from typing import Union

JsonValue = Union[Dict, List, float, int]


class JsonRandom:
    r"""random generator from json.

    The following direction can be used for config as randomized values:
     - :code:`[a, b]`: uniform distribution on :math:`[a, b)`. Both bounds are converted into float and
       a float is returned, so :math:`a \leq x < b` always holds; convert it into int by yourself if needed.
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
        """Initialization.

        Args:
            prng (random.Random): pseudo random number generator for this event.

        Returns:
            None

        """
        self.prng: random.Random = prng

    def _next_uniform(self, min_value: float, max_value: float) -> float:
        r"""Get next uniform.

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
        r"""Get next normal.

        Its probability density function is
        :math:`p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}`
        where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation.

        Args:
            mu (float): mu.
            sigma (float): sigma.

        Returns:
            float: normal.

        """
        return self.prng.gauss(mu=mu, sigma=sigma)

    def _next_exponential(self, lam: float) -> float:
        r"""Get next exponential.

        Its probability density function is
        :math:`p(x) = \frac{1}{\lambda} e^{- x / \lambda}`
        for :math:`x \geq 0` and 0 elsewhere, where :math:`\lambda` is the scale parameter,
        i.e., both the mean and the standard deviation of the distribution.

        Args:
            lam (float): lambda (mean of the distribution).

        Returns:
            float: exponential.

        """
        return lam * -math.log(self.prng.random())

    def _get_distribution_args(
        self, json_value: Dict, key: str, n_args: int, expected: str
    ) -> List:
        """Get the validated arguments of a distribution (internal method).

        Args:
            json_value (Dict): distribution, e.g. :code:`{"normal": [u, s]}`.
            key (str): distribution type in ``json_value``.
            n_args (int): required number of arguments.
            expected (str): expected form used in error messages,
                e.g. "Constant must be [value]".

        Returns:
            List: arguments of the distribution.

        Raises:
            ValueError: if the arguments are not a list of ``n_args`` elements.

        """
        args = json_value[key]
        if not isinstance(args, list):
            raise ValueError(expected + " (list) but " + json.dumps(json_value))
        if len(args) != n_args:
            raise ValueError(expected + " but " + json.dumps(json_value))
        return args

    def random(self, json_value: JsonValue) -> float:
        """Get a random value.

        Args:
            json_value (JsonValue): random type.
                                    This can include the parameter "const", "uniform", "normal", and "expon".

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
                args = self._get_distribution_args(
                    json_value, "const", 1, "Constant must be [value]"
                )
                value = float(args[0])
                return value
            if "uniform" in json_value:
                args = self._get_distribution_args(
                    json_value, "uniform", 2, "Uniform distribution must be [min, max]"
                )
                min_value = float(args[0])
                max_value = float(args[1])
                return self._next_uniform(min_value=min_value, max_value=max_value)
            if "normal" in json_value:
                args = self._get_distribution_args(
                    json_value, "normal", 2, "Normal distribution must be [mu, sigma]"
                )
                mu = float(args[0])
                sigma = float(args[1])
                return self._next_normal(mu=mu, sigma=sigma)
            if "expon" in json_value:
                args = self._get_distribution_args(
                    json_value, "expon", 1, "Exponential distribution must be [lambda]"
                )
                lam = float(args[0])
                return self._next_exponential(lam=lam)
            raise ValueError("Unknown distribution type: " + json.dumps(json_value))
        return float(json_value)
