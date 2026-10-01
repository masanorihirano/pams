import math
import random


class BlackScholes:
    """Monte Carlo pricing of a European call option under the Black-Scholes model.

    The price of the underlying asset follows a geometric Brownian motion, which is
    simulated by the Euler method. This class is used as a CPU workload by
    :class:`samples.black_scholes.workload_fcn_agent.WorkloadFCNAgent`.
    """

    def __init__(
        self,
        prng: random.Random,
        initial_price: float,
        strike_price: float,
        risk_free_rate: float,
        volatility: float,
        maturity_time: float,
    ) -> None:
        """Initialize.

        Args:
            prng (random.Random): pseudo random number generator for the Monte Carlo
                simulation.
            initial_price (float): initial price of the underlying asset.
            strike_price (float): strike price of the option.
            risk_free_rate (float): risk-free interest rate.
            volatility (float): volatility of the underlying asset.
            maturity_time (float): time to the maturity of the option.

        Returns:
            None

        """
        self.prng: random.Random = prng
        self.initial_price: float = initial_price
        self.strike_price: float = strike_price
        self.risk_free_rate: float = risk_free_rate
        self.volatility: float = volatility
        self.maturity_time: float = maturity_time

    def compute(self, n_samples: int, n_steps: int) -> float:
        """Estimate the option price by the Monte Carlo simulation.

        Args:
            n_samples (int): number of the sample paths.
            n_steps (int): number of the time steps of each sample path.

        Returns:
            float: estimated option price.

        """
        if n_samples < 1:
            raise ValueError("n_samples have to be positive")
        if n_steps < 1:
            raise ValueError("n_steps have to be positive")
        r: float = self.risk_free_rate
        sigma: float = self.volatility
        dt: float = self.maturity_time / n_steps
        sqrt_dt: float = math.sqrt(dt)
        total: float = 0.0
        for _ in range(n_samples):
            price: float = self.initial_price
            for _ in range(n_steps):
                price += (
                    price * r * dt
                    + price * self.prng.gauss(mu=0.0, sigma=1.0) * sigma * sqrt_dt
                )
            total += max(price - self.strike_price, 0.0)
        return math.exp(-r * self.maturity_time) * (total / n_samples)

    def theoretical_price(self) -> float:
        """Calculate the option price by the Black-Scholes formula.

        Returns:
            float: theoretical option price.

        """
        sigma_sqrt_t: float = self.volatility * math.sqrt(self.maturity_time)
        d1: float = (
            math.log(self.initial_price / self.strike_price)
            + (self.risk_free_rate + self.volatility**2 / 2) * self.maturity_time
        ) / sigma_sqrt_t
        d2: float = d1 - sigma_sqrt_t
        return self.initial_price * _normal_cdf(x=d1) - self.strike_price * math.exp(
            -self.risk_free_rate * self.maturity_time
        ) * _normal_cdf(x=d2)


def _normal_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal distribution (internal).

    Args:
        x (float): value.

    Returns:
        float: probability.

    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
