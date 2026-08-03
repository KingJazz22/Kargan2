"""Beta-Binomial posterior over a win probability, pooled across a symbol
basket (one shared instance mutated across every symbol's backtest run).

Prior defaults to uninformative Beta(1,1) (uniform, 50% mean) -- deliberate,
not a placeholder: what this tracks ("does a grid resolve via take-profit")
is a different statistic from any individual entry signal's own win rate, so
there's no directly transferable informed prior to seed from.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class BetaBinomialPosterior:
    alpha: float = 1.0
    beta: float = 1.0
    n_updates: int = 0

    def __post_init__(self):
        self._alpha0 = self.alpha
        self._beta0 = self.beta

    def update(self, win: bool) -> None:
        if win:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.n_updates += 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_lower(self, pct: float = 0.20, n_samples: int = 20_000, seed: int = 0) -> float:
        """pct-quantile of Beta(alpha, beta), via Monte Carlo (dependency-free)."""
        rng = np.random.default_rng(seed)
        samples = rng.beta(self.alpha, self.beta, n_samples)
        return float(np.percentile(samples, pct * 100))

    def credible_interval(self, width: float = 0.80) -> tuple[float, float]:
        lo = (1 - width) / 2
        return self.credible_lower(lo), self.credible_lower(1 - lo)

    def reset(self) -> None:
        self.alpha, self.beta, self.n_updates = self._alpha0, self._beta0, 0
