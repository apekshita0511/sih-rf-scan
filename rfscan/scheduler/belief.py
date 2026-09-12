"""Stateful decaying Beta-Bernoulli occupancy belief, per channel.

This is the live, incremental analogue of
:func:`rfscan.perception.features.decaying_beta_posterior`. That function
replays a channel's full scan history from scratch on every call (fine as a
*feature*, computed once per FeatureBuilder.build() call over bounded
history). :class:`BeliefState` instead keeps ``(alpha, beta)`` arrays that are
updated in O(1) per slot per channel, because the Phase 6 closed loop needs an
up-to-date uncertainty term for *every* channel on *every* decision -- the
priority score compares all channels each slot (docs/architecture.md S1.2,
S7) -- and Phase 6's exit criterion is latency < 5 ms/slot.

Update rule (docs/architecture.md S7), implemented as a single uniform
per-slot forgetting step applied to every channel, followed by an additive
evidence update for whichever channel was actually scanned:

    decay()          : for every channel c,
                       alpha[c] <- lambda*alpha[c] + (1-lambda)*prior_alpha
                       beta[c]  <- lambda*beta[c]  + (1-lambda)*prior_beta
    update(c, y)     : alpha[c] += y ; beta[c] += (1 - y)

Calling ``decay()`` once per slot for every channel (not just the unscanned
ones) is a deliberate simplification of S7's literal wording ("every
*unscanned* channel decays"): at the top of the closed loop the scheduler does
not yet know which channel it is about to scan, so there is no way to exempt
it in advance. Decaying uniformly first and then layering the scanned
channel's fresh evidence on top is the standard exponential-forgetting
recursive Bayes filter, produces the same qualitative behaviour (unscanned
channels relax toward the prior and grow less certain; the scanned channel's
posterior sharpens), and is exactly what a fixed-cadence, vectorised, O(1)
per-slot object needs to hit the latency budget.
"""

from __future__ import annotations

import numpy as np


class BeliefState:
    """Per-channel decaying Beta(alpha, beta) occupancy posterior."""

    def __init__(
        self,
        n_channels: int,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay_lambda: float = 0.98,
        seed: int = 0,
    ) -> None:
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if not 0.0 < decay_lambda <= 1.0:
            raise ValueError(f"decay_lambda must be in (0, 1], got {decay_lambda}")
        if prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError("prior_alpha and prior_beta must be > 0")
        self._n = n_channels
        self._prior_alpha = float(prior_alpha)
        self._prior_beta = float(prior_beta)
        self._lambda = float(decay_lambda)
        self._seed = int(seed)
        self._alpha = np.full(n_channels, self._prior_alpha, dtype=np.float64)
        self._beta = np.full(n_channels, self._prior_beta, dtype=np.float64)
        self._rng = np.random.default_rng(self._seed)

    @property
    def n_channels(self) -> int:
        return self._n

    def decay(self) -> None:
        """Relax every channel's posterior one slot toward the prior."""
        lam = self._lambda
        self._alpha = lam * self._alpha + (1.0 - lam) * self._prior_alpha
        self._beta = lam * self._beta + (1.0 - lam) * self._prior_beta

    def update(self, channel_index: int, detected: bool) -> None:
        """Fold in one slot's scan outcome for ``channel_index``."""
        self._check(channel_index)
        if detected:
            self._alpha[channel_index] += 1.0
        else:
            self._beta[channel_index] += 1.0

    def mean(self) -> np.ndarray:
        """Posterior mean per channel, shape ``(n_channels,)``."""
        return self._alpha / (self._alpha + self._beta)

    def std(self) -> np.ndarray:
        """Posterior std per channel, shape ``(n_channels,)`` -- the
        exploration/uncertainty term the priority score consumes."""
        a, b = self._alpha, self._beta
        total = a + b
        variance = (a * b) / ((total**2) * (total + 1.0))
        return np.sqrt(np.maximum(variance, 0.0))

    def sample(self) -> np.ndarray:
        """One Beta(alpha, beta) draw per channel -- for Thompson-sampling
        style strategies (docs/architecture.md S7, Phase 8's ``thompson.py``).
        Not used by AdaptiveScheduler v1's argmax priority rule."""
        return self._rng.beta(self._alpha, self._beta)

    def reset(self) -> None:
        self._alpha = np.full(self._n, self._prior_alpha, dtype=np.float64)
        self._beta = np.full(self._n, self._prior_beta, dtype=np.float64)
        self._rng = np.random.default_rng(self._seed)

    def _check(self, channel_index: int) -> None:
        if not 0 <= channel_index < self._n:
            raise IndexError(f"channel {channel_index} out of range [0, {self._n})")
