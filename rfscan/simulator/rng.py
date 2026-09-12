"""Deterministic, position-independent randomness for the synthetic RF world.

Fairness invariant (docs/architecture.md S2): the world's stochastic realisation
is keyed by ``(world_seed, slot, channel_index)`` -- never by RNG-stream
position. Whether the agent scans CH3 or CH6 at slot 5 cannot change what CH9
does at slot 6, because every channel-slot cell draws from its own independently
keyed generator.

Implementation: a counter-based **Philox** generator, ``key = world_seed`` and
``counter`` packed from ``(slot, channel_index)``. Counter-based generators are
built for exactly this access pattern -- construction is far cheaper than hashing
a ``SeedSequence`` per cell, and the mapping is a pure bijection, stable across
processes and platforms.
"""

from __future__ import annotations

import numpy as np

_MASK64 = (1 << 64) - 1
_CHANNEL_BITS = 24  # up to 16.7M channels per slot; slot index uses the rest


def draw_rng(world_seed: int, slot: int, channel_index: int) -> np.random.Generator:
    """Return the generator that owns every random draw for one (slot, channel)
    cell of the world identified by ``world_seed``.

    Callers must consume values from the returned generator in a fixed order so
    the mapping stays deterministic; see :mod:`rfscan.simulator.environment`.
    """
    counter = ((int(slot) & _MASK64) << _CHANNEL_BITS) | (
        int(channel_index) & ((1 << _CHANNEL_BITS) - 1)
    )
    bit_gen = np.random.Philox(key=int(world_seed) & _MASK64, counter=counter)
    return np.random.Generator(bit_gen)
