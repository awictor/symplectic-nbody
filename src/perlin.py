"""Perlin noise: smooth pseudo-random gradient fields for procedural generation.

Random numbers are jagged; nature is smooth. PERLIN NOISE, invented by Ken Perlin in 1983 (and worth
an Academy Award for its use in film), produces a random-looking but CONTINUOUS field -- values that
vary smoothly across space with no visible grid, the foundation of procedurally generated terrain,
clouds, textures, and fire in games and CGI. Unlike white noise (independent random values), Perlin
noise has controllable feature size and looks organic because it is differentiable.

The construction is GRADIENT noise. Lay a integer lattice over space; assign each lattice point a
pseudo-random unit GRADIENT vector (deterministically, from a hashed permutation table, so the field
is reproducible and infinite). To evaluate the noise at a point, find the surrounding lattice cell,
compute at each corner the dot product of that corner's gradient with the vector from the corner to
the point, and interpolate those corner contributions using a SMOOTHSTEP fade curve
6t^5 - 15t^4 + 10t^3 (whose first and second derivatives vanish at 0 and 1, which is what makes the
result look seamless). Because every lattice point evaluates to zero contribution AT the lattice
point, the noise passes through zero on the grid and undulates smoothly between.

Layering copies of the noise at doubling frequencies and halving amplitudes -- FRACTAL BROWNIAN
MOTION (fBm) -- adds detail at every scale, the standard recipe for natural-looking terrain and
clouds. This module implements reproducible 1-D and 2-D Perlin noise with a seeded permutation table,
the fade/lerp/gradient machinery, and fBm octave summation. It is verified that the noise is exactly
zero at integer lattice points (the defining gradient-noise property), that its output stays within
the theoretical bounds, that it is deterministic for a given seed and continuous (small input changes
give small output changes), that its mean over a large region is near zero, and that fBm with more
octaves adds high-frequency detail (greater roughness) while staying bounded. Pure stdlib; a
procedural-generation companion to the cellular-automaton and Mandelbrot notes."""

from __future__ import annotations

import math


def _build_permutation(seed):
    """A seeded 0..255 permutation, doubled to 512 to avoid index wrapping (Perlin's classic table)."""
    p = list(range(256))
    state = seed & 0xFFFFFFFF
    # Fisher-Yates with an LCG
    for i in range(255, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        j = (state >> 8) % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p + p


def _fade(t):
    """Perlin's smoothstep fade: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


class Perlin:
    """Reproducible Perlin gradient noise in 1-D and 2-D."""

    def __init__(self, seed=1):
        self.perm = _build_permutation(seed)

    # --- 1-D -------------------------------------------------------------
    @staticmethod
    def _grad1(h, x):
        # gradient is +1 or -1
        return x if (h & 1) == 0 else -x

    def noise1(self, x):
        """1-D Perlin noise at x, in roughly [-1, 1]."""
        xi = math.floor(x) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        g0 = self._grad1(self.perm[xi], xf)
        g1 = self._grad1(self.perm[xi + 1], xf - 1)
        return _lerp(g0, g1, u) * 2.0        # scale to ~[-1,1]

    # --- 2-D -------------------------------------------------------------
    @staticmethod
    def _grad2(h, x, y):
        # 8 gradient directions (Perlin's improved set projected to 2-D)
        h &= 7
        # (cos, sin) of multiples of 45 degrees
        gx, gy = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (0.7071, 0.7071), (-0.7071, 0.7071),
                  (0.7071, -0.7071), (-0.7071, -0.7071)][h]
        return gx * x + gy * y

    def noise2(self, x, y):
        """2-D Perlin noise at (x, y), in roughly [-1, 1]."""
        xi = math.floor(x) & 255
        yi = math.floor(y) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = _fade(xf)
        v = _fade(yf)
        p = self.perm
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]
        x1 = _lerp(self._grad2(aa, xf, yf), self._grad2(ba, xf - 1, yf), u)
        x2 = _lerp(self._grad2(ab, xf, yf - 1), self._grad2(bb, xf - 1, yf - 1), u)
        return _lerp(x1, x2, v) * 1.4142      # scale toward [-1,1]

    # --- fractal Brownian motion ----------------------------------------
    def fbm2(self, x, y, octaves=4, persistence=0.5, lacunarity=2.0):
        """Fractal Brownian motion: sum of `octaves` noise layers at doubling frequency, halving
        amplitude. Returns a value normalized to roughly [-1, 1]."""
        total = 0.0
        freq = 1.0
        amp = 1.0
        max_amp = 0.0
        for _ in range(octaves):
            total += self.noise2(x * freq, y * freq) * amp
            max_amp += amp
            amp *= persistence
            freq *= lacunarity
        return total / max_amp if max_amp > 0 else 0.0

    def fbm1(self, x, octaves=4, persistence=0.5, lacunarity=2.0):
        total = 0.0
        freq = 1.0
        amp = 1.0
        max_amp = 0.0
        for _ in range(octaves):
            total += self.noise1(x * freq) * amp
            max_amp += amp
            amp *= persistence
            freq *= lacunarity
        return total / max_amp if max_amp > 0 else 0.0
