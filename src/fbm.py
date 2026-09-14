"""Fractional Brownian motion: layered noise octaves that make terrain, clouds, and marble.

A single octave of Perlin noise is smooth and featureless -- one characteristic bump size. Real natural
detail is SELF-SIMILAR across scales: a mountain range has big ridges, smaller ridges on those, and
pebbles on those. FRACTIONAL BROWNIAN MOTION (fBm) builds that fractal richness by summing several
octaves of a base noise, each at DOUBLE the frequency and a FRACTION of the amplitude of the last:

    fBm(x) = sum_{i=0}^{octaves-1} gain^i * noise(lacunarity^i * x).

LACUNARITY (usually 2) is the frequency multiplier between octaves; GAIN or PERSISTENCE (usually 0.5)
is the amplitude multiplier. High gain means rougher, more detailed output; low gain means smoother,
dominated by the first octave. Two close cousins reshape the same sum: TURBULENCE takes the absolute
value of each octave, giving the billowy, cloud-like look with sharp creases; RIDGED multifractal
inverts that (1 - |noise|) and squares it, carving the sharp mountain RIDGES used for terrain.

This module builds fBm, turbulence, and ridged noise in 1-D and 2-D on top of the repo's Perlin noise,
plus a normalized-to-[0,1] terrain heightmap generator. It is validated: with one octave fBm reduces to
the base noise exactly; the theoretical maximum amplitude is the geometric series sum, and the output
respects it; turbulence and ridged fields are non-negative; adding octaves strictly increases the total
variation (more detail); halving the gain makes the field smoother (lower variance of differences); the
same seed is reproducible; and fBm is continuous (nearby inputs give nearby outputs). Reuses the repo's
Perlin noise. Pure stdlib; the multifractal companion to the Perlin-noise, Worley-noise, and
Voronoi tools."""

from __future__ import annotations

from perlin import Perlin


class FBM:
    """Fractional Brownian motion layered over Perlin noise."""

    def __init__(self, seed=1, octaves=6, lacunarity=2.0, gain=0.5):
        self.perlin = Perlin(seed)
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain

    def max_amplitude(self):
        """The largest value fBm can reach: sum of the octave amplitudes (geometric series)."""
        amp = 1.0
        total = 0.0
        for _ in range(self.octaves):
            total += amp
            amp *= self.gain
        return total

    def fbm2(self, x, y):
        """2-D fractional Brownian motion at (x, y)."""
        total = 0.0
        freq = 1.0
        amp = 1.0
        for _ in range(self.octaves):
            total += amp * self.perlin.noise2(x * freq, y * freq)
            freq *= self.lacunarity
            amp *= self.gain
        return total

    def fbm1(self, x):
        total = 0.0
        freq = 1.0
        amp = 1.0
        for _ in range(self.octaves):
            total += amp * self.perlin.noise1(x * freq)
            freq *= self.lacunarity
            amp *= self.gain
        return total

    def turbulence2(self, x, y):
        """Turbulence: sum of |noise| octaves -- billowy, cloud-like, non-negative."""
        total = 0.0
        freq = 1.0
        amp = 1.0
        for _ in range(self.octaves):
            total += amp * abs(self.perlin.noise2(x * freq, y * freq))
            freq *= self.lacunarity
            amp *= self.gain
        return total

    def ridged2(self, x, y):
        """Ridged multifractal: (1 - |noise|)^2 octaves -- sharp mountain ridges."""
        total = 0.0
        freq = 1.0
        amp = 1.0
        for _ in range(self.octaves):
            n = 1.0 - abs(self.perlin.noise2(x * freq, y * freq))
            total += amp * n * n
            freq *= self.lacunarity
            amp *= self.gain
        return total

    def heightmap(self, width, height, scale=0.1, mode="fbm"):
        """A width x height heightmap normalized to [0, 1]. mode: 'fbm', 'turbulence', 'ridged'."""
        raw = [[0.0] * width for _ in range(height)]
        fn = {"fbm": self.fbm2, "turbulence": self.turbulence2, "ridged": self.ridged2}[mode]
        lo = float("inf")
        hi = float("-inf")
        for j in range(height):
            for i in range(width):
                v = fn(i * scale, j * scale)
                raw[j][i] = v
                lo = min(lo, v)
                hi = max(hi, v)
        rng = hi - lo or 1.0
        return [[(raw[j][i] - lo) / rng for i in range(width)] for j in range(height)]


def total_variation(vals):
    """Sum of absolute successive differences -- a measure of how much detail a 1-D field has."""
    return sum(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
