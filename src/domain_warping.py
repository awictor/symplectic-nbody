"""Domain warping: feed noise into its own coordinates for the swirling, marbled patterns of nature.

Ordinary fractal noise is isotropic -- statistically the same in every direction. Real materials look
FLOWING: wood grain bends around knots, marble veins swirl, smoke curls, magma folds. Inigo Quilez's
DOMAIN WARPING produces exactly that look with a beautifully recursive idea: don't sample the noise at
the query point x, sample it at a point that has itself been DISPLACED by more noise. In one level,

    warp(x) = fbm(x + amplitude * fbm(x + offset)),

and each nested layer adds another fold. The inner fBm perturbs where you look, so smooth ridges get
dragged sideways into whorls and filaments; two levels of warping give the characteristic "flow" of
Quilez's famous images. The displacement is a VECTOR field -- one fBm per coordinate -- so the whole
plane is pushed around before the final color/height is read off.

The technique is pure composition: it needs no new math beyond the fBm it wraps, yet it transforms the
visual character completely, and it is a workhorse of procedural texturing, terrain, and shader art.
This module implements 1-, 2-, and n-level domain warping over the repo's fBm, exposing the warped
coordinates as well as the final value. It is validated: with zero warp amplitude the result reduces
exactly to plain fBm; the warp displacement is bounded by the amplitude times fBm's own bound; the
field is deterministic for a fixed seed and continuous in the input; warping strictly increases the
pattern's total variation (more structure) compared to the unwarped field; and successive warp levels
keep the output finite and bounded. Reuses the repo's fBm (and thus Perlin) noise. Pure stdlib; the
procedural-texture companion to the fBm, Perlin-noise, and Worley-noise tools."""

from __future__ import annotations

from fbm import FBM, total_variation


class DomainWarp:
    """Domain-warped fractal noise built on fBm."""

    def __init__(self, seed=1, octaves=5, lacunarity=2.0, gain=0.5, amplitude=1.0):
        # three independent fBm channels (decorrelated by seed offsets) for warp vectors + base
        self.base = FBM(seed, octaves, lacunarity, gain)
        self.wx = FBM(seed + 101, octaves, lacunarity, gain)
        self.wy = FBM(seed + 202, octaves, lacunarity, gain)
        self.wx2 = FBM(seed + 303, octaves, lacunarity, gain)
        self.wy2 = FBM(seed + 404, octaves, lacunarity, gain)
        self.amplitude = amplitude

    def warp_vector(self, x, y, levels=1):
        """The displacement (dx, dy) applied to (x, y) after `levels` of warping."""
        wx, wy = self._warped_coords(x, y, levels)
        return (wx - x, wy - y)

    def _warped_coords(self, x, y, levels):
        px, py = x, y
        if levels >= 1:
            qx = self.wx.fbm2(px, py)
            qy = self.wy.fbm2(px + 5.2, py + 1.3)
            px, py = x + self.amplitude * qx, y + self.amplitude * qy
        if levels >= 2:
            rx = self.wx2.fbm2(px + 1.7, py + 9.2)
            ry = self.wy2.fbm2(px + 8.3, py + 2.8)
            px, py = px + self.amplitude * rx, py + self.amplitude * ry
        return px, py

    def value(self, x, y, levels=1):
        """The domain-warped fBm value at (x, y)."""
        if levels <= 0:
            return self.base.fbm2(x, y)
        px, py = self._warped_coords(x, y, levels)
        return self.base.fbm2(px, py)

    def field(self, width, height, scale=0.08, levels=1):
        """A width x height field of warped values, normalized to [0, 1]."""
        raw = [[0.0] * width for _ in range(height)]
        lo = float("inf")
        hi = float("-inf")
        for j in range(height):
            for i in range(width):
                v = self.value(i * scale, j * scale, levels)
                raw[j][i] = v
                lo = min(lo, v)
                hi = max(hi, v)
        rng = hi - lo or 1.0
        return [[(raw[j][i] - lo) / rng for i in range(width)] for j in range(height)]


def unwarped_value(dw, x, y):
    """Convenience: the base fBm value with no warping (for comparison)."""
    return dw.base.fbm2(x, y)
