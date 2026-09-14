"""Gauss-Kronrod demo: integrate with a built-in error estimate, and watch adaptive bisection swarm a spike."""

import heapq
import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import gauss_kronrod as gk


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def _adaptive_with_intervals(f, a, b, tol, max_intervals=2000):
    """Same as gk.integrate but also returns the final list of (lo, hi) subintervals, for the SVG."""
    k, g, e = gk.gauss_kronrod_15(f, a, b)
    heap = [(-e, a, b, k)]
    total_val, total_err, n = k, e, 1
    while total_err > tol and n < max_intervals:
        neg_e, ia, ib, ival = heapq.heappop(heap)
        mid = 0.5 * (ia + ib)
        k1, g1, e1 = gk.gauss_kronrod_15(f, ia, mid)
        k2, g2, e2 = gk.gauss_kronrod_15(f, mid, ib)
        total_val += (k1 + k2) - ival
        total_err += (e1 + e2) - (-neg_e)
        heapq.heappush(heap, (-e1, ia, mid, k1))
        heapq.heappush(heap, (-e2, mid, ib, k2))
        n += 1
    intervals = sorted((ia, ib) for (_ne, ia, ib, _v) in heap)
    return total_val, total_err, intervals


def main():
    lines = []
    lines.append("Gauss-Kronrod quadrature -- integrate and estimate your own error")
    lines.append("=" * 66)
    lines.append("")
    lines.append("G7-K15: a 7-point Gauss rule embedded in a 15-point Kronrod rule.")
    lines.append("One batch of 15 evaluations gives TWO estimates; |K - G| is the error.")
    lines.append("")

    # single-panel error estimate vs true error on a smooth integral
    lines.append("Single panel on [0, pi], integrand sin(x)  (true = 2):")
    k, g, e = gk.gauss_kronrod_15(math.sin, 0.0, math.pi)
    lines.append(f"  Gauss (7 pt)   = {g:.12f}   true error {abs(g - 2):.2e}")
    lines.append(f"  Kronrod (15pt) = {k:.12f}   true error {abs(k - 2):.2e}")
    lines.append(f"  estimate |K-G| = {abs(k - g):.2e}  (no extra evaluations)")
    lines.append("")

    # polynomial exactness ladder
    lines.append("Polynomial exactness (integral of x^p on [-1,1]):")
    lines.append("    p    Gauss err     Kronrod err")
    lines.append("   " + "-" * 34)
    for p in (12, 13, 14, 16, 20, 22, 24):
        k, g, e = gk.gauss_kronrod_15(lambda x, p=p: x ** p, -1.0, 1.0)
        exact = 2.0 / (p + 1) if p % 2 == 0 else 0.0
        lines.append(f"   {p:3d}    {abs(g - exact):.2e}     {abs(k - exact):.2e}")
    lines.append("  (Gauss exact to deg 13, Kronrod to deg 22)")
    lines.append("")

    # adaptive on a narrow spike
    def spike(x):
        return math.exp(-((x - 0.3) / 0.004) ** 2)
    true_spike = 0.004 * math.sqrt(math.pi)
    v, err, intervals = _adaptive_with_intervals(spike, 0.0, 1.0, tol=1e-12)
    lines.append(f"Adaptive on a spike (width 0.004 at x=0.3), true = {true_spike:.10f}:")
    lines.append(f"  result = {v:.10f}   |error| = {abs(v - true_spike):.2e}")
    lines.append(f"  used {len(intervals)} subintervals, clustered at the peak.")
    lines.append("")

    # endpoint singularity
    v, err, ni = gk.integrate(lambda x: 1.0 / math.sqrt(x) if x > 0 else 0.0,
                              0.0, 1.0, tol=1e-8, max_intervals=4000)
    lines.append(f"Adaptive on 1/sqrt(x) singularity on [0,1] (true = 2):")
    lines.append(f"  result = {v:.8f}   |error| = {abs(v - 2):.2e}   ({ni} subintervals)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(spike, intervals, true_spike)
    return text, svg


def _svg(f, intervals, true_val):
    W, H = 640, 420
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Adaptive Gauss-Kronrod swarming a narrow spike</text>')

    x0, x1, y0, y1 = 55, 610, 60, 330
    a, b = 0.0, 1.0
    ymax = 1.05  # spike peak is 1.0

    def px(x):
        return x0 + (x - a) / (b - a) * (x1 - x0)

    def py(y):
        return y1 - (y / ymax) * (y1 - y0)

    # baseline
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')

    # the integrand curve (dense sampling)
    N = 400
    pts = []
    for i in range(N + 1):
        x = a + (b - a) * i / N
        pts.append(f"{px(x):.1f},{py(f(x)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{P["blue"]}" stroke-width="1.8"/>')

    # subinterval boundaries as vertical ticks -- density shows where points were spent
    for (lo, hi) in intervals:
        for xb in (lo, hi):
            parts.append(f'<line x1="{px(xb):.1f}" y1="{y1}" x2="{px(xb):.1f}" y2="{y1 + 10}" '
                         f'stroke="{P["yellow"]}" stroke-width="1" opacity="0.8"/>')
    # midpoint dots on each subinterval, near the baseline
    for (lo, hi) in intervals:
        m = 0.5 * (lo + hi)
        parts.append(f'<circle cx="{px(m):.1f}" cy="{y1 + 22:.1f}" r="2.2" fill="{P["green"]}"/>')

    parts.append(f'<text x="{x0}" y="{y1 + 48}" fill="{P["gray"]}" font-size="11">'
                 f'yellow ticks = subinterval edges; green = panel centers -- '
                 f'{len(intervals)} panels, packed under the peak</text>')
    parts.append(f'<text x="{px(0.3):.1f}" y="{py(1.0) - 8:.1f}" fill="{P["red"]}" '
                 f'font-size="11" text-anchor="middle">spike at x=0.3</text>')
    parts.append(f'<text x="20" y="{H - 14}" fill="{P["gray"]}" font-size="11">'
                 f'The driver splits the worst-error panel first, so points swarm the spike and '
                 f'ignore the flat tails.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
