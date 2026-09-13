"""Demo: exponential smoothing -- forecasting a seasonal, trending series with Holt-Winters.

Builds a monthly series with a rising trend and a seasonal cycle, fits SES, Holt, and Holt-Winters,
and forecasts a year ahead, showing only the seasonal method captures the pattern. Draws the series,
the fitted level, and the forecast.

    python examples/holt_winters_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from holt_winters import ses, holt, holt_winters, one_step_errors, rmse  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Exponential smoothing: level, trend, and season from a decaying weighted average\n")

    m = 12
    rng_state = [12345]

    def rng():
        rng_state[0] = (1664525 * rng_state[0] + 1013904223) & 0xFFFFFFFF
        return (rng_state[0] >> 8) / (1 << 24)

    def signal(t):
        return 100 + 1.2 * t + 15 * math.sin(2 * math.pi * (t % m) / m)
    n = 60
    y = [signal(t) + (rng() - 0.5) * 4 for t in range(n)]
    truth_future = [signal(n + i) for i in range(m)]

    print(f"  Monthly series ({n} months): trend +1.2/month + 12-month seasonal cycle + noise.\n")

    # fit all three, forecast 12 months, compare to the true future
    _, ses_fc = ses(y, 0.3)
    _, _, holt_fc = holt(y, 0.3, 0.1)
    lv, tr, seas, hw_fc = holt_winters(y, 0.3, 0.05, 0.3, period=m)

    def fc_rmse(fc):
        f = fc(m)
        return (sum((f[i] - truth_future[i]) ** 2 for i in range(m)) / m) ** 0.5

    print(f"  {'method':>16}  {'12-month forecast RMSE vs truth':>32}")
    print(f"  {'SES (flat)':>16}  {fc_rmse(ses_fc):>32.3f}")
    print(f"  {'Holt (trend)':>16}  {fc_rmse(holt_fc):>32.3f}")
    print(f"  {'Holt-Winters':>16}  {fc_rmse(hw_fc):>32.3f}")

    print("\n  SES forecasts a flat line (ignores trend and season); Holt adds the slope but misses")
    print("  the seasonal swing; Holt-Winters reinstates both and forecasts the year ahead accurately.")

    print(f"\n  Recovered seasonal pattern (12 months, additive deviations):")
    print("    " + "  ".join(f"{s:+.0f}" for s in seas))

    _svg(os.path.join(outdir, "holt_winters.svg"), y, lv, hw_fc(m), truth_future)
    print(f"\n  wrote {os.path.join(outdir, 'holt_winters.svg')}")


def _svg(path, y, levels, forecast, truth_future, width=760, height=400):
    n = len(y)
    h = len(forecast)
    total = n + h
    allv = y + forecast + truth_future
    lo, hi = min(allv), max(allv)
    ox, oy, ow, oh = 45, 50, width - 80, height - 100

    def sx(i):
        return ox + ow * i / (total - 1)

    def sy(v):
        return oy + oh * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Series (gray), Holt-Winters level (blue), forecast (green) vs truth (red dashed)</text>',
    ]
    # divider at forecast start
    parts.append(f'<line x1="{sx(n-1):.1f}" y1="{oy}" x2="{sx(n-1):.1f}" y2="{oy+oh}" '
                 f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
    # observed series
    for i in range(n):
        parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(y[i]):.1f}" r="1.8" fill="#8b949e" '
                     f'opacity="0.7"/>')
    # fitted level
    lp = " ".join(f"{sx(i):.1f},{sy(levels[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{lp}" fill="none" stroke="#4dabf7" stroke-width="1.5"/>')
    # forecast
    fp = " ".join(f"{sx(n + i):.1f},{sy(forecast[i]):.1f}" for i in range(h))
    parts.append(f'<polyline points="{fp}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    # truth future
    tp = " ".join(f"{sx(n + i):.1f},{sy(truth_future[i]):.1f}" for i in range(h))
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#ff6b6b" stroke-width="1.5" '
                 f'stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{sx(n)+5:.0f}" y="{oy+14}" fill="#8b949e" font-size="10">forecast -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
