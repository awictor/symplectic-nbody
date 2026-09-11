"""Demo: the Kalman filter tracking a noisy moving object.

An object moves at roughly constant velocity; a noisy sensor reports only its position. The filter
fuses the motion model with the measurements to estimate position AND the unobserved velocity,
driving its error well below the raw sensor's; the RTS smoother then uses the whole record to do
better still. Shows the estimate variance collapsing to a steady state and the tracks overlaid.

    python examples/kalman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kalman import constant_velocity, rts_smoother  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 5

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    def gauss(sd):
        u1 = max(1e-9, rng())
        u2 = rng()
        return sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    dt = 1.0
    meas_sd = 6.0
    T = 80
    # a gently accelerating truth so the constant-velocity model is only approximate
    truth = [0.5 * 0.02 * t * t + 2.0 * t for t in range(T)]
    meas = [[truth[t] + gauss(meas_sd)] for t in range(T)]

    kf = constant_velocity(dt, process_var=0.05, meas_var=meas_sd ** 2, x0=[0.0, 0.0])
    means, covs, _, _ = kf.filter(meas)
    kf2 = constant_velocity(dt, process_var=0.05, meas_var=meas_sd ** 2, x0=[0.0, 0.0])
    sm_means, sm_covs = rts_smoother(kf2, meas)

    def rmse(est, lo=8):
        return math.sqrt(sum((est[t] - truth[t]) ** 2 for t in range(lo, T)) / (T - lo))

    raw = rmse([m[0] for m in meas])
    filt = rmse([m[0] for m in means])
    smooth = rmse([m[0] for m in sm_means])

    print("Kalman filter: optimal tracking of a hidden state from noisy measurements\n")
    print(f"  {T} steps, position sensor with noise sd = {meas_sd}\n")
    print(f"  RMSE vs truth:  raw measurements {raw:5.2f}")
    print(f"                  Kalman filter    {filt:5.2f}   ({100*(1-filt/raw):.0f}% better than raw)")
    print(f"                  RTS smoother     {smooth:5.2f}   ({100*(1-smooth/raw):.0f}% better than raw)\n")

    print(f"  Estimate variance collapses from {covs[0][0][0]:.1f} to a steady "
          f"{covs[-1][0][0]:.2f} (measurement variance is {meas_sd**2:.0f}):")
    for t in (0, 1, 2, 5, 10, 40, T - 1):
        bar = "#" * int(round(covs[t][0][0] / covs[0][0][0] * 30))
        print(f"    step {t:>2}: var {covs[t][0][0]:6.2f} {bar}")

    print(f"\n  Velocity is never measured, only inferred: final estimate "
          f"{means[-1][1]:.2f} (truth ~{2.0 + 0.02 * T:.2f}).\n")

    print("  Predict pushes the belief through the motion model (variance grows); update folds in")
    print("  a measurement weighted by the Kalman gain (variance shrinks). The fused estimate beats")
    print("  either the model or the sensor alone -- and the backward smoother, using future data,")
    print("  beats the causal filter. This is the math behind GPS, guidance, and sensor fusion.")

    _svg(os.path.join(outdir, "kalman.svg"), truth, meas, means, sm_means, covs)
    print(f"\n  wrote {os.path.join(outdir, 'kalman.svg')}")


def _svg(path, truth, meas, means, sm_means, covs, width=760, height=430):
    T = len(truth)
    lx0, lx1 = 45, width // 2 - 10
    y0, y1 = height - 45, 65
    allv = [m[0] for m in meas] + truth
    va, vb = min(allv), max(allv)

    def LX(t):
        return lx0 + t / (T - 1) * (lx1 - lx0)

    def LY(v):
        return y0 - (v - va) / (vb - va) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Kalman filter: fusing a motion model with a noisy sensor</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: truth vs noisy measurements vs filtered vs smoothed track; '
        f'right: estimate variance collapsing</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]

    # measurements as scattered dots
    for t in range(T):
        parts.append(f'<circle cx="{LX(t):.1f}" cy="{LY(meas[t][0]):.1f}" r="2" '
                     f'fill="#8b949e" opacity="0.55"/>')
    # truth (green), filter (blue), smoother (yellow)
    def poly(seq, color, w):
        pts = " ".join(f"{LX(t):.1f},{LY(seq[t]):.1f}" for t in range(T))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}"/>'

    parts.append(poly(truth, "#06d6a0", 2.2))
    parts.append(poly([m[0] for m in means], "#4dabf7", 1.8))
    parts.append(poly([m[0] for m in sm_means], "#ffd43b", 1.8))
    # legend
    leg = [("#06d6a0", "truth"), ("#8b949e", "measured"), ("#4dabf7", "filtered"),
           ("#ffd43b", "smoothed")]
    for i, (c, lab) in enumerate(leg):
        yy = y1 + 4 + i * 16
        parts.append(f'<rect x="{lx0+6}" y="{yy-9}" width="12" height="6" fill="{c}"/>')
        parts.append(f'<text x="{lx0+22}" y="{yy-3}" fill="#8b949e" font-size="10">{lab}</text>')

    # right: variance collapse
    rx0, rx1 = width // 2 + 45, width - 30
    vv = [covs[t][0][0] for t in range(T)]
    vmax = max(vv)

    def RX(t):
        return rx0 + t / (T - 1) * (rx1 - rx0)

    def RY(v):
        return y0 - v / vmax * (y0 - y1)

    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(t):.1f},{RY(vv[t]):.1f}" for t in range(T))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#ff6b6b" stroke-width="2.2"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">position-estimate variance vs step</text>')
    parts.append(f'<text x="{rx0+4:.1f}" y="{RY(vmax)-3:.1f}" fill="#8b949e" font-size="9">'
                 f'{vmax:.0f}</text>')
    parts.append(f'<text x="{rx0+4:.1f}" y="{y0-3:.1f}" fill="#8b949e" font-size="9">0</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
