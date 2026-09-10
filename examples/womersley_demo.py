"""Demo: the Womersley number -- pulsatile flow across the vascular tree.

Prints the Womersley number, penetration depth and phase lag for vessels from aorta to
capillary, then draws oscillatory velocity profiles: the quasi-steady parabola of small alpha
versus the blunt, wall-sheared plug of large alpha that lags the heartbeat.

    python examples/womersley_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from womersley import (womersley_from_heart_rate, penetration_depth,  # noqa: E402
                       phase_lag, pulse_wave_speed)


NU_BLOOD = 3.5e-6


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Womersley number alpha = R sqrt(omega/nu): does flow keep up with the heartbeat?\n")
    print("  Blood nu = 3.5e-6 m^2/s, heart rate 60 bpm\n")
    print(f"  {'vessel':<16}{'radius':>10}{'alpha':>9}{'phase lag':>12}{'regime':>16}")
    vessels = [
        ("aorta", 0.011),
        ("large artery", 0.004),
        ("arteriole", 1.5e-4),
        ("capillary", 4e-6),
    ]
    for name, R in vessels:
        a = womersley_from_heart_rate(R, 60.0, NU_BLOOD)
        lag_deg = math.degrees(phase_lag(a))
        regime = "quasi-steady" if a < 1 else ("plug (inertial)" if a > 10 else "transitional")
        print(f"  {name:<16}{R*1000:>8.2f}mm{a:>9.2f}{lag_deg:>10.0f} deg{regime:>16}")

    c = pulse_wave_speed(5e5, 0.002, 0.011)
    print("\n  Aortic pulse-wave speed (Moens-Korteweg): %.1f m/s -- the pressure pulse races" % c)
    print("  down the arterial tree far faster than the blood itself moves. High alpha in the")
    print("  aorta flattens the profile to a plug and makes the flow lag the pressure by ~90 deg.")

    _svg(os.path.join(outdir, "womersley.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'womersley.svg')}")


def _profile(eta, alpha):
    """Schematic oscillatory profile u(r)/u_max at radius fraction eta (0 center..1 wall).
    Small alpha -> parabola (1-eta^2); large alpha -> blunt plug with a thin wall layer."""
    para = 1.0 - eta * eta
    # plug: flat core, dropping to 0 only within ~1/alpha of the wall
    layer = max(0.05, 1.0 / alpha)
    if eta < 1.0 - layer:
        plug = 1.0
    else:
        plug = (1.0 - eta) / layer
    # blend from parabola (alpha small) to plug (alpha large)
    w = alpha * alpha / (alpha * alpha + 9.0)
    return (1.0 - w) * para + w * max(0.0, plug)


def _svg(path, size=720, pad=70):
    alphas = [(0.5, "#4dabf7", "alpha=0.5 (parabola)"),
              (3.0, "#06d6a0", "alpha=3 (transitional)"),
              (15.0, "#ff922b", "alpha=15 aorta (plug)")]

    # three side-by-side tube cross-sections showing the profile across the diameter
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Pulsatile velocity profiles vs Womersley number</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'low alpha follows the pressure as a parabola; high alpha lags it and flattens into a plug</text>',
    ]

    n = len(alphas)
    panel_w = (size - 2 * pad) / n
    tube_h = size * 0.42
    cy = size * 0.52
    prof_w = panel_w * 0.6

    for i, (alpha, col, label) in enumerate(alphas):
        cx = pad + panel_w * (i + 0.5)
        top = cy - tube_h / 2
        bot = cy + tube_h / 2
        # tube walls
        parts.append(f'<line x1="{cx-prof_w/2-20:.1f}" y1="{top:.1f}" x2="{cx+prof_w/2+20:.1f}" y2="{top:.1f}" '
                     f'stroke="#8b949e" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-prof_w/2-20:.1f}" y1="{bot:.1f}" x2="{cx+prof_w/2+20:.1f}" y2="{bot:.1f}" '
                     f'stroke="#8b949e" stroke-width="2"/>')
        # centre axis
        parts.append(f'<line x1="{cx:.1f}" y1="{top:.1f}" x2="{cx:.1f}" y2="{bot:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')

        # velocity profile: horizontal displacement proportional to u at each height
        m = 40
        pts = []
        for k in range(m + 1):
            frac = k / m                 # 0 at top wall .. 1 at bottom wall
            eta = abs(2.0 * frac - 1.0)  # radius fraction (0 center)
            u = _profile(eta, alpha)
            y = top + frac * tube_h
            pts.append(f"{cx + u * prof_w / 2:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        # a few arrows
        for k in (0.2, 0.35, 0.5, 0.65, 0.8):
            eta = abs(2.0 * k - 1.0)
            u = _profile(eta, alpha)
            y = top + k * tube_h
            parts.append(f'<line x1="{cx:.1f}" y1="{y:.1f}" x2="{cx + u*prof_w/2:.1f}" y2="{y:.1f}" '
                         f'stroke="{col}" stroke-width="1.2" opacity="0.55"/>')

        parts.append(f'<text x="{cx:.1f}" y="{bot+22:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{label}</text>')
        lag = math.degrees(phase_lag(alpha))
        parts.append(f'<text x="{cx:.1f}" y="{bot+38:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">lag ~ {lag:.0f} deg</text>')

    parts.append(f'<text x="{size/2:.1f}" y="{size-24:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">flow (arrow length = velocity across the tube diameter)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
