"""Demo: the Tisserand parameter is conserved across a gravity assist.

Integrates a massless body through a planetary flyby and shows its semi-major
axis and eccentricity jumping at the encounter while the Tisserand parameter
stays flat. Renders the three time series to SVG.

    python examples/tisserand_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tisserand import flyby  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    out = flyby(a0=1.6, e0=0.4, a_planet=1.0, m_planet=1e-3,
                dt=5e-4, steps=80000, sample_every=100)
    ts = [o[0] for o in out]
    a = [o[1] for o in out]
    e = [o[2] for o in out]
    T = [o[3] for o in out]

    k = max(1, len(out) // 5)
    def avg(seg, f):
        return sum(f(o) for o in seg) / len(seg)
    a0, a1 = avg(out[:k], lambda o: o[1]), avg(out[-k:], lambda o: o[1])
    e0, e1 = avg(out[:k], lambda o: o[2]), avg(out[-k:], lambda o: o[2])
    T0, T1 = avg(out[:k], lambda o: o[3]), avg(out[-k:], lambda o: o[3])

    print("Tisserand parameter across a gravity assist\n")
    print(f"{'':<10}{'before':>10}{'after':>10}{'change':>10}")
    print("-" * 40)
    print(f"{'a (AU)':<10}{a0:>10.3f}{a1:>10.3f}{a1-a0:>+10.3f}")
    print(f"{'e':<10}{e0:>10.3f}{e1:>10.3f}{e1-e0:>+10.3f}")
    print(f"{'T_planet':<10}{T0:>10.4f}{T1:>10.4f}{T1-T0:>+10.5f}")
    print("\na and e are reshaped by the flyby; the Tisserand parameter is nearly")
    print("unchanged. That is how Tisserand recognized returning comets whose orbits")
    print("Jupiter had scrambled -- and it bounds what one flyby can do.")

    _svg(ts, a, e, T, os.path.join(outdir, "tisserand.svg"))
    print(f"\nwrote {os.path.join(outdir, 'tisserand.svg')}")


def _svg(ts, a, e, T, path, size=720, pad=56):
    tmax = ts[-1]

    def sx(t):
        return pad + t / tmax * (size - 2 * pad)

    def scaler(vals):
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return lambda v: size - (pad + (v - lo) / span * (size - 2 * pad))

    sy_a, sy_e, sy_T = scaler(a), scaler(e), scaler(T)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
    ]
    for vals, sy, col, label in ((a, sy_a, "#e63946", "a"),
                                 (e, sy_e, "#f4a261", "e"),
                                 (T, sy_T, "#4cc9f0", "Tisserand")):
        poly = " ".join(f"{sx(ts[i]):.1f},{sy(vals[i]):.1f}" for i in range(len(ts)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    parts.append(f'<text x="{pad}" y="30" fill="#e6edf3" font-size="18">'
                 f'Gravity assist: a (red), e (orange) jump; Tisserand (blue) flat</text>')
    parts.append(f'<text x="{pad}" y="50" fill="#8b949e" font-size="12">'
                 f'each curve auto-scaled; the blue Tisserand track is nearly a straight line</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
