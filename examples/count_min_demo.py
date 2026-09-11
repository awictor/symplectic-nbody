"""Demo: Count-Min sketch -- frequency estimates in sublinear memory.

Estimates item counts from a skewed stream, shows the estimate never falls below the truth and
stays within the error bound, and how wider tables shrink the error. Draws the estimate-vs-true
scatter (all on or above the diagonal) and the error-vs-width curve.

    python examples/count_min_demo.py [output_dir]
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from count_min import CountMinSketch  # noqa: E402


def _skewed(n, seed=1):
    state = seed
    out = []
    for _ in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        r = (state >> 8) / (1 << 24)
        if r < 0.30:
            out.append("A")
        elif r < 0.48:
            out.append("B")
        elif r < 0.58:
            out.append("C")
        else:
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            out.append(f"t{(state >> 16) % 3000}")
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    stream = _skewed(20000, seed=7)
    exact = Counter(stream)
    cms = CountMinSketch.from_error(0.001, 0.01).update(stream)

    print(f"Count-Min sketch: {len(exact)} distinct items counted in a "
          f"{cms.depth} x {cms.width} grid ({cms.depth * cms.width} counters)\n")
    print(f"  {'item':>8}{'true':>8}{'estimate':>10}{'error':>7}")
    for item in ("A", "B", "C"):
        est = cms.estimate(item)
        print(f"  {item:>8}{exact[item]:>8}{est:>10}{est - exact[item]:>7}")
    under = sum(1 for it in exact if cms.estimate(it) < exact[it])
    print(f"\n  never underestimates: {under} of {len(exact)} items fell below the truth")
    print(f"  error bound e/w * total = {cms.error_bound():.1f}; "
          f"max observed error = {max(cms.estimate(it) - exact[it] for it in exact)}\n")

    print("  Wider tables mean smaller error (fixed depth 4, 20k-item stream):")
    print(f"  {'width':>8}{'counters':>10}{'max error':>11}")
    for wexp in (7, 8, 9, 10, 11, 12):
        w = 1 << wexp
        sk = CountMinSketch(w, 4).update(stream)
        maxerr = max(sk.estimate(it) - exact[it] for it in exact)
        print(f"  {w:>8}{4*w:>10}{maxerr:>11}")
    print("\n  It never undercounts (collisions can only inflate a counter, and the query takes")
    print("  the min of the d rows), overshoots by a bounded amount, and merges by addition --")
    print("  so counting is distributed. Powers network flow monitors and n-gram frequency tables.")

    _svg(os.path.join(outdir, "count_min.svg"), stream, exact)
    print(f"\n  wrote {os.path.join(outdir, 'count_min.svg')}")


def _svg(path, stream, exact, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Count-Min sketch: estimates never fall below the truth</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'estimate vs true count -- all points on/above the diagonal (left); '
        f'max error shrinks with table width (right)</text>',
    ]

    # left: estimate vs true scatter (log-log), all >= diagonal
    import math
    cms = CountMinSketch(512, 4).update(stream)
    pts = [(exact[it], cms.estimate(it)) for it in exact]
    lx0, lx1 = 60, w // 2 - 20
    ly0, ly1 = h - 55, 65
    vmax = max(max(t, e) for t, e in pts)
    lo = 1

    def LX(v):
        return lx0 + (math.log10(max(v, lo)) / math.log10(vmax)) * (lx1 - lx0)

    def LY(v):
        return ly0 - (math.log10(max(v, lo)) / math.log10(vmax)) * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # perfect-estimate diagonal
    parts.append(f'<line x1="{LX(lo):.1f}" y1="{LY(lo):.1f}" x2="{LX(vmax):.1f}" y2="{LY(vmax):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{LX(vmax):.1f}" y="{LY(vmax)-5:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">exact</text>')
    for t, e in pts:
        parts.append(f'<circle cx="{LX(t):.1f}" cy="{LY(e):.1f}" r="2" fill="#4dabf7" opacity="0.6"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">true count (log) -> estimate (log)</text>')

    # right: max error vs width (fixed depth), log-x
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 65
    widths = [1 << e for e in range(6, 13)]
    errs = []
    for wd in widths:
        sk = CountMinSketch(wd, 4).update(stream)
        errs.append(max(sk.estimate(it) - exact[it] for it in exact))
    emax = max(errs) or 1

    def RX(wd):
        return rx0 + (math.log2(wd) - math.log2(widths[0])) / (math.log2(widths[-1]) - math.log2(widths[0])) * (rx1 - rx0)

    def RY(er):
        return ry0 - er / emax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    line = " ".join(f"{RX(wd):.1f},{RY(er):.1f}" for wd, er in zip(widths, errs))
    parts.append(f'<polyline points="{line}" fill="none" stroke="#ff922b" stroke-width="2.5"/>')
    for wd, er in zip(widths, errs):
        parts.append(f'<circle cx="{RX(wd):.1f}" cy="{RY(er):.1f}" r="2.5" fill="#ff922b"/>')
    for wd in (widths[0], widths[len(widths)//2], widths[-1]):
        parts.append(f'<text x="{RX(wd):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{wd}</text>')
    parts.append(f'<text x="{rx0-6:.1f}" y="{ry1-2:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">max error</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">table width (log) -> overestimate</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
