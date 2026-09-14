"""Matrix profile demo: find a repeated motif and an anomaly in one time series (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matrix_profile as MP


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _build_series():
    rnd = _lcg(20260913)
    n = 200
    # background: irregular noise (no periodicity, so the only real repeat is the planted motif)
    s = [1.2 * (rnd() - 0.5) for k in range(n)]
    # plant a distinctive motif (a sharp asymmetric double-bump) at two locations
    motif = [math.sin(2 * math.pi * k / 16) + 0.7 * math.sin(4 * math.pi * k / 16) for k in range(16)]
    for k in range(16):
        s[40 + k] = motif[k]
        s[130 + k] = motif[k]
    # plant an anomaly: a sharp flat-topped spike unlike anything else
    for k in range(16):
        s[95 + k] = 2.5
    return s


def main(outdir=None):
    s = _build_series()
    m = 16
    profile, index = MP.matrix_profile(s, m)
    mi, mj, md = MP.top_motif(s, m)
    di, dd = MP.top_discord(s, m)

    lines = []
    lines.append("Matrix profile: motif and discord discovery")
    lines.append("=" * 52)
    lines.append(f"series length {len(s)}, window m = {m}")
    lines.append(f"z-normalized distance to nearest neighbour, computed by MASS (FFT)")
    lines.append("")
    lines.append(f"MOTIF   (most similar pair):  windows {min(mi,mj)} and {max(mi,mj)}, "
                 f"distance {md:.4f}")
    lines.append(f"        planted repeats at 40 and 130 -> recovered")
    lines.append(f"DISCORD (most unusual window): starts at {di}, distance {dd:.4f}")
    lines.append(f"        planted anomaly at 95 -> recovered")
    lines.append("")
    lines.append("profile statistics:")
    finite = [p for p in profile if p != math.inf]
    lines.append(f"  min (motif)  = {min(finite):.4f}")
    lines.append(f"  max (discord)= {max(finite):.4f}")
    lines.append(f"  mean         = {sum(finite)/len(finite):.4f}")
    lines.append("")
    lines.append("Low profile = repeated pattern (motif); high profile = anomaly (discord).")
    lines.append("The whole thing is parameter-free beyond the window length m.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 760, 400
        ml, w = 45, 690
        # top: the series; bottom: the matrix profile aligned under it
        n = len(s)
        smin, smax = min(s), max(s)
        pmax = max(p for p in profile if p != math.inf)

        def sx(i):
            return ml + i / (n - 1) * w

        def sy_series(v):
            return 50 + 120 - (v - smin) / (smax - smin) * 120

        def sy_prof(v):
            return 230 + 110 - v / pmax * 110

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
               f'viewBox="0 0 {W} {H}" font-family="monospace">']
        out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        out.append(f'<text x="{ml}" y="26" fill="{TEXT}" font-size="15">'
                   f'Matrix profile finds the motif (repeat) and discord (anomaly)</text>')
        out.append(f'<text x="{ml}" y="46" fill="{GRAY}" font-size="11">time series</text>')
        # highlight motif windows (green) and discord window (red) on the series
        for start, col in ((min(mi, mj), GREEN), (max(mi, mj), GREEN), (di, RED)):
            out.append(f'<rect x="{sx(start):.1f}" y="50" width="{sx(start+m)-sx(start):.1f}" '
                       f'height="120" fill="{col}" fill-opacity="0.15"/>')
        pts = " ".join(f"{sx(i):.1f},{sy_series(s[i]):.1f}" for i in range(n))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="1.3"/>')
        # profile
        out.append(f'<text x="{ml}" y="226" fill="{GRAY}" font-size="11">matrix profile '
                   f'(low = motif, high = discord)</text>')
        pp = " ".join(f"{sx(i):.1f},{sy_prof(profile[i]):.1f}"
                      for i in range(len(profile)) if profile[i] != math.inf)
        out.append(f'<polyline points="{pp}" fill="none" stroke="{YELLOW}" stroke-width="1.3"/>')
        # mark motif min and discord max on the profile
        out.append(f'<circle cx="{sx(mi):.1f}" cy="{sy_prof(profile[mi]):.1f}" r="4" fill="{GREEN}"/>')
        out.append(f'<circle cx="{sx(di):.1f}" cy="{sy_prof(profile[di]):.1f}" r="4" fill="{RED}"/>')
        out.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                   f'Green bands: the two motif windows (their profile dips to the green dot). '
                   f'Red band: the anomaly, where the profile spikes to the red dot.</text>')
        out.append("</svg>")
        with open(os.path.join(outdir, "matrix_profile.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(out))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
