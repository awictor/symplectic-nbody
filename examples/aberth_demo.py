"""Aberth-Ehrlich demo: find every root of a polynomial at once, and watch the estimates spiral in from a circle."""

import cmath
import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import aberth
import durand_kerner as dk


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def _roots_with_trajectory(coeffs, max_iter, tol, seed):
    """Run Aberth but record every root estimate at each sweep, for the SVG spiral."""
    mono = aberth._normalise(coeffs)
    n = len(mono) - 1
    z = aberth._initial_guesses(mono, seed)
    traj = [list(z)]
    for it in range(max_iter):
        new_z = list(z)
        done = True
        for i in range(n):
            p, dp = aberth._horner(mono, z[i])
            if p == 0:
                continue
            if dp == 0:
                dp = 1e-30
            ratio = p / dp
            s = sum(1.0 / (z[i] - z[j]) for j in range(n) if j != i and z[i] != z[j])
            denom = 1.0 - ratio * s
            if denom == 0:
                denom = 1e-30
            w = ratio / denom
            new_z[i] = z[i] - w
            if abs(w) > tol * (1 + abs(z[i])):
                done = False
        z = new_z
        traj.append(list(z))
        if done:
            break
    return z, traj


def main():
    lines = []
    lines.append("Aberth-Ehrlich -- all polynomial roots at once, cubic convergence")
    lines.append("=" * 66)
    lines.append("")

    # a degree-7 polynomial with mixed real + complex roots
    true_roots = [2.0, -3.0, 1 + 2j, 1 - 2j, -1 + 1j, -1 - 1j, 4.0]
    coeffs = aberth.from_roots(true_roots)
    z, traj = _roots_with_trajectory(coeffs, max_iter=100, tol=1e-14, seed=7)

    lines.append(f"Degree-{len(true_roots)} polynomial, roots refined simultaneously from a circle.")
    lines.append(f"Converged in {len(traj) - 1} sweeps.")
    lines.append("")
    lines.append("   found root                 |p(root)|")
    lines.append("  " + "-" * 44)
    zs = sorted(z, key=lambda w: (round(w.real, 4), round(w.imag, 4)))
    for w in zs:
        cw = aberth._clean(w, 1e-9)
        lines.append(f"  {cw.real:+8.5f} {cw.imag:+8.5f}i     {aberth.residual(coeffs, w):.2e}")
    lines.append("")

    # convergence rate: max residual per sweep (should fall super-fast)
    lines.append("Max root-residual per sweep (cubic: digits roughly triple):")
    lines.append("   sweep   max |p(z_i)|")
    lines.append("   " + "-" * 26)
    resids = []
    last = len(traj) - 1
    show = {0, 5, 10, last - 3, last - 2, last - 1, last}
    for k, step in enumerate(traj):
        mx = max(aberth.residual(coeffs, w) for w in step)
        resids.append(mx)
        if k in show:
            lines.append(f"   {k:4d}    {mx:.3e}")
    lines.append("")

    # iteration-count comparison vs Durand-Kerner (same circle seeding)
    _, it_ab = aberth.roots(coeffs, tol=1e-12, track=True)
    it_dk = _dk_count(coeffs, tol=1e-12)
    lines.append(f"Sweeps to 1e-12 (same seeding): Aberth {it_ab}  vs  Durand-Kerner {it_dk}")
    lines.append("  -> cubic convergence reaches tolerance in fewer sweeps.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(traj, true_roots)
    return text, svg


def _dk_count(coeffs, tol):
    mono = aberth._normalise(coeffs)
    n = len(mono) - 1
    z = aberth._initial_guesses(mono, 12345)

    def pev(x):
        p = mono[0]
        for c in mono[1:]:
            p = p * x + c
        return p

    for it in range(500):
        ms = 0.0
        new = list(z)
        for i in range(n):
            d = mono[0]
            for j in range(n):
                if j != i:
                    d *= (z[i] - z[j])
            if d == 0:
                d = 1e-30
            w = pev(z[i]) / d
            new[i] = z[i] - w
            ms = max(ms, abs(w))
        z = new
        if ms < tol:
            return it + 1
    return 500


def _svg(traj, true_roots):
    W, H = 640, 460
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Aberth-Ehrlich: {len(true_roots)} root estimates spiral in from a circle</text>')

    # extent from all points in all sweeps + true roots
    allpts = [w for step in traj for w in step] + [complex(r) for r in true_roots]
    reals = [w.real for w in allpts]
    imags = [w.imag for w in allpts]
    rmin, rmax = min(reals), max(reals)
    imin, imax = min(imags), max(imags)
    padr = 0.1 * max(rmax - rmin, 1e-9)
    padi = 0.1 * max(imax - imin, 1e-9)
    rmin -= padr
    rmax += padr
    imin -= padi
    imax += padi

    x0, x1, y0, y1 = 55, 610, 55, 400

    def px(re):
        return x0 + (re - rmin) / (rmax - rmin) * (x1 - x0)

    def py(im):
        return y1 - (im - imin) / (imax - imin) * (y1 - y0)

    # zero axes
    if rmin < 0 < rmax:
        parts.append(f'<line x1="{px(0):.1f}" y1="{y0}" x2="{px(0):.1f}" y2="{y1}" stroke="#333" stroke-width="1"/>')
    if imin < 0 < imax:
        parts.append(f'<line x1="{x0}" y1="{py(0):.1f}" x2="{x1}" y2="{py(0):.1f}" stroke="#333" stroke-width="1"/>')

    n = len(true_roots)
    # trajectory of each root estimate (one polyline per root index)
    colors = [P["blue"], P["yellow"], P["green"], P["purple"], P["red"], "#f783ac", "#63e6be"]
    for i in range(n):
        pts = " ".join(f"{px(step[i].real):.1f},{py(step[i].imag):.1f}" for step in traj)
        col = colors[i % len(colors)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                     f'stroke-width="1.3" opacity="0.75"/>')
        # start marker (hollow, on the circle)
        s = traj[0][i]
        parts.append(f'<circle cx="{px(s.real):.1f}" cy="{py(s.imag):.1f}" r="3" '
                     f'fill="none" stroke="{col}" stroke-width="1"/>')

    # true roots as big white X's
    for r in true_roots:
        cr = complex(r)
        x, y = px(cr.real), py(cr.imag)
        parts.append(f'<line x1="{x-5:.1f}" y1="{y-5:.1f}" x2="{x+5:.1f}" y2="{y+5:.1f}" '
                     f'stroke="{P["text"]}" stroke-width="2"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{y+5:.1f}" x2="{x+5:.1f}" y2="{y-5:.1f}" '
                     f'stroke="{P["text"]}" stroke-width="2"/>')

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'hollow circles = initial guesses on the Cauchy-bound circle; '
                 f'white X = true roots.</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'The estimates repel each other (never collide) and lock onto every root at once.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
