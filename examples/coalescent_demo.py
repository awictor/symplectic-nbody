"""Coalescent demo: build a gene genealogy backward in time, verify E[TMRCA], and estimate theta from mutations."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import coalescent as co


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Kingman's coalescent -- gene genealogies backward in time")
    lines.append("=" * 58)
    lines.append("")

    lines.append("Coalescent expectations vs simulation (time in units of N generations):")
    lines.append("   n     E[T_MRCA]   sim      E[total len]   sim")
    lines.append("   " + "-" * 48)
    for n in (2, 5, 10, 20, 50):
        # spread seeds widely -- sequential LCG seeds give correlated first draws
        tm = [co.simulate(n, seed=(r + 1) * 7919)["t_mrca"] for r in range(5000)]
        tl = [co.simulate(n, seed=(r + 1) * 7919)["total_length"] for r in range(5000)]
        lines.append(f"   {n:3d}   {co.expected_t_mrca(n):8.3f}   {sum(tm)/len(tm):.3f}   "
                     f"{co.expected_total_length(n):8.3f}      {sum(tl)/len(tl):.3f}")
    lines.append("")
    lines.append("T_MRCA approaches 2 (most depth is the last two lineages merging);")
    lines.append("total length grows only as 2 H_(n-1) (logarithmically) -- adding samples adds little.")
    lines.append("")

    # Watterson's estimator
    lines.append("Watterson's estimator of theta = 4 N mu from segregating sites S:")
    lines.append("   true theta   n    mean S    theta_hat = S / H_(n-1)")
    lines.append("   " + "-" * 48)
    for theta in (1.0, 5.0, 10.0):
        n = 20
        Ss = [co.segregating_sites(n, theta, seed=(r + 1) * 7919)[0] for r in range(3000)]
        meanS = sum(Ss) / len(Ss)
        th = co.watterson_theta(meanS, n)
        lines.append(f"   {theta:8.1f}   {n:3d}   {meanS:6.2f}    {th:.3f}")
    lines.append("")
    lines.append("S grows with both mutation rate and (log of) sample size; theta_hat inverts it.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'A coalescent genealogy: 8 samples merging back to their MRCA</text>')

    n = 8
    geneal = co.simulate(n, seed=7)
    ct = geneal["coalescence_times"]
    t_mrca = geneal["t_mrca"]

    # cumulative times (from present=0 at bottom to MRCA at top)
    cum = [0.0]
    for t in ct:
        cum.append(cum[-1] + t)
    # heights: level j (0..n-1) at cumulative time cum[j]; leaves at time 0

    x0, x1, y0, y1 = 55, 610, 60, 370

    def ty(t):
        return y1 - t / t_mrca * (y1 - y0)

    # We build the tree by merging adjacent active lineages (random pairing simplified to nearest x).
    # positions of the n leaves
    xs = [x0 + i / (n - 1) * (x1 - x0) for i in range(n)]
    active = [(xs[i], 0.0) for i in range(n)]   # (x, birth-time) of each active lineage

    rng = co._Rng(7)
    level = 0
    for step in range(n - 1):
        t_coal = cum[step + 1]
        # pick two random lineages to merge
        i = int(rng.u() * len(active))
        j = int(rng.u() * (len(active) - 1))
        if j >= i:
            j += 1
        (xi, bi), (xj, bj) = active[i], active[j]
        # draw the two branches up to the coalescence height
        for (xx, bb) in ((xi, bi), (xj, bj)):
            parts.append(f'<line x1="{xx:.1f}" y1="{ty(bb):.1f}" x2="{xx:.1f}" y2="{ty(t_coal):.1f}" '
                         f'stroke="{P["blue"]}" stroke-width="1.5"/>')
        # horizontal connector
        parts.append(f'<line x1="{xi:.1f}" y1="{ty(t_coal):.1f}" x2="{xj:.1f}" y2="{ty(t_coal):.1f}" '
                     f'stroke="{P["blue"]}" stroke-width="1.5"/>')
        # the merged lineage sits at the midpoint
        merged_x = (xi + xj) / 2
        new_active = [active[k] for k in range(len(active)) if k not in (i, j)]
        new_active.append((merged_x, t_coal))
        active = new_active

    # leaf markers
    for i in range(n):
        parts.append(f'<circle cx="{xs[i]:.1f}" cy="{ty(0):.1f}" r="4" fill="{P["green"]}"/>')
    # MRCA marker
    parts.append(f'<circle cx="{active[0][0]:.1f}" cy="{ty(t_mrca):.1f}" r="5" fill="{P["red"]}"/>')
    parts.append(f'<text x="{active[0][0] + 8:.1f}" y="{ty(t_mrca) + 4:.1f}" fill="{P["red"]}" '
                 f'font-size="11">MRCA (t={t_mrca:.2f})</text>')

    # time axis
    parts.append(f'<line x1="{x0 - 15}" y1="{ty(0):.1f}" x2="{x0 - 15}" y2="{ty(t_mrca):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x0 - 45}" y="{ty(0) + 4:.1f}" fill="{P["gray"]}" font-size="10">present</text>')
    parts.append(f'<text x="{x0 - 45}" y="{ty(t_mrca) + 4:.1f}" fill="{P["gray"]}" font-size="10">past</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'green = samples today; each merge is a coalescence; waiting times shrink with more '
                 f'lineages, so the last merge dominates the depth.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
