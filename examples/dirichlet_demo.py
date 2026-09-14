"""Demo: Dirichlet convolution -- the classical identities and Mobius inversion in one algebra.

Verifies the famous convolution identities (mu*1=eps, phi*1=id, 1*1=d, id*1=sigma, mu*id=phi),
demonstrates Mobius inversion recovering a function from its divisor sums, and tabulates the standard
arithmetic functions. Draws the arithmetic functions over 1..40.

    python examples/dirichlet_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dirichlet import (  # noqa: E402
    dirichlet_convolution, epsilon, one, identity, mobius, totient,
    divisor_count, divisor_sum, divisor_sum_transform, mobius_inversion,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    N = 40
    print("Dirichlet convolution: the arithmetic functions as a ring\n")

    eps, ones, idf = epsilon(N), one(N), identity(N)
    mu, phi = mobius(N), totient(N)

    def eq(a, b):
        return all(a[n] == b[n] for n in range(1, N + 1))

    print(f"  the classical identities (all verified exactly on 1..{N}):")
    print(f"    mu * 1  = epsilon   :  {eq(dirichlet_convolution(mu, ones, N), eps)}")
    print(f"    phi * 1 = id        :  {eq(dirichlet_convolution(phi, ones, N), idf)}")
    print(f"    1 * 1   = d (divisors): {eq(dirichlet_convolution(ones, ones, N), divisor_count(N))}")
    print(f"    id * 1  = sigma     :  {eq(dirichlet_convolution(idf, ones, N), divisor_sum(N))}")
    print(f"    mu * id = phi       :  {eq(dirichlet_convolution(mu, idf, N), phi)}")

    print(f"\n  standard arithmetic functions:")
    print(f"    {'n':>4}{'mu':>5}{'phi':>5}{'d':>4}{'sigma':>7}")
    d, sig = divisor_count(N), divisor_sum(N)
    for n in [1, 2, 6, 12, 30, 36]:
        print(f"    {n:>4}{mu[n]:>5}{phi[n]:>5}{d[n]:>4}{sig[n]:>7}")

    # Mobius inversion demo
    print(f"\n  Mobius inversion: recover f from its divisor sums F = f * 1, via f = F * mu")
    f = [0] + [n * n for n in range(1, N + 1)]   # f(n) = n^2
    F = divisor_sum_transform(f, N)
    recovered = mobius_inversion(F, N)
    print(f"    f(n) = n^2;  F = divisor sums;  F(12) = {F[12]}")
    print(f"    recovered f(12) = {recovered[12]} (should be 144): {recovered[12] == 144}")
    print(f"    full recovery exact: {all(recovered[n] == f[n] for n in range(1, N + 1))}")

    print(f"\n  This is the discrete fundamental theorem of calculus: 1 and mu are inverse under")
    print(f"  convolution, so divisor-summing and Mobius-inverting undo each other.")

    _svg(os.path.join(outdir, "dirichlet.svg"), mu, phi, d, sig, N)
    print(f"\n  wrote {os.path.join(outdir, 'dirichlet.svg')}")


def _svg(path, mu, phi, d, sig, N, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Arithmetic functions on 1..{N}: mu, phi, d, sigma (Dirichlet convolutions of 1, id)</text>',
    ]
    ox, oy, ow, oh = 45, 50, width - 80, height - 100
    # normalize each function to its own scale for display
    funcs = [("phi", phi, "#4dabf7"), ("sigma", sig, "#ff922b"),
             ("d x 5", [x * 5 for x in d], "#06d6a0")]
    vmax = max(max(vals[1:N + 1]) for _, vals, _ in funcs)

    def px(n):
        return ox + ow * (n - 1) / (N - 1)

    def py(v):
        return oy + oh * (1 - v / vmax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for name, vals, color in funcs:
        pts = " ".join(f"{px(n):.1f},{py(vals[n]):.1f}" for n in range(1, N + 1))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for n in range(1, N + 1):
            parts.append(f'<circle cx="{px(n):.1f}" cy="{py(vals[n]):.1f}" r="2" fill="{color}"/>')
    # mu as +1/0/-1 markers along the bottom
    for n in range(1, N + 1):
        y = oy + oh + 10
        col = "#06d6a0" if mu[n] == 1 else ("#ff6b6b" if mu[n] == -1 else "#484f58")
        parts.append(f'<circle cx="{px(n):.1f}" cy="{y:.1f}" r="2.5" fill="{col}"/>')
    parts.append(f'<text x="{ox}" y="{oy+oh+28:.0f}" fill="#8b949e" font-size="9">'
                 f'mu row: green +1, red -1, gray 0 (squareful)</text>')
    # legend
    ly = oy + 12
    for name, _, color in funcs:
        parts.append(f'<rect x="{ox+ow-90}" y="{ly-8}" width="12" height="3" fill="{color}"/>')
        parts.append(f'<text x="{ox+ow-74}" y="{ly-3}" fill="#e6edf3" font-size="10">{name}</text>')
        ly += 15
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
