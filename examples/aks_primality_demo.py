"""AKS demo: the polynomial-congruence primality proof, showing the identity that separates prime from composite (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aks_primality as A


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
GREEN = "#06d6a0"
RED = "#ff6b6b"
BLUE = "#4dabf7"


def main(outdir=None):
    lines = []
    lines.append("AKS: deterministic, unconditional, polynomial-time primality")
    lines.append("=" * 60)
    lines.append("core identity: n is prime  <=>  (x + a)^n == x^n + a  in Z[x]  (gcd(a,n)=1)")
    lines.append("AKS checks it modulo (x^r - 1, n) for a small r and a bounded set of a.")
    lines.append("")
    lines.append("certificates (r = order parameter, a_bound = # polynomial checks):")
    lines.append(f"{'n':>6}{'prime?':>11}{'r':>6}{'a_bound':>9}")
    for n in (31, 33, 97, 100, 101, 561, 1009):
        ok, r, ab = A.is_prime(n, certificate=True)
        lines.append(f"{n:>6}{('PRIME' if ok else 'composite'):>11}{str(r):>6}{str(ab):>9}")
    lines.append("")
    lines.append("the polynomial identity, made concrete for small n and r:")
    for n, r in ((7, 5), (9, 5), (11, 7), (15, 7)):
        holds = A._check_polynomial(n, r, 1)
        truth = "prime" if A.trial_division_prime(n) else "composite"
        lines.append(f"  (x+1)^{n} == x^{n}+1 mod (x^{r}-1,{n})?  {holds}   (n is {truth})")
    lines.append("")
    lines.append("Every prime satisfies the congruence for all a; every composite fails it for")
    lines.append("some a. AKS proved this can be checked in polynomial time -- primality is in P.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: coefficient grids of (x+1)^n and x^n+1 mod (x^r-1, n) side by side for a prime and composite
        W, H = 720, 400
        r = 7

        def coeff_grid(s, n, ox, oy, title):
            base = [0] * r
            base[0] = 1 % n
            base[1] = (base[1] + 1) % n
            left = A._poly_pow(base, n, r, n)
            right = [0] * r
            right[n % r] = (right[n % r] + 1) % n
            right[0] = (right[0] + 1) % n
            s.append(f'<text x="{ox}" y="{oy-10}" fill="{TEXT}" font-size="12">{title}</text>')
            cell = 34
            for row, (label, poly) in enumerate([("(x+1)^n", left), ("x^n+1", right)]):
                s.append(f'<text x="{ox-2}" y="{oy+row*(cell+22)+cell*0.65:.0f}" fill="{GRAY}" '
                         f'font-size="10">{label}</text>')
                for k in range(r):
                    match = left[k] == right[k]
                    col = "#161b22"
                    s.append(f'<rect x="{ox+60+k*cell}" y="{oy+row*(cell+22)}" width="{cell-2}" '
                             f'height="{cell-2}" fill="{col}" stroke="{GRAY}" stroke-width="0.5"/>')
                    s.append(f'<text x="{ox+60+k*cell+(cell-2)/2:.0f}" '
                             f'y="{oy+row*(cell+22)+cell*0.62:.0f}" fill="{TEXT}" font-size="12" '
                             f'text-anchor="middle">{poly[k]}</text>')
            # verdict row: green if all match, red otherwise
            allmatch = left == right
            vc = GREEN if allmatch else RED
            s.append(f'<text x="{ox+60}" y="{oy+2*(cell+22)+16:.0f}" fill="{vc}" font-size="12">'
                     f'{"identical -> consistent with prime" if allmatch else "differ -> COMPOSITE"}</text>')

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="30" fill="{TEXT}" font-size="15">'
                 f'AKS polynomial identity mod (x^{r}-1, n): prime vs composite</text>')
        coeff_grid(s, 11, 30, 80, "n = 11 (prime): coefficient vectors match")
        coeff_grid(s, 15, 30, 240, "n = 15 (composite): coefficient vectors differ")
        s.append(f'<text x="30" y="{H-12}" fill="{GRAY}" font-size="10">'
                 f'Each cell is a polynomial coefficient mod n; when the two rows are identical for '
                 f'all a, n is prime -- the theorem AKS turned into a fast algorithm.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "aks_primality.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
