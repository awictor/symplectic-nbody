"""Demo: Shor's algorithm factoring 21 -- the quantum period-finding step and the classical wrap-up.

Runs Shor's algorithm end to end on N = 21: picks a base a, finds the period r of a^x mod 21 via the
simulated QFT, and recovers the factors by gcd. Prints the period-finding and shows the QFT output
spectrum whose peaks encode the period.

    python examples/shor_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shor import shor_factor, multiplicative_order, gcd  # noqa: E402
from qft import qft, state_from_amplitudes  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Shor's algorithm: factoring by quantum period-finding\n")

    N = 21
    a = 2
    print(f"  factor N = {N}, base a = {a}")
    print(f"  f(x) = {a}^x mod {N} is periodic; its period is the order of {a} mod {N}\n")

    # show the periodic sequence
    seq = [pow(a, x, N) for x in range(12)]
    print(f"  {a}^x mod {N}:  {seq}")
    r = multiplicative_order(a, N)
    print(f"  period r = {r}  (since {a}^{r} mod {N} = {pow(a, r, N)})\n")

    # quantum period-finding spectrum
    n_count = 7
    Q = 1 << n_count
    support = [x for x in range(Q) if x % r == 0]
    amps = [0j] * Q
    v = 1 / math.sqrt(len(support))
    for x in support:
        amps[x] = v
    st = state_from_amplitudes(amps)
    qft(st)
    probs = st.probabilities()
    peaks = sorted(range(Q), key=lambda k: -probs[k])[:r]
    print(f"  QFT over {Q} states: probability peaks at multiples of {Q}/{r} = {Q // r}:")
    print(f"    peaks at {sorted(peaks)}")

    # classical wrap-up
    x = pow(a, r // 2, N)
    p, q = gcd(x - 1, N), gcd(x + 1, N)
    print(f"\n  r is even, so compute a^(r/2) = {a}^{r // 2} mod {N} = {x}")
    print(f"  gcd({x}-1, {N}) = {p},  gcd({x}+1, {N}) = {q}")

    factors = shor_factor(N, quantum=True)
    print(f"\n  Shor's algorithm result: {N} = {factors[0]} x {factors[1]}")
    print(f"\n  The exponential speed-up is entirely in finding r: classically hard, but the QFT")
    print(f"  reads the period off the interference pattern above in one shot. This is why a large")
    print(f"  quantum computer would break RSA.")

    _svg(os.path.join(outdir, "shor.svg"), probs, Q // r, r)
    print(f"\n  wrote {os.path.join(outdir, 'shor.svg')}")


def _svg(path, probs, spacing, r, width=760, height=380):
    N = len(probs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Shor period-finding: QFT peaks at multiples of Q/r encode the period r={r}</text>',
    ]
    ox, oy, ow, oh = 50, 50, width - 90, height - 100
    pmax = max(probs) * 1.1

    def bx(k):
        return ox + ow * k / N

    bw = max(ow / N * 0.8, 1.0)
    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for k in range(N):
        h = oh * probs[k] / pmax
        color = "#ffd43b" if probs[k] > 0.4 * max(probs) else "#4dabf7"
        parts.append(f'<rect x="{bx(k):.1f}" y="{oy+oh-h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'fill="{color}"/>')
    kk = 0
    while kk < N:
        parts.append(f'<line x1="{bx(kk):.1f}" y1="{oy}" x2="{bx(kk):.1f}" y2="{oy+oh}" '
                     f'stroke="#06d6a0" stroke-width="0.7" stroke-dasharray="3 3"/>')
        kk += spacing
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">output basis state (green lines every Q/r = {spacing})</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
