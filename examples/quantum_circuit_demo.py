"""Demo: quantum circuit simulation -- Grover's search amplifying a marked state, and a Bell pair.

Runs Grover's algorithm over a 2^n search space, printing the marked state's probability after each
iteration as it rises toward 1, then overshoots if you keep going -- the amplitude-amplification
signature. Draws the probability of the marked state vs iteration count.

    python examples/quantum_circuit_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_circuit import (  # noqa: E402
    QuantumState,
    grover_search,
    grover_optimal_iterations,
    bell_pair,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Quantum circuit simulator: Grover's search by amplitude amplification\n")

    # --- Bell pair first, to show entanglement ---
    bell = bell_pair()
    corr = bell.measure_correlations()
    print(f"  Bell pair (|00> + |11>)/sqrt(2):")
    print(f"    P(00)={corr['00']:.3f}  P(01)={corr['01']:.3f}  "
          f"P(10)={corr['10']:.3f}  P(11)={corr['11']:.3f}")
    print(f"    -> perfectly correlated: only 00 or 11, never 01 or 10 (entanglement)\n")

    # --- Grover over a 2^n space ---
    n = 6
    N = 1 << n
    marked = 42
    opt = grover_optimal_iterations(n)
    print(f"  Grover search: N = 2^{n} = {N} items, marked item = {marked}")
    print(f"  classical search needs ~{N // 2} queries; Grover needs ~{opt} "
          f"((pi/4)sqrt(N))\n")

    print(f"    {'iteration':>10}{'P(marked)':>12}")
    curve = []
    max_iters = 2 * opt + 2
    for it in range(0, max_iters + 1):
        if it == 0:
            st = QuantumState(n)
            st.hadamard_all()
        else:
            st, _ = grover_search(marked, n, iterations=it)
        pm = st.probability(marked)
        curve.append((it, pm))
        mark = "  <- optimal" if it == opt else ("  (overshoot)" if it > opt and pm < curve[it - 1][1] else "")
        if it <= opt + 2 or it == max_iters:
            print(f"    {it:>10}{pm:>12.4f}{mark}")

    best = max(curve, key=lambda c: c[1])
    print(f"\n  Peak P(marked) = {best[1]:.4f} at iteration {best[0]} "
          f"(starting from a uniform 1/{N} = {1/N:.4f})")
    print(f"  Beyond the optimum the probability falls again -- amplitude amplification")
    print(f"  is a rotation, so too many steps rotate past the target.")

    _svg(os.path.join(outdir, "quantum_circuit.svg"), curve, opt, 1 / N)
    print(f"\n  wrote {os.path.join(outdir, 'quantum_circuit.svg')}")


def _svg(path, curve, opt, uniform, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Grover: probability of the marked state vs iteration (amplitude amplification)</text>',
    ]
    ox, oy, ow, oh = 55, 55, width - 100, height - 110
    imax = curve[-1][0]

    def px(i):
        return ox + ow * i / imax

    def py(p):
        return oy + oh * (1 - p)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py(frac)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{frac:.2f}</text>')
    # uniform baseline
    parts.append(f'<line x1="{ox}" y1="{py(uniform):.1f}" x2="{ox+ow}" y2="{py(uniform):.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{ox+4}" y="{py(uniform)-3:.1f}" fill="#8b949e" font-size="9">'
                 f'uniform 1/N</text>')
    # optimal marker
    parts.append(f'<line x1="{px(opt):.1f}" y1="{oy}" x2="{px(opt):.1f}" y2="{oy+oh}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{px(opt)+4:.0f}" y="{oy+14}" fill="#06d6a0" font-size="10">'
                 f'optimal ~{opt}</text>')
    # curve
    pts = " ".join(f"{px(i):.1f},{py(p):.1f}" for i, p in curve)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for i, p in curve:
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(p):.1f}" r="3" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+24:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">Grover iterations</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
