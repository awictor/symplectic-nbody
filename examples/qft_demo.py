"""Demo: the quantum Fourier transform finding the period of a signal -- the core of Shor's algorithm.

Prepares an n-qubit register in a state periodic in the computational basis (amplitude on every r-th
basis state), applies the QFT, and shows the probability piling up at multiples of N/r -- exactly the
period-finding step that lets Shor's algorithm factor integers. Also runs phase estimation. Draws the
QFT output probability spectrum.

    python examples/qft_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qft import qft, phase_estimation  # noqa: E402
from quantum_circuit import QuantumState  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Quantum Fourier transform: period finding (the heart of Shor's algorithm)\n")

    n = 5
    N = 1 << n
    r = 4  # period
    # a state uniform over basis states congruent to an offset mod r (like Shor's post-measurement state)
    offset = 1
    support = [k for k in range(N) if k % r == offset]
    st = QuantumState(n)
    st.amp = [0j] * N
    a = 1 / math.sqrt(len(support))
    for k in support:
        st.amp[k] = a

    print(f"  register: {n} qubits, N = {N} basis states")
    print(f"  input state: uniform over k = {offset} mod {r}  ->  {support}")
    print(f"  (this is the state Shor's algorithm holds after measuring the modular-exponent register)\n")

    qft(st)
    probs = st.probabilities()

    print(f"  after QFT, probability concentrates at multiples of N/r = {N}/{r} = {N // r}:")
    print(f"    {'basis k':>8}{'probability':>14}")
    peaks = sorted(range(N), key=lambda k: -probs[k])[:r]
    for k in sorted(peaks):
        print(f"    {k:>8}{probs[k]:>14.4f}   <- peak")
    detected = [k for k in range(N) if probs[k] > 0.5 / r]
    print(f"\n  peaks at {sorted(detected)} = multiples of {N // r}; reading the spacing recovers r = {r}")

    # phase estimation cameo
    print(f"\n  Phase estimation (inverse QFT reading a phase off an eigenstate):")
    for phi in [0.25, 0.125, 0.3]:
        est, m = phase_estimation(phi, 6)
        print(f"    true phi = {phi:.3f}  ->  estimate {est:.4f}  (integer readout {m}/64)")

    _svg(os.path.join(outdir, "qft.svg"), probs, N // r)
    print(f"\n  wrote {os.path.join(outdir, 'qft.svg')}")


def _svg(path, probs, spacing, width=760, height=380):
    N = len(probs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'QFT output spectrum: probability peaks at multiples of N/r (period finding)</text>',
    ]
    ox, oy, ow, oh = 50, 50, width - 90, height - 100
    pmax = max(probs) * 1.1

    def bx(k):
        return ox + ow * k / N

    bw = ow / N * 0.8
    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for k in range(N):
        h = oh * probs[k] / pmax
        peak = (k % spacing == 1 % spacing) if False else False
        color = "#ffd43b" if probs[k] > 0.5 * max(probs) else "#4dabf7"
        parts.append(f'<rect x="{bx(k):.1f}" y="{oy+oh-h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'fill="{color}"/>')
    # spacing guides
    kk = 0
    while kk < N:
        parts.append(f'<line x1="{bx(kk):.1f}" y1="{oy}" x2="{bx(kk):.1f}" y2="{oy+oh}" '
                     f'stroke="#06d6a0" stroke-width="0.7" stroke-dasharray="3 3"/>')
        kk += spacing
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">output basis state k (green lines every N/r = {spacing})</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
