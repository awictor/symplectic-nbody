"""Demo: the matched filter pulling a known pulse out of heavy noise, and delay estimation by correlation.

Buries a chirp-like pulse in noise so it is invisible to the eye, then recovers its location with a
matched filter (correlation against the template), and separately estimates the delay between a signal
and a delayed echo. Draws the noisy signal and the matched-filter response with its sharp detection
peak.

    python examples/cross_correlation_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cross_correlation import matched_filter, detect_pulse, estimate_delay, autocorrelation  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cross-correlation & matched filter: finding a signal, and where it hides, in noise\n")

    rng = _lcg(2024)
    # a short chirp template
    L = 12
    template = [math.sin(2 * math.pi * (0.1 + 0.02 * j) * j) for j in range(L)]

    n = 120
    true_start = 70
    noise_amp = 1.5
    signal = [noise_amp * (rng() - 0.5) for _ in range(n)]
    for j in range(L):
        signal[true_start + j] += template[j]

    # signal-to-noise: pulse peak vs noise std
    import statistics
    noise_std = statistics.pstdev(signal[:60])
    print(f"  a length-{L} chirp buried at index {true_start} in noise (amp {noise_amp})")
    print(f"  the pulse peak ({max(template):.2f}) is comparable to the noise std ({noise_std:.2f}) --")
    print(f"  invisible to a raw threshold.\n")

    detected = detect_pulse(signal, template)
    raw_argmax = max(range(n), key=lambda i: signal[i])
    print(f"  raw signal argmax:       index {raw_argmax}  (a noise spike, wrong)")
    print(f"  matched-filter detection: index {detected}  (true start {true_start})")
    print(f"  the matched filter is the maximum-SNR detector -- it integrates the whole template.\n")

    # delay estimation between a signal and its echo
    base = [math.sin(2 * math.pi * i / 15) * math.exp(-((i - 20) ** 2) / 100) for i in range(60)]
    delay = 17
    echo = [0.0] * delay + base[:60 - delay]
    est = estimate_delay(base, echo)
    print(f"  delay estimation (sonar-style): a pulse and its echo delayed by {delay} samples")
    print(f"  cross-correlation peak gives delay = {est}  (true {delay})")

    _svg(os.path.join(outdir, "cross_correlation.svg"), signal, matched_filter(signal, template),
         true_start, detected)
    print(f"\n  wrote {os.path.join(outdir, 'cross_correlation.svg')}")


def _svg(path, signal, mf, true_start, detected, width=760, height=420):
    lags, resp = mf
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="24" fill="#e6edf3" font-size="14">'
        f'Noisy signal (top): pulse invisible. Matched-filter response (bottom): sharp detection peak</text>',
    ]
    n = len(signal)
    ox, ow = 45, width - 80

    # top: noisy signal
    ty, th = 45, 140
    smax = max(abs(v) for v in signal)

    def sxs(i):
        return ox + ow * i / (n - 1)

    def sys(v):
        return ty + th / 2 - th / 2 * v / smax

    parts.append(f'<rect x="{ox}" y="{ty}" width="{ow}" height="{th}" fill="none" stroke="#30363d"/>')
    pts = " ".join(f"{sxs(i):.1f},{sys(signal[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#8b949e" stroke-width="1"/>')
    parts.append(f'<line x1="{sxs(true_start):.1f}" y1="{ty}" x2="{sxs(true_start):.1f}" y2="{ty+th}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{sxs(true_start)+4:.0f}" y="{ty+12}" fill="#06d6a0" font-size="9">'
                 f'true pulse start</text>')

    # bottom: matched filter response (only positive lags mapped to signal indices)
    by, bh = 240, 140
    rmax = max(abs(v) for v in resp)
    # response index corresponding to signal position = lag; show lags 0..n-1
    zero = lags.index(0)
    seg = resp[zero:zero + n]

    def bxs(i):
        return ox + ow * i / (n - 1)

    def bys(v):
        return by + bh / 2 - bh / 2 * v / rmax

    parts.append(f'<rect x="{ox}" y="{by}" width="{ow}" height="{bh}" fill="none" stroke="#30363d"/>')
    pts = " ".join(f"{bxs(i):.1f},{bys(seg[i]):.1f}" for i in range(len(seg)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.5"/>')
    # mark detection
    di = detected
    if 0 <= di < len(seg):
        parts.append(f'<circle cx="{bxs(di):.1f}" cy="{bys(seg[di]):.1f}" r="5" fill="#ffd43b"/>')
        parts.append(f'<text x="{bxs(di)+6:.0f}" y="{bys(seg[di]):.0f}" fill="#ffd43b" font-size="9">'
                     f'detection @ {di}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{by+bh+20:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">sample index -- the matched filter integrates the template to '
                 f'a clean peak</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
