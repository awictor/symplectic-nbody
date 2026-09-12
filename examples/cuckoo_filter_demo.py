"""Demo: a cuckoo filter -- Bloom-like membership that also supports deletion.

Adds items, confirms no false negatives, measures the false-positive rate at different fingerprint
sizes, and demonstrates deletion (which Bloom filters cannot do). Draws the false-positive rate
falling as fingerprint bits grow.

    python examples/cuckoo_filter_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuckoo_filter import CuckooFilter  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cuckoo filter: approximate membership with deletion\n")

    cf = CuckooFilter(capacity=8192, bucket_size=4, fingerprint_bits=16, seed=1)
    items = [f"user-{i}" for i in range(2000)]
    for x in items:
        cf.add(x)
    print(f"  added {len(items)} items, load factor {cf.load_factor():.2f}")
    print(f"  no false negatives: {all(x in cf for x in items)}")

    absent = [f"stranger-{i}" for i in range(50000)]
    fp = sum(1 for x in absent if x in cf)
    print(f"  false positives on 50000 absent items: {fp} (rate {fp/50000:.5f})")

    print("\n  Deletion (impossible with a Bloom filter):")
    print(f"    'user-500' present: {'user-500' in cf}")
    cf.delete("user-500")
    print(f"    after delete:        {'user-500' in cf}")
    print(f"    'user-501' unaffected: {'user-501' in cf}")

    # false-positive rate vs fingerprint bits
    print("\n  False-positive rate shrinks as the fingerprint grows:")
    results = []
    for bits in [4, 6, 8, 10, 12, 16]:
        c = CuckooFilter(capacity=4096, bucket_size=4, fingerprint_bits=bits, seed=2)
        for i in range(1000):
            c.add(f"m{i}")
        f = sum(1 for i in range(50000) if f"z{i}" in c)
        rate = f / 50000
        results.append((bits, rate))
        print(f"    {bits:2d} bits: false-positive rate {rate:.5f}")

    print("\n  Each item stores a small fingerprint in one of two candidate buckets, the second found")
    print("  by XORing the first with hash(fingerprint) -- so either bucket recovers the other from")
    print("  the fingerprint alone. Full buckets kick out a resident to its alternate (the 'cuckoo').")

    _svg(os.path.join(outdir, "cuckoo_filter.svg"), results)
    print(f"\n  wrote {os.path.join(outdir, 'cuckoo_filter.svg')}")


def _svg(path, results, width=760, height=420):
    import math
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    # y = log10(rate); guard against 0
    def logr(r):
        return math.log10(max(r, 1e-6))

    bits = [b for b, _ in results]
    xmin, xmax = min(bits), max(bits)
    ys = [logr(r) for _, r in results]
    ymin, ymax = min(ys), max(ys)

    def px(b):
        return m_left + (b - xmin) / (xmax - xmin) * pw

    def py(ly):
        return m_top + ph - (ly - ymin) / (ymax - ymin + 1e-9) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Cuckoo filter: false-positive rate vs fingerprint bits</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'more fingerprint bits -> exponentially fewer false positives (log scale); no false negatives ever</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    pts = " ".join(f"{px(b):.1f},{py(logr(r)):.1f}" for b, r in results)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for b, r in results:
        parts.append(f'<circle cx="{px(b):.1f}" cy="{py(logr(r)):.1f}" r="4" fill="#4dabf7"/>')
        parts.append(f'<text x="{px(b):.0f}" y="{m_top+ph+18:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{b}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">fingerprint bits</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
