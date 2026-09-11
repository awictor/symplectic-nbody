"""Demo: LRU and LFU caching policies.

Traces an LRU cache evicting in least-recently-used order, then compares LRU and LFU hit rates
across three workload shapes -- uniform-random, skewed (a few hot keys), and a repeating loop --
showing that the right eviction policy depends on the access pattern.

    python examples/lru_cache_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lru_cache import LRUCache, LFUCache  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("LRU / LFU caching: which item to evict when memory is full\n")

    # --- trace an LRU cache ---
    print("  LRU trace (capacity 3), MRU-to-LRU order after each op:")
    c = LRUCache(3)
    script = [("put", "A"), ("put", "B"), ("put", "C"), ("get", "A"),
              ("put", "D"), ("get", "C"), ("put", "E")]
    for op, k in script:
        if op == "put":
            c.put(k, k)
            note = f"put {k}"
        else:
            hit = c.get(k) is not None
            note = f"get {k} ({'hit' if hit else 'miss'})"
        print(f"    {note:<12} -> [{', '.join(map(str, c.keys_mru_to_lru()))}]")
    print("    (D's insertion evicted B, the least-recently-used; E evicted A)\n")

    # --- workload comparison ---
    state = 9

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    cap = 10
    n_keys = 200
    n_ops = 6000

    def run(workload):
        lru = LRUCache(cap)
        lfu = LFUCache(cap)
        for _ in range(n_ops):
            k = workload()
            if lru.get(k) is None:
                lru.put(k, k)
            if lfu.get(k) is None:
                lfu.put(k, k)
        return lru.hit_rate(), lfu.hit_rate()

    loop_state = [0]

    def uniform():
        return int(rng() * n_keys)

    def skewed():
        return int(n_keys * (rng() ** 3))      # a few hot keys dominate

    def looping():
        loop_state[0] = (loop_state[0] + 1) % (cap + 5)   # sweep just past capacity
        return loop_state[0]

    print("  Hit rate by workload (cache holds 10 of 200 keys):")
    results = []
    for name, wl in [("uniform random", uniform), ("skewed (hot keys)", skewed),
                     ("looping sweep", looping)]:
        lru_hr, lfu_hr = run(wl)
        results.append((name, lru_hr, lfu_hr))
        print(f"    {name:<18}: LRU {lru_hr:5.1%}   LFU {lfu_hr:5.1%}")

    print("\n  No policy wins everywhere: LFU shines when a few keys are truly hot (it remembers")
    print("  popularity), LRU adapts faster to shifting working sets. Both do get/put in O(1) --")
    print("  LRU via a hash map plus a recency-ordered doubly-linked list, LFU via frequency buckets.")

    _svg(os.path.join(outdir, "lru_cache.svg"), results)
    print(f"\n  wrote {os.path.join(outdir, 'lru_cache.svg')}")


def _svg(path, results, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Cache hit rate: LRU vs LFU by workload</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the best eviction policy depends on the access pattern (cache holds 10 of 200 '
        f'keys)</text>',
    ]
    y0, y1 = height - 55, 75
    plot_h = y0 - y1
    group_w = (width - 90) / len(results)
    bw = group_w * 0.3
    parts.append(f'<line x1="45" y1="{y0}" x2="{width-30}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="45" y1="{y0}" x2="45" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    for frac in (0.25, 0.5, 0.75, 1.0):
        yy = y0 - frac * plot_h
        parts.append(f'<line x1="45" y1="{yy:.1f}" x2="{width-30}" y2="{yy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="40" y="{yy+4:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{int(frac*100)}%</text>')
    for i, (name, lru_hr, lfu_hr) in enumerate(results):
        cx = 70 + i * group_w
        # LRU bar
        h1 = lru_hr * plot_h
        parts.append(f'<rect x="{cx:.1f}" y="{y0-h1:.1f}" width="{bw:.1f}" height="{h1:.1f}" '
                     f'fill="#4dabf7"/>')
        parts.append(f'<text x="{cx+bw/2:.1f}" y="{y0-h1-4:.1f}" fill="#4dabf7" font-size="9" '
                     f'text-anchor="middle">{lru_hr:.0%}</text>')
        # LFU bar
        h2 = lfu_hr * plot_h
        parts.append(f'<rect x="{cx+bw+6:.1f}" y="{y0-h2:.1f}" width="{bw:.1f}" height="{h2:.1f}" '
                     f'fill="#ff922b"/>')
        parts.append(f'<text x="{cx+bw+6+bw/2:.1f}" y="{y0-h2-4:.1f}" fill="#ff922b" font-size="9" '
                     f'text-anchor="middle">{lfu_hr:.0%}</text>')
        parts.append(f'<text x="{cx+bw:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{name}</text>')
    parts.append(f'<rect x="{width-160}" y="60" width="11" height="7" fill="#4dabf7"/>')
    parts.append(f'<text x="{width-145}" y="67" fill="#8b949e" font-size="10">LRU</text>')
    parts.append(f'<rect x="{width-100}" y="60" width="11" height="7" fill="#ff922b"/>')
    parts.append(f'<text x="{width-85}" y="67" fill="#8b949e" font-size="10">LFU</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
