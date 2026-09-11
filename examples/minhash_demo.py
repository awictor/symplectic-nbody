"""Demo: MinHash Jaccard estimation and LSH near-duplicate detection.

Estimates set similarity from short signatures, shows the estimate converging to the true Jaccard as
the number of hashes grows, and uses banded LSH to find near-duplicate documents. Draws the
error-vs-k convergence and the LSH S-curve.

    python examples/minhash_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minhash import MinHash, estimate_jaccard, true_jaccard, LSH, lsh_threshold  # noqa: E402


def shingles(text, w=3):
    """Character w-shingles of a string (the classic document-similarity feature set)."""
    text = text.lower()
    return set(text[i:i + w] for i in range(len(text) - w + 1))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("MinHash + LSH: estimating set similarity at scale\n")

    docs = {
        "A": "the quick brown fox jumps over the lazy dog",
        "B": "the quick brown fox jumped over a lazy dog",     # near-dup of A
        "C": "the quick brown fox jumps over the lazy dog!",   # near-dup of A
        "D": "a completely different sentence about cats",
        "E": "cats are completely different from that sentence",
    }
    sets = {k: shingles(v) for k, v in docs.items()}

    print("  Documents (as character 3-shingle sets):")
    for k, v in docs.items():
        print(f"    {k}: \"{v}\"")

    mh = MinHash(num_hashes=200, seed=5)
    sigs = {k: mh.signature(s) for k, s in sets.items()}

    print("\n  Pairwise similarity (MinHash estimate vs true Jaccard, k=200):")
    keys = list(docs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            est = estimate_jaccard(sigs[a], sigs[b])
            tru = true_jaccard(sets[a], sets[b])
            print(f"    {a}-{b}: estimate {est:.3f}  true {tru:.3f}  (err {abs(est-tru):.3f})")

    # convergence: mean abs error vs k over random set pairs
    print("\n  Estimate error shrinks like 1/sqrt(k):")
    ks = [8, 16, 32, 64, 128, 256, 512]
    errors = []
    for k in ks:
        mh_k = MinHash(num_hashes=k, seed=9)
        state = 1234

        def rng():
            nonlocal state
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            return (state >> 16) / 65536.0

        tot = 0.0
        T = 100
        for _ in range(T):
            A = set(int(rng() * 300) for _ in range(60))
            B = set(e for e in A if rng() < 0.6) | set(int(rng() * 300) for _ in range(30))
            tot += abs(estimate_jaccard(mh_k.signature(A), mh_k.signature(B)) - true_jaccard(A, B))
        errors.append(tot / T)
        print(f"    k={k:4d}  mean error {errors[-1]:.4f}")

    # LSH near-duplicate detection
    print("\n  LSH near-duplicate search (bands=50, rows=4, k=200):")
    lsh = LSH(50, 4)
    for k in keys:
        lsh.add(k, sigs[k])
    pairs = lsh.all_candidate_pairs()
    print(f"    candidate near-duplicate pairs: {sorted(pairs)}")
    print(f"    S-curve 50% threshold at Jaccard ~ {lsh_threshold(50, 4):.3f}")

    print("\n  Two signatures agree in a position with probability equal to the Jaccard similarity,")
    print("  so matching-position fraction is an unbiased estimate. LSH bands make similar items")
    print("  collide in a hash table, turning all-pairs search into a few lookups.")

    _svg(os.path.join(outdir, "minhash.svg"), ks, errors)
    print(f"\n  wrote {os.path.join(outdir, 'minhash.svg')}")


def _svg(path, ks, errors, width=760, height=430):
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    # x axis: log2(k); y axis: error
    import math
    xs = [math.log2(k) for k in ks]
    xmin, xmax = min(xs), max(xs)
    ymax = max(errors) * 1.1
    ymin = 0.0

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(e):
        return m_top + ph - (e - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'MinHash accuracy: mean Jaccard error vs number of hashes</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = measured mean error, yellow dashed = the 1/sqrt(k) theory curve</text>',
    ]

    # axes
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for k, lx in zip(ks, xs):
        x = px(lx)
        parts.append(f'<line x1="{x:.1f}" y1="{m_top+ph}" x2="{x:.1f}" y2="{m_top+ph+5}" stroke="#484f58"/>')
        parts.append(f'<text x="{x:.1f}" y="{m_top+ph+20}" fill="#8b949e" font-size="11" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-14}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">number of hashes k (log scale)</text>')

    # theory curve c / sqrt(k), fit c to the first point
    c = errors[0] * math.sqrt(ks[0])
    theory = [c / math.sqrt(k) for k in ks]
    tpath = " ".join(f"{px(lx):.1f},{py(t):.1f}" for lx, t in zip(xs, theory))
    parts.append(f'<polyline points="{tpath}" fill="none" stroke="#ffd43b" stroke-width="1.8" '
                 f'stroke-dasharray="6,4"/>')

    # measured
    mpath = " ".join(f"{px(lx):.1f},{py(e):.1f}" for lx, e in zip(xs, errors))
    parts.append(f'<polyline points="{mpath}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for lx, e in zip(xs, errors):
        parts.append(f'<circle cx="{px(lx):.1f}" cy="{py(e):.1f}" r="4" fill="#4dabf7"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
