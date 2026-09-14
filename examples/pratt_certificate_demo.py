"""Pratt certificate demo: the recursive primality proof tree for a prime, drawn out (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pratt_certificate as P


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
GREEN = "#06d6a0"
BLUE = "#4dabf7"
YELLOW = "#ffd43b"


def _render_tree(cert, indent=0, lines=None):
    if lines is None:
        lines = []
    num, a, sub = cert
    if num == 2:
        lines.append("  " * indent + "2  (base case)")
    else:
        facts = ", ".join(str(q) for q, _ in sub)
        lines.append("  " * indent + f"{num}: witness a={a}, {num}-1 factors into [{facts}]")
        for q, cq in sub:
            _render_tree(cq, indent + 1, lines)
    return lines


def main(outdir=None):
    n = 997
    cert = P.build(n)

    out = []
    out.append("Pratt certificate: a short, checkable proof of primality")
    out.append("=" * 58)
    out.append("Lucas test: n is prime iff some witness a has a^(n-1)=1 and")
    out.append("a^((n-1)/q)!=1 for every prime q | n-1 -- and each such q is itself certified.")
    out.append("")
    out.append(f"certificate for {n} ({P.certificate_size(cert)} nodes):")
    out += ["  " + ln for ln in _render_tree(cert)]
    out.append("")
    out.append(f"verifies? {P.verify(cert)}")
    out.append("")
    out.append("certificate sizes (nodes) grow polynomially in the number of digits:")
    out.append(f"{'prime':>10}{'nodes':>8}{'witness':>9}")
    for p in (7, 101, 997, 7919, 104729, 1299709):
        c = P.build(p)
        out.append(f"{p:>10}{P.certificate_size(c):>8}{c[1]:>9}")
    out.append("")
    out.append("This is why PRIMES is in NP: the certificate is short and anyone can re-check it")
    out.append("with a handful of modular exponentiations -- no need to redo the primality search.")

    text = "\n".join(out)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: draw the certificate tree as a node-link diagram
        W, H = 700, 420

        # assign positions by DFS with a level-based layout
        positions = {}
        node_list = []

        def layout(cert, depth, xcounter):
            num, a, sub = cert
            children_x = []
            for q, cq in sub:
                cx = layout(cq, depth + 1, xcounter)
                children_x.append(cx)
            if children_x:
                x = sum(children_x) / len(children_x)
            else:
                x = xcounter[0]
                xcounter[0] += 1
            idx = len(node_list)
            node_list.append((num, a, depth, x))
            positions[id(cert)] = (x, depth, num, a)
            return x

        xc = [0]
        layout(cert, 0, xc)
        maxdepth = max(d for _, _, d, _ in node_list)
        maxx = max(x for _, _, _, x in node_list) or 1
        ml, mt = 50, 70
        w = W - 2 * ml
        vgap = (H - 130) / (maxdepth + 1)

        def sx(x):
            return ml + (x / maxx) * w if maxx > 0 else W / 2

        def sy(d):
            return mt + d * vgap

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Pratt certificate tree for {n}: every factor recursively proved prime</text>')

        # edges: re-walk to draw parent-child links
        def draw_edges(cert):
            num, a, sub = cert
            px, pd, _, _ = positions[id(cert)]
            for q, cq in sub:
                cx, cd, _, _ = positions[id(cq)]
                s.append(f'<line x1="{sx(px):.1f}" y1="{sy(pd):.1f}" x2="{sx(cx):.1f}" '
                         f'y2="{sy(cd):.1f}" stroke="{GRAY}" stroke-width="1"/>')
                draw_edges(cq)
        draw_edges(cert)

        def draw_nodes(cert):
            num, a, sub = cert
            x, d, _, _ = positions[id(cert)]
            col = YELLOW if num == 2 else (GREEN if not sub else BLUE)
            s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(d):.1f}" r="15" fill="{col}" '
                     f'stroke="{BG}" stroke-width="2"/>')
            s.append(f'<text x="{sx(x):.1f}" y="{sy(d)+4:.1f}" fill="{BG}" font-size="10" '
                     f'text-anchor="middle" font-weight="bold">{num}</text>')
            for q, cq in sub:
                draw_nodes(cq)
        draw_nodes(cert)
        s.append(f'<text x="{ml}" y="{H-16}" fill="{GRAY}" font-size="10">'
                 f'Each node n carries a witness and the factors of n-1; its children prove those '
                 f'factors prime, bottoming out at 2 (yellow).</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "pratt_certificate.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
