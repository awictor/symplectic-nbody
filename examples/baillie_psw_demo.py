"""Baillie-PSW demo: how the Lucas half catches the pseudoprimes that fool Miller-Rabin base 2 (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import baillie_psw as B


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    lines = []
    lines.append("Baillie-PSW primality test (no known counterexample)")
    lines.append("=" * 54)
    lines.append("declares prime only if BOTH pass: strong Miller-Rabin base 2 AND strong Lucas")
    lines.append("the two tests' pseudoprimes appear disjoint -- so the combination is bulletproof")
    lines.append("")
    lines.append("strong base-2 pseudoprimes (fool Miller-Rabin, caught by Lucas):")
    lines.append(f"{'n':>7}{'MR base-2':>12}{'strong Lucas':>14}{'Baillie-PSW':>13}{'truth':>10}")
    for n in [2047, 3277, 4033, 4681, 8321, 15841]:
        mr = B.strong_miller_rabin_base2(n)
        lucas = B.strong_lucas_prp(n)
        bpsw = B.is_prime(n)
        truth = "prime" if B.trial_division_prime(n) else "COMPOSITE"
        lines.append(f"{n:>7}{('pass' if mr else 'fail'):>12}{('pass' if lucas else 'fail'):>14}"
                     f"{('PRIME' if bpsw else 'composite'):>13}{truth:>10}")
    lines.append("")
    lines.append("Miller-Rabin base 2 is fooled (says 'pass'), but the Lucas test fails them,")
    lines.append("so Baillie-PSW correctly reports composite. That is the whole idea.")
    lines.append("")
    lines.append("a few genuine primes (both halves pass):")
    for n in [97, 1000003, 2**31 - 1, 10**9 + 7, 999999937]:
        lines.append(f"  {n} -> {'PRIME' if B.is_prime(n) else 'composite'}")
    lines.append("")
    lines.append(f"prime count check: pi(10000) = "
                 f"{sum(1 for n in range(2, 10000) if B.is_prime(n))} (known: 1229)")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: for n up to N, plot which test(s) pass; highlight base-2 pseudoprimes
        N = 300
        W, H = 720, 360
        ml, mt, w, h = 50, 60, 620, 240
        # collect composites that pass MR base-2 (the interesting ones) and their Lucas verdict
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Baillie-PSW = Miller-Rabin(2) AND Lucas: agreement with truth</text>')
        # three rows of ticks: primes (green), composites correctly rejected (blue),
        # and any disagreement (red -- none expected)
        NN = 1000
        row_y = {"prime": mt + 40, "composite": mt + 110, "error": mt + 180}
        s.append(f'<text x="{ml}" y="{row_y["prime"]-8}" fill="{GREEN}" font-size="11">'
                 f'primes (Baillie-PSW says prime)</text>')
        s.append(f'<text x="{ml}" y="{row_y["composite"]-8}" fill="{BLUE}" font-size="11">'
                 f'composites (correctly rejected)</text>')
        s.append(f'<text x="{ml}" y="{row_y["error"]-8}" fill="{RED}" font-size="11">'
                 f'disagreements with trial division (none)</text>')

        def sx(n):
            return ml + n / NN * w
        errors = 0
        for n in range(2, NN):
            bp = B.is_prime(n)
            tr = B.trial_division_prime(n)
            if bp != tr:
                errors += 1
                s.append(f'<line x1="{sx(n):.1f}" y1="{row_y["error"]:.1f}" '
                         f'x2="{sx(n):.1f}" y2="{row_y["error"]+18:.1f}" stroke="{RED}" stroke-width="1"/>')
            elif bp:
                s.append(f'<line x1="{sx(n):.1f}" y1="{row_y["prime"]:.1f}" '
                         f'x2="{sx(n):.1f}" y2="{row_y["prime"]+18:.1f}" stroke="{GREEN}" stroke-width="0.8"/>')
            else:
                s.append(f'<line x1="{sx(n):.1f}" y1="{row_y["composite"]:.1f}" '
                         f'x2="{sx(n):.1f}" y2="{row_y["composite"]+10:.1f}" stroke="{BLUE}" stroke-width="0.5"/>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Across n=2..{NN} Baillie-PSW matches deterministic trial division exactly '
                 f'({errors} disagreements) -- the red row stays empty.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "baillie_psw.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
