"""Demo: Monte Carlo Tree Search picking tic-tac-toe moves, visit counts concentrating on the best.

Shows MCTS on two tic-tac-toe positions -- one with an immediate winning move, one requiring a block --
printing how the root visit counts pile up on the correct move, and how agreement with the minimax
optimum grows with the search budget. Draws the root-move visit distribution as a bar chart.

    python examples/mcts_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcts import TicTacToe, minimax_best_moves, uct_search  # noqa: E402
import mcts as _m


def _root_visits(game, state, iterations, seed):
    """Run UCT and return {move: (visits, mean_value)} for the root children."""
    rng = _m._RNG(seed)
    root = _m._Node(state, game)
    for _ in range(iterations):
        node = root
        while not node.untried and node.children:
            node = _m._best_uct_child(node, 1.4142)
        if node.untried:
            move = node.untried.pop(int(rng.random() * len(node.untried)))
            child = _m._Node(game.apply(node.state, move), game, parent=node, move=move)
            node.children.append(child)
            node = child
        reward = _m._rollout(game, node.state, rng)
        _m._backpropagate(node, reward, game)
    return {ch.move: (ch.visits, ch.value / ch.visits if ch.visits else 0.0) for ch in root.children}


def _show_board(board):
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(3):
        print("      " + " ".join(sym[board[r * 3 + c]] for c in range(3)))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    game = TicTacToe()
    print("Monte Carlo Tree Search: self-play rollouts guided by the UCT bandit rule\n")

    # position 1: X to move, immediate win at cell 2
    board = (1, 1, 0, 2, 2, 0, 0, 0, 0)
    state = (board, 1)
    print("  Position 1 (X to move, X = player 1):")
    _show_board(board)
    best = minimax_best_moves(game, state)
    print(f"  minimax-optimal move(s): {best}  (cell 2 completes the top row)\n")

    visits = _root_visits(game, state, iterations=2000, seed=1)
    print(f"  MCTS root visit counts after 2000 iterations:")
    print(f"    {'cell':>5}{'visits':>9}{'win rate':>10}")
    for move in sorted(visits):
        v, q = visits[move]
        mark = "  <- chosen" if move == max(visits, key=lambda k: visits[k][0]) else ""
        print(f"    {move:>5}{v:>9}{q:>10.3f}{mark}")

    # convergence with budget
    print(f"\n  Fraction of correct moves vs search budget (random midgame positions):")
    rng = _m._RNG(7)
    positions = []
    for _ in range(30):
        s = game.initial()
        for _ in range(int(rng.random() * 4)):
            if game.is_terminal(s):
                break
            mv = game.legal_moves(s)
            s = game.apply(s, mv[int(rng.random() * len(mv))])
        if not game.is_terminal(s):
            positions.append(s)
    print(f"    {'iterations':>12}{'agreement':>12}")
    for iters in [50, 200, 1000, 5000]:
        agree = 0
        for i, s in enumerate(positions):
            bestset = set(minimax_best_moves(game, s))
            if uct_search(game, s, iterations=iters, seed=i) in bestset:
                agree += 1
        print(f"    {iters:>12}{agree / len(positions):>11.0%}")

    print(f"\n  More rollouts -> more agreement with perfect play. The visit count, not the raw")
    print(f"  win rate, is the output: a good move is explored both often AND deeply.")

    _svg(os.path.join(outdir, "mcts.svg"), visits, best)
    print(f"\n  wrote {os.path.join(outdir, 'mcts.svg')}")


def _svg(path, visits, best, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'MCTS root visit counts: the search concentrates on the winning move (cell 2)</text>',
    ]
    ox, oy, ow, oh = 55, 55, width - 100, height - 110
    moves = sorted(visits)
    vmax = max(v for v, _ in visits.values()) * 1.1
    bw = ow / len(moves) * 0.6
    gap = ow / len(moves)
    bestset = set(best)

    def by(v):
        return oy + oh * (1 - v / vmax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for i, mv in enumerate(moves):
        v, q = visits[mv]
        x = ox + i * gap + (gap - bw) / 2
        color = "#06d6a0" if mv in bestset else "#4dabf7"
        parts.append(f'<rect x="{x:.1f}" y="{by(v):.1f}" width="{bw:.1f}" height="{oy+oh-by(v):.1f}" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{oy+oh+14:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">cell {mv}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{by(v)-4:.1f}" fill="#e6edf3" font-size="9" '
                     f'text-anchor="middle">{v}</text>')
    parts.append(f'<text x="{ox+ow-150}" y="{oy+12}" fill="#06d6a0" font-size="10">'
                 f'green = minimax-optimal</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
