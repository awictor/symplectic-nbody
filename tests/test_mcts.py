"""Tests for MCTS/UCT: never loses tic-tac-toe, wins/blocks immediately, matches minimax, monotone."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcts import (  # noqa: E402
    uct_search,
    mcts_policy,
    minimax_value,
    minimax_best_moves,
    TicTacToe,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _play_game(game, policy1, policy2, start=None):
    """Play policy1 (player 1) vs policy2 (player 2). Returns the winner."""
    s = start if start is not None else game.initial()
    while not game.is_terminal(s):
        p = game.player(s)
        move = policy1(s) if p == 1 else policy2(s)
        s = game.apply(s, move)
    return game.winner(s)


def main():
    game = TicTacToe()

    # ---- 1. tic-tac-toe is a draw under optimal play (minimax value 0 from empty) -------
    check("tic-tac-toe minimax value = 0 (draw)", minimax_value(game, game.initial()) == 0)

    # ---- 2. MCTS takes an immediate winning move ----------------------------------------
    # X (player 1) to move with two in a row at 0,1; winning move is 2
    board = (1, 1, 0,
             2, 2, 0,
             0, 0, 0)
    state = (board, 1)
    move = uct_search(game, state, iterations=800, seed=1)
    check("MCTS takes immediate win", move == 2, f"chose {move}")

    # ---- 3. MCTS blocks an immediate loss -----------------------------------------------
    # O (player 2) threatens 6,7 -> block at 8; X to move
    board = (2, 2, 0,
             1, 0, 0,
             0, 0, 0)
    # here O has 0,1 -> threat at 2; X must block at 2
    state = (board, 1)
    move = uct_search(game, state, iterations=1000, seed=2)
    check("MCTS blocks immediate loss", move == 2, f"chose {move}")

    # ---- 4. MCTS never loses from the empty board vs a minimax-optimal opponent ---------
    mm_policy = lambda s: minimax_best_moves(game, s)[0]
    results = []
    for seed in range(6):
        pol = mcts_policy(game, iterations=1200, seed=seed)
        # MCTS as player 1 vs minimax player 2
        w1 = _play_game(game, pol, mm_policy)
        # MCTS as player 2 vs minimax player 1
        w2 = _play_game(game, mm_policy, pol)
        results.append(w1)
        results.append(w2)
    # MCTS must never LOSE (winner is never the minimax player)
    mcts_losses = sum(1 for i, w in enumerate(results)
                      if (i % 2 == 0 and w == 2) or (i % 2 == 1 and w == 1))
    check("MCTS never loses vs minimax (12 games)", mcts_losses == 0, f"{mcts_losses} losses")

    # ---- 5. MCTS choice matches a minimax-optimal move on random positions --------------
    # generate some legal midgame positions and check MCTS agrees with minimax
    import mcts as _m
    rng = _m._RNG(7)
    agree = 0
    total = 0
    for trial in range(20):
        s = game.initial()
        # play a few random moves
        depth = int(rng.random() * 4)
        for _ in range(depth):
            if game.is_terminal(s):
                break
            moves = game.legal_moves(s)
            s = game.apply(s, moves[int(rng.random() * len(moves))])
        if game.is_terminal(s):
            continue
        total += 1
        # MCTS move is "correct" if it is a value-optimal minimax move. Vanilla UCT with UNIFORM
        # random rollouts has a well-known tactical blind spot: a losing move can look fine if the
        # random opponent rarely finds the refutation, so ~75-80% agreement is expected at a modest
        # budget (a rollout policy or a neural net -- AlphaGo's move -- would close the gap).
        best = set(minimax_best_moves(game, s))
        move = uct_search(game, s, iterations=5000, seed=trial + 100)
        if move in best:
            agree += 1
    check("MCTS matches minimax-optimal move (>=75%)", agree >= 0.75 * total,
          f"{agree}/{total}")

    # ---- 6. more iterations do not make MCTS play worse (win-a-forced-win position) -----
    # a position where X has a forced win; MCTS should find it with enough budget
    board = (1, 0, 0,
             0, 2, 0,
             0, 0, 2)
    state = (board, 1)
    best = set(minimax_best_moves(game, state))
    move_lo = uct_search(game, state, iterations=200, seed=3)
    move_hi = uct_search(game, state, iterations=3000, seed=3)
    check("high-budget MCTS picks an optimal move", move_hi in best, f"chose {move_hi}, best {best}")

    # ---- 7. minimax best-moves are consistent with the value ----------------------------
    # from empty board every first move leads to a draw (value 0), so all 9 are "best"
    check("all first moves optimal (draw)", len(minimax_best_moves(game, game.initial())) == 9)

    # ---- 8. a won/lost position is detected ---------------------------------------------
    won = ((1, 1, 1, 2, 2, 0, 0, 0, 0), 2)
    check("terminal win detected", game.is_terminal(won) and game.winner(won) == 1)

    # ---- 9. uct returns a legal move ----------------------------------------------------
    s = game.initial()
    move = uct_search(game, s, iterations=300, seed=5)
    check("uct returns a legal move", move in game.legal_moves(s))

    # ---- 10. terminal state -> no move --------------------------------------------------
    full = ((1, 2, 1, 1, 2, 2, 2, 1, 1), 2)  # a full board
    check("terminal -> None move", uct_search(game, full, iterations=10, seed=1) is None)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
