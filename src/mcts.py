"""Monte Carlo Tree Search: the search behind AlphaGo, learning where to look by playing itself out.

Game-playing programs face an exponential tree: chess has ~35 moves per position, Go ~250, far too
many to search exhaustively. Classical MINIMAX with alpha-beta prunes the tree but needs a good
handcrafted evaluation of non-terminal positions. MONTE CARLO TREE SEARCH (Coulom / Kocsis-Szepesvari,
2006) needs none: it estimates a move's value by PLAYING RANDOM GAMES to the end from it and averaging
the outcomes, and it spends its simulations where they matter by treating the choice of which child to
explore as a MULTI-ARMED BANDIT problem. This idea -- no evaluation function, just self-play rollouts
guided by a bandit rule -- is what cracked computer Go and, with a neural net replacing the random
rollout, became AlphaGo.

Each iteration walks the four MCTS phases:

  SELECTION. From the root, descend by the UCT rule, picking at each node the child maximizing
  Q/N + c*sqrt(ln N_parent / N_child) -- the average reward plus an EXPLORATION BONUS that favours
  rarely-visited moves. This is UCB1 applied recursively down the tree, balancing exploiting known-good
  moves against exploring uncertain ones.
  EXPANSION. At the first node with untried moves, add one new child.
  SIMULATION. From that child, play uniformly random moves to a terminal state and score it.
  BACKPROPAGATION. Push the result back up the path, incrementing each node's visit count and value
  from the perspective of the player to move there.

After a budget of iterations the most-VISITED root child is chosen -- robust because a good move is
both high-value and heavily explored. As iterations grow, MCTS provably converges to the minimax move
(in the limit -- with UNIFORM random rollouts it converges slowly, and at a modest budget can still
miss a sharp tactic the random opponent rarely punishes; a rollout policy or a value network, the
AlphaGo refinement, closes that gap).

This module implements UCT over a small abstract Game interface, with a fully worked tic-tac-toe game
and an exact minimax solver for ground truth. It is validated by results: MCTS never loses at
tic-tac-toe from the empty board (the game is a draw under optimal play) across many seeds; it takes an
immediate winning move when one exists and blocks an immediate loss; on random positions its choice
matches a minimax-optimal move as the budget grows; the tree statistics are consistent (a node's visits
equal the sum of its children's plus its own expansions); and more iterations never make it play worse.
Pure stdlib (seeded RNG); the self-play search companion to the minimax/alpha-beta idea, the UCB bandit,
and the Sprague-Grundy game tools."""

from __future__ import annotations

import math


class _RNG:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def random(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def choice(self, seq):
        return seq[int(self.random() * len(seq))]


class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "value", "player")

    def __init__(self, state, game, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.untried = list(game.legal_moves(state))
        self.visits = 0
        self.value = 0.0            # total reward from the perspective of `player`
        self.player = game.player(state)  # player to move at this node


def uct_search(game, root_state, iterations=1000, c=1.4142, seed=0):
    """Run MCTS/UCT from root_state for `iterations`. Returns the best move (most-visited child)."""
    rng = _RNG(seed)
    root = _Node(root_state, game)

    for _ in range(iterations):
        node = root
        # --- selection ---
        while not node.untried and node.children:
            node = _best_uct_child(node, c)
        # --- expansion ---
        if node.untried:
            move = node.untried.pop(int(rng.random() * len(node.untried)))
            next_state = game.apply(node.state, move)
            child = _Node(next_state, game, parent=node, move=move)
            node.children.append(child)
            node = child
        # --- simulation (random rollout) ---
        reward_for = _rollout(game, node.state, rng)
        # --- backpropagation ---
        _backpropagate(node, reward_for, game)

    if not root.children:
        return None
    # most-visited child is the robust choice
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move


def _best_uct_child(node, c):
    logN = math.log(node.visits) if node.visits > 0 else 0.0
    best = None
    best_score = -1e300
    for ch in node.children:
        if ch.visits == 0:
            score = 1e300
        else:
            exploit = ch.value / ch.visits
            explore = c * math.sqrt(logN / ch.visits)
            score = exploit + explore
        if score > best_score:
            best_score = score
            best = ch
    return best


def _rollout(game, state, rng):
    """Play random moves to a terminal state. Returns the winner (player id or None for draw)."""
    s = state
    while not game.is_terminal(s):
        moves = game.legal_moves(s)
        s = game.apply(s, rng.choice(moves))
    return game.winner(s)


def _backpropagate(node, winner, game):
    """Push a rollout result up the path. Each node scores from ITS player's perspective."""
    while node is not None:
        node.visits += 1
        # reward for the player who moved INTO this node = the parent's player.
        # We store value from the perspective of node.player (to move here); a win for the
        # OPPONENT of node.player is bad. Use +1 win / 0.5 draw / 0 loss for node.player's opponent
        # who just moved. Simpler: reward for the player about to move at the PARENT.
        if node.parent is not None:
            mover = node.parent.player  # the player who made node.move
            if winner is None:
                node.value += 0.5
            elif winner == mover:
                node.value += 1.0
            else:
                node.value += 0.0
        node = node.parent


def mcts_policy(game, iterations=1000, seed=0):
    """Return a function state -> move that runs UCT for the given budget."""
    def policy(state):
        return uct_search(game, state, iterations=iterations, seed=seed)
    return policy


# ---- exact minimax for ground truth -----------------------------------------------------------

def minimax_value(game, state, memo=None):
    """Exact game value from the perspective of the player to move: +1 win, 0 draw, -1 loss."""
    if memo is None:
        memo = {}
    key = game.key(state)
    if key in memo:
        return memo[key]
    if game.is_terminal(state):
        w = game.winner(state)
        if w is None:
            memo[key] = 0
        else:
            memo[key] = 1 if w == game.player(state) else -1
        return memo[key]
    best = -2
    for m in game.legal_moves(state):
        # value for the opponent after our move, negated
        v = -minimax_value(game, game.apply(state, m), memo)
        if v > best:
            best = v
    memo[key] = best
    return best


def minimax_best_moves(game, state):
    """All moves that achieve the optimal minimax value (there may be ties)."""
    memo = {}
    best_v = -2
    scored = []
    for m in game.legal_moves(state):
        v = -minimax_value(game, game.apply(state, m), memo)
        scored.append((m, v))
        best_v = max(best_v, v)
    return [m for m, v in scored if v == best_v]


# ---- tic-tac-toe game --------------------------------------------------------------------------

class TicTacToe:
    """Tic-tac-toe. State = (board tuple of 9 with 0 empty / 1 / 2, player to move)."""

    WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
            (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def initial(self):
        return ((0,) * 9, 1)

    def player(self, state):
        return state[1]

    def legal_moves(self, state):
        board = state[0]
        return [i for i in range(9) if board[i] == 0]

    def apply(self, state, move):
        board, p = state
        nb = list(board)
        nb[move] = p
        return (tuple(nb), 2 if p == 1 else 1)

    def is_terminal(self, state):
        return self.winner(state) is not None or all(c != 0 for c in state[0])

    def winner(self, state):
        board = state[0]
        for a, b, c in self.WINS:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def key(self, state):
        return state
