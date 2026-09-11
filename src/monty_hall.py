"""The Monty Hall problem: why switching doors wins.

On the game show, a car hides behind one of three doors and goats behind the other two. You
pick a door. The host -- who knows where the car is -- opens a different door, always revealing
a goat, and offers to let you switch to the remaining closed door. Should you?

The famous, counterintuitive answer is YES: switching wins with probability 2/3, staying only
1/3. The intuition: your first pick is right 1/3 of the time, and the host's forced reveal
concentrates the entire remaining 2/3 onto the one other closed door. The host is not acting at
random -- his knowledge is exactly the information that shifts the odds. (If instead the host
opened a door blindly and it happened to show a goat, switching would only be 1/2; the paradox
lives entirely in the host's knowledge.)

The problem generalizes: with N doors, one car, you pick one, and the knowing host opens k goat
doors (1 <= k <= N-2), then you may switch to one of the remaining N-k-1 closed doors. Staying
wins 1/N; switching to a specific remaining door wins

    P(switch) = (1/(N-k-1)) * (N-1)/N,

which always beats staying, and the advantage grows with N. For the classic N=3, k=1 this is
(1)(2/3) = 2/3.

This module gives the exact stay and switch probabilities for the classic and generalized game,
the informative-vs-random-host comparison, and a seeded Monte-Carlo play that confirms them.
Pure stdlib; the conditional-probability companion to the birthday and gambler's-ruin notes.
"""

from __future__ import annotations


def stay_probability(doors: int = 3) -> float:
    """Probability of winning by staying with the first pick: 1/N (the host's reveal never
    changes the chance your original door hides the car)."""
    if doors < 3:
        raise ValueError("need at least 3 doors")
    return 1.0 / doors


def switch_probability(doors: int = 3, reveals: int = 1) -> float:
    """Probability of winning by switching to one specific remaining closed door, when the
    knowing host opens `reveals` goat doors.

        P = (1/(N - reveals - 1)) * (N-1)/N.

    For N=3, reveals=1 this is 2/3."""
    if doors < 3:
        raise ValueError("need at least 3 doors")
    if not 1 <= reveals <= doors - 2:
        raise ValueError("reveals must be in 1..N-2 (host leaves your door and one other)")
    remaining = doors - reveals - 1  # other closed doors you could switch to
    return (1.0 / remaining) * (doors - 1) / doors


def switch_advantage(doors: int = 3, reveals: int = 1) -> float:
    """How many times more likely switching is than staying: P(switch)/P(stay)."""
    return switch_probability(doors, reveals) / stay_probability(doors)


def random_host_switch_probability(doors: int = 3) -> float:
    """Switch-win probability in the N=3 variant where the host opens a door AT RANDOM and it
    happens to reveal a goat (the 'Monty Fall' variant): 1/2, not 2/3 -- the paradox vanishes
    when the host has no knowledge."""
    if doors != 3:
        raise ValueError("random-host result implemented for the classic 3-door game")
    return 0.5


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def simulate(strategy: str, doors: int = 3, reveals: int = 1, trials: int = 20000,
             seed: int = 1, informed_host: bool = True) -> float:
    """Monte-Carlo win rate. `strategy` is 'stay' or 'switch'. The knowing host opens `reveals`
    goat doors among the doors you did not pick; with informed_host=False (only for the 3-door
    single-reveal game) the host opens a random other door and the trial is discarded if it
    reveals the car (the conditional 'happened to show a goat' setup). Returns win fraction."""
    rng = _Rng(seed)
    wins = 0
    counted = 0
    for _ in range(trials):
        car = rng.randint(doors)
        pick = rng.randint(doors)

        if informed_host:
            # host opens `reveals` goat doors, never the pick and never the car
            openable = [d for d in range(doors) if d != pick and d != car]
            # (there are doors-1 or doors-2 such doors; we only need to know which stay closed)
            # choose `reveals` of them to open
            opened = set()
            # simple partial shuffle: pick `reveals` distinct doors from openable
            pool = openable[:]
            for _ in range(reveals):
                j = rng.randint(len(pool))
                opened.add(pool.pop(j))
            closed_others = [d for d in range(doors) if d != pick and d not in opened]
        else:
            # random host (3 doors, 1 reveal): open a random non-pick door
            others = [d for d in range(doors) if d != pick]
            opened_door = others[rng.randint(len(others))]
            if opened_door == car:
                continue  # revealed the car -> not the conditional we're measuring
            opened = {opened_door}
            closed_others = [d for d in range(doors) if d != pick and d not in opened]

        counted += 1
        if strategy == "stay":
            chosen = pick
        else:  # switch to the first remaining closed other door
            chosen = closed_others[0]
        if chosen == car:
            wins += 1
    return wins / counted if counted else 0.0
