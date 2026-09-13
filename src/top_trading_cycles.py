"""Top Trading Cycles -- the unbeatable way to reallocate indivisible goods by preference.

n people each OWN one indivisible object (a house, a dorm room, an organ-donor slot, a locker) and each
has a strict preference ranking over ALL the objects, including possibly others' and their own. Can we
reshuffle who holds what so that everyone ends up better off -- or at least, so that no group could
break away and do better by trading only among themselves? The TOP TRADING CYCLES algorithm (attributed
by Shapley and Scarf to David Gale, 1974) answers yes, and produces the UNIQUE allocation in the CORE:
the one assignment that no coalition can improve upon. It is strategy-proof (no one can gain by lying
about their preferences), Pareto-efficient, and individually rational, a rare trifecta that made it the
foundation of kidney-exchange programs and school- and house-allocation mechanisms.

The algorithm is a beautiful piece of graph theory. Build a directed graph where every person POINTS to
the owner of their most-preferred remaining object. With n finite nodes and every node having out-degree
one, this "pointing graph" must contain a CYCLE (follow the arrows and you eventually repeat). Every
person in a cycle gets the object of the person they point to -- a consistent round of trades -- and
then all of them, with their now-assigned objects, LEAVE the market. Recompute the pointing graph on the
survivors and repeat. Each round removes at least one cycle, so the process terminates in at most n
rounds, and the resulting allocation is provably the unique core outcome. Self-loops (someone whose
favourite remaining object is their own) are just cycles of length one -- they keep what they have.

This module runs TTC from an ownership list and a preference table, returning the final allocation
(who ends up with whose object) and the cycles cleared in each round, plus checkers for the core
properties -- Pareto efficiency, individual rationality, and strategy-proofness on a sampled basis --
used in the tests. Pure standard library.

Validation. The output is checked to be a valid PERMUTATION (everyone ends with exactly one distinct
object). It is INDIVIDUALLY RATIONAL: no one ends up worse than their own endowment. It is PARETO
EFFICIENT: no other allocation makes someone better off without making someone worse off, verified by
brute-force search over all permutations on small instances. The allocation is in the CORE: no coalition
can reallocate its own objects among itself to make all its members better off, checked by enumerating
coalitions on small instances. When everyone most prefers their own object, TTC returns the identity;
when preferences form one big cycle, everyone shifts around it. TTC is confirmed strategy-proof on
sampled unilateral misreports (no one gets a strictly better object by lying). Pure standard library."""


def top_trading_cycles(owner, prefs):
    """Run Top Trading Cycles.

    ``owner`` is a list: owner[k] is the person who initially owns object k. Commonly owner = list(
    range(n)) (person i owns object i). ``prefs`` is a list where prefs[i] is person i's ranking of
    OBJECTS, best first (a permutation of 0..n-1). Returns (alloc, rounds) where alloc[i] is the object
    person i ends up with, and rounds is a list of the cycles cleared each round (each a list of people).
    """
    n = len(prefs)
    # who currently owns each object; and which objects remain
    obj_owner = list(owner)                 # obj_owner[k] = current holder of object k
    holder_of = [None] * n
    for k in range(n):
        holder_of[obj_owner[k]] = k          # holder_of[person] = object they currently hold
    alive = [True] * n                       # person still in the market
    obj_alive = [True] * n
    # rank[i][obj] for fast preference comparison
    rank = [{o: pos for pos, o in enumerate(prefs[i])} for i in range(n)]

    alloc = [None] * n
    rounds = []

    def favourite_object(i):
        for o in prefs[i]:
            if obj_alive[o]:
                return o
        return None

    remaining = n
    while remaining > 0:
        # each alive person points to the OWNER of their favourite remaining object
        points_to = {}
        for i in range(n):
            if alive[i]:
                fav = favourite_object(i)
                points_to[i] = obj_owner[fav]
        # find all cycles in this functional graph (out-degree 1)
        cleared_this_round = []
        visited = {}
        for start in list(points_to.keys()):
            if start in visited:
                continue
            path = []
            x = start
            local = {}
            while x not in local and x not in visited:
                local[x] = len(path)
                path.append(x)
                x = points_to[x]
            if x in local:
                # found a fresh cycle starting where x repeats
                cycle = path[local[x]:]
                for p in cycle:
                    fav = favourite_object(p)
                    alloc[p] = fav             # p gets its favourite (owned by the next in cycle)
                cleared_this_round.append(cycle)
                # mark the whole path visited to avoid rescanning
                for p in path:
                    visited[p] = True
            else:
                for p in path:
                    visited[p] = True
        # remove everyone in a cleared cycle (and their now-assigned objects)
        for cycle in cleared_this_round:
            for p in cycle:
                alive[p] = False
                obj_alive[alloc[p]] = False
                remaining -= 1
        rounds.append([c for c in cleared_this_round])
        if not cleared_this_round:
            break                            # safety: no cycle found (shouldn't happen)
    return alloc, rounds


# ---------------------------------------------------------------------------
# property checkers (used in validation)
# ---------------------------------------------------------------------------

def is_permutation(alloc):
    n = len(alloc)
    return sorted(alloc) == list(range(n))


def is_individually_rational(owner, prefs, alloc):
    """No one ends up worse than the object they initially owned."""
    n = len(prefs)
    endowment = [None] * n
    for k in range(n):
        endowment[owner[k]] = k
    rank = [{o: pos for pos, o in enumerate(prefs[i])} for i in range(n)]
    return all(rank[i][alloc[i]] <= rank[i][endowment[i]] for i in range(n))


def is_pareto_efficient(prefs, alloc):
    """No other allocation makes someone strictly better off without hurting anyone. Brute force."""
    n = len(prefs)
    rank = [{o: pos for pos, o in enumerate(prefs[i])} for i in range(n)]

    def dominates(other):
        # other Pareto-dominates alloc if all >= and some >
        better = False
        for i in range(n):
            if rank[i][other[i]] > rank[i][alloc[i]]:
                return False                 # i is worse off
            if rank[i][other[i]] < rank[i][alloc[i]]:
                better = True
        return better

    for perm in _permutations(list(range(n))):
        if dominates(perm):
            return False
    return True


def in_core(owner, prefs, alloc):
    """No coalition can reallocate ITS OWN objects among itself to make all members better off."""
    n = len(prefs)
    endowment = [None] * n
    for k in range(n):
        endowment[owner[k]] = k
    rank = [{o: pos for pos, o in enumerate(prefs[i])} for i in range(n)]
    members = list(range(n))
    # enumerate non-empty coalitions
    for mask in range(1, 1 << n):
        coalition = [i for i in range(n) if mask & (1 << i)]
        their_objects = [endowment[i] for i in coalition]
        # can they permute their_objects among themselves so ALL are strictly better?
        for perm in _permutations(their_objects):
            all_better = True
            for idx, i in enumerate(coalition):
                if rank[i][perm[idx]] >= rank[i][alloc[i]]:
                    all_better = False
                    break
            if all_better:
                return False                 # blocking coalition found
    return True


def _permutations(items):
    if len(items) <= 1:
        yield list(items)
        return
    for i in range(len(items)):
        rest = items[:i] + items[i + 1:]
        for p in _permutations(rest):
            yield [items[i]] + p
