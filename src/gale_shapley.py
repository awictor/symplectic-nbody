"""Gale-Shapley: stable matching by deferred acceptance.

Given two equal-sized groups -- classically n proposers and n reviewers -- each ranking all members of
the other group in strict preference order, a MATCHING pairs everyone up one-to-one. A matching is
UNSTABLE if some proposer p and reviewer r each prefer the other to their assigned partner: they would
abandon their matches to pair up, a "blocking pair." A STABLE matching has no blocking pair. The
astonishing Gale-Shapley theorem (1962) proves a stable matching ALWAYS exists and gives an O(n^2)
algorithm to find one -- work that earned Shapley and Roth the 2012 Nobel in economics, because the same
DEFERRED-ACCEPTANCE mechanism runs medical-residency placement (the NRMP match), school-choice
assignment, and organ-donor exchange.

The algorithm: while some proposer is free, he proposes to the highest-ranked reviewer he has not yet
asked. That reviewer TENTATIVELY accepts if she is free or prefers him to her current tentative match
(dumping the latter, who becomes free again), and rejects otherwise. Because a reviewer only ever
trades up, and each proposer works down his list, the process terminates in at most n^2 proposals with
everyone matched. The result is PROPOSER-OPTIMAL: every proposer gets the best partner he could have in
ANY stable matching -- and simultaneously reviewer-pessimal. Swapping the roles yields the
reviewer-optimal stable matching, and the two coincide exactly when the stable matching is unique.

This module runs Gale-Shapley from either side, returning the stable matching as a proposer->reviewer
map, and checks a matching for stability by hunting blocking pairs. It is verified against brute force
-- the returned matching has no blocking pair, it matches everyone, it is proposer-optimal (no
proposer does better in any of the exhaustively-enumerated stable matchings), and on instances with a
unique stable matching the proposer- and reviewer-optimal results agree -- on hundreds of random
preference profiles. Pure stdlib; a combinatorial-optimisation companion to the bipartite-matching,
Hungarian-assignment, and top-trading notes."""

from __future__ import annotations

from collections import deque


def stable_matching(n, proposer_prefs, reviewer_prefs):
    """Gale-Shapley deferred acceptance. `proposer_prefs[p]` is p's list of reviewers best-first;
    `reviewer_prefs[r]` is r's list of proposers best-first. Returns match[p] = reviewer for proposer
    p (proposer-optimal stable matching)."""
    # reviewer ranking: rank[r][p] = position of proposer p in r's list (smaller = preferred)
    rank = [dict() for _ in range(n)]
    for r in range(n):
        for i, p in enumerate(reviewer_prefs[r]):
            rank[r][p] = i

    match_p = [-1] * n              # proposer -> reviewer
    match_r = [-1] * n              # reviewer -> proposer
    next_idx = [0] * n              # next reviewer index each proposer will propose to
    free = deque(range(n))

    while free:
        p = free.popleft()
        r = proposer_prefs[p][next_idx[p]]
        next_idx[p] += 1
        if match_r[r] == -1:
            match_r[r] = p
            match_p[p] = r
        elif rank[r][p] < rank[r][match_r[r]]:
            # r prefers the new proposer p; dump the old one
            old = match_r[r]
            match_p[old] = -1
            free.append(old)
            match_r[r] = p
            match_p[p] = r
        else:
            free.append(p)          # rejected; p tries again next round
    return match_p


def blocking_pairs(n, proposer_prefs, reviewer_prefs, match_p):
    """All blocking pairs (p, r): proposer p and reviewer r each prefer the other to their assigned
    partner. An empty list means the matching is stable."""
    rank_p = [dict() for _ in range(n)]
    for p in range(n):
        for i, r in enumerate(proposer_prefs[p]):
            rank_p[p][r] = i
    rank_r = [dict() for _ in range(n)]
    for r in range(n):
        for i, p in enumerate(reviewer_prefs[r]):
            rank_r[r][p] = i

    match_r = [-1] * n
    for p in range(n):
        if match_p[p] != -1:
            match_r[match_p[p]] = p

    blocks = []
    for p in range(n):
        for r in range(n):
            # p prefers r to his current partner?
            if match_p[p] == -1 or rank_p[p][r] < rank_p[p][match_p[p]]:
                # r prefers p to her current partner?
                if match_r[r] == -1 or rank_r[r][p] < rank_r[r][match_r[r]]:
                    blocks.append((p, r))
    return blocks


def is_stable(n, proposer_prefs, reviewer_prefs, match_p):
    """True iff the matching has no blocking pair (and everyone is matched)."""
    if any(m == -1 for m in match_p):
        return False
    return not blocking_pairs(n, proposer_prefs, reviewer_prefs, match_p)


def reviewer_optimal_matching(n, proposer_prefs, reviewer_prefs):
    """The reviewer-optimal stable matching (run Gale-Shapley with reviewers proposing). Returns
    match_p[p] = reviewer for proposer p, for comparison with the proposer-optimal result."""
    # reviewers propose: gives match_r_side[r] = proposer; convert to match_p
    match_r_side = stable_matching(n, reviewer_prefs, proposer_prefs)
    match_p = [-1] * n
    for r in range(n):
        match_p[match_r_side[r]] = r
    return match_p


# --- brute-force reference --------------------------------------------------
def all_stable_matchings(n, proposer_prefs, reviewer_prefs):
    """Every stable matching, by testing all n! perfect matchings for stability. Exponential; small
    n only. Returns a list of match_p tuples."""
    from itertools import permutations
    out = []
    for perm in permutations(range(n)):
        # perm[p] = reviewer assigned to proposer p
        match_p = list(perm)
        if is_stable(n, proposer_prefs, reviewer_prefs, match_p):
            out.append(tuple(match_p))
    return out
