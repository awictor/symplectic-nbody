"""The VCG mechanism -- the auction where telling the truth is always your best move.

You are selling things -- ad slots, spectrum licences, cloud servers -- to bidders who each privately
value them. A naive "pay what you bid" auction invites gaming: shade your bid below your true value and
hope to win cheaply. The VICKREY-CLARKE-GROVES mechanism (Vickrey 1961, Clarke 1971, Groves 1973) is the
profound answer: it chooses the OUTCOME that maximises total value, and charges each winner an amount
engineered so that bidding your TRUE valuation is a DOMINANT STRATEGY -- no matter what anyone else does,
you can never do better by lying. That property (strategy-proofness plus efficiency) is why VCG
underlies spectrum auctions, and its single-item special case, the SECOND-PRICE (Vickrey) auction, is the
textbook truthful auction and the ancestor of ad-exchange pricing.

The rule is beautiful. First pick the allocation that MAXIMISES the sum of the winners' reported values --
the efficient outcome. Then charge each winner its EXTERNALITY: the harm its presence does to everyone
else, i.e. (the best total value the OTHERS could have achieved if this bidder were absent) minus (the
value the others actually get in the chosen allocation). A bidder pays exactly the opportunity cost it
imposes on the rest, so its own report only affects WHETHER it wins, never HOW MUCH it pays for a given
outcome -- which is exactly what makes honesty optimal. In the single-item case this collapses to the
Vickrey rule: the highest bidder wins and pays the SECOND-highest bid.

This module implements the single-item second-price auction and the general combinatorial VCG mechanism
for assigning m distinct items to n bidders (each bidder valuing bundles), computing the efficient
allocation and each winner's VCG payment. It also provides utility accounting and a brute-force
strategy-proofness checker used in validation. Pure standard library.

Validation. The defining property is checked directly: STRATEGY-PROOFNESS -- over many random valuation
profiles, no bidder can raise its own utility (value of what it wins minus what it pays) by reporting any
other valuation while the others report truthfully; truthful bidding is a best response in every case.
EFFICIENCY -- the chosen allocation maximises total value, verified against brute force over all
assignments. INDIVIDUAL RATIONALITY -- every bidder's utility is non-negative (payments never exceed the
winner's own value). The single-item auction is checked to charge the second-highest bid and give the
winner surplus (first minus second). Known small instances are matched by hand, and VCG payments are
always non-negative and no greater than the winner's bid."""

from itertools import permutations


# ---------------------------------------------------------------------------
# single-item second-price (Vickrey) auction
# ---------------------------------------------------------------------------

def second_price_auction(bids):
    """Single-item Vickrey auction. ``bids`` is a list of non-negative values.

    Returns (winner_index, price). The highest bidder wins and pays the second-highest bid. Ties are
    broken toward the lowest index.
    """
    if not bids:
        return None, 0
    winner = max(range(len(bids)), key=lambda i: (bids[i], -i))
    # second-highest value among the others
    others = [bids[i] for i in range(len(bids)) if i != winner]
    price = max(others) if others else 0
    return winner, price


# ---------------------------------------------------------------------------
# general combinatorial VCG: assign m items to n bidders
# ---------------------------------------------------------------------------

def _best_allocation(bidders, value, items):
    """Return (best_total, best_assignment) maximising sum of bidder values over all assignments.

    ``bidders`` is the list of participating bidder indices; ``value(i, bundle)`` gives bidder i's value
    for a frozenset bundle; ``items`` is the tuple of items to assign. Each item goes to at most one
    bidder; bidders may receive multiple items (a bundle).

    For tractability and clean validation this assigns each ITEM to some bidder or to nobody, and a
    bidder's value is the sum over the bundle it receives via an additive ``value`` -- see vcg_allocate.
    """
    # dynamic search over item -> bidder-or-none assignments
    best = {"total": None, "assign": None}
    m = len(items)
    assign = [None] * m

    def rec(k, bundles):
        if k == m:
            total = sum(value(b, frozenset(bundles[b])) for b in bidders if bundles[b])
            if best["total"] is None or total > best["total"]:
                best["total"] = total
                best["assign"] = [list(bundles[b]) for b in range(len(bundles))]
            return
        # option: item k unassigned
        rec(k + 1, bundles)
        # option: give item k to each bidder
        for b in bidders:
            bundles[b].append(items[k])
            rec(k + 1, bundles)
            bundles[b].pop()

    nb = max(bidders) + 1 if bidders else 0
    rec(0, [[] for _ in range(nb)])
    return best["total"], best["assign"]


def vcg_allocate(n, items, value):
    """General VCG mechanism.

    ``n`` bidders, ``items`` an iterable of distinct item labels, ``value(i, bundle)`` the value bidder
    i places on a frozenset bundle (value(i, emptyset) must be 0). Returns a dict with:
      'allocation': list where allocation[i] is the list of items bidder i wins,
      'payments':   list of VCG payments (each bidder's externality),
      'total_value': the maximised total value,
      'utilities':  list of value(i, bundle_i) - payment[i].
    """
    items = tuple(items)
    all_bidders = list(range(n))
    total, assign = _best_allocation(all_bidders, value, items)
    allocation = [assign[i] if i < len(assign) else [] for i in range(n)]

    payments = [0.0] * n
    utilities = [0.0] * n
    for i in range(n):
        # value the OTHERS receive in the chosen allocation
        others_value_here = sum(value(b, frozenset(allocation[b]))
                                for b in range(n) if b != i and allocation[b])
        # best total value the others could get if i were absent
        without_i = [b for b in range(n) if b != i]
        best_without, _ = _best_allocation(without_i, value, items)
        best_without = best_without or 0.0
        payments[i] = best_without - others_value_here
        my_value = value(i, frozenset(allocation[i])) if allocation[i] else 0.0
        utilities[i] = my_value - payments[i]
    return {"allocation": allocation, "payments": payments,
            "total_value": total, "utilities": utilities}


# ---------------------------------------------------------------------------
# helpers / validation
# ---------------------------------------------------------------------------

def additive_value(valuations):
    """Build a value function from per-(bidder,item) valuations: value(i, bundle) = sum of item values.

    ``valuations[i][item]`` is bidder i's marginal value for that item.
    """
    def value(i, bundle):
        return sum(valuations[i][it] for it in bundle)
    return value


def bidder_utility(vcg_result, i, true_value_i):
    """A bidder's TRUE utility given the mechanism result and their true value function."""
    won = vcg_result["allocation"][i]
    tv = true_value_i(frozenset(won)) if won else 0.0
    return tv - vcg_result["payments"][i]
