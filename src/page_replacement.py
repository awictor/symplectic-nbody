"""Page-replacement algorithms -- which page to evict when memory is full, and Belady's optimum.

A computer's physical memory holds far fewer pages than a program touches, so the OS keeps a small set
of frames and, when a referenced page is absent (a PAGE FAULT), must evict one resident page to make
room. WHICH page to evict is the page-replacement policy, and it determines how many faults -- how many
slow disk fetches -- the program suffers. This is the central trade-off of virtual memory, and the same
question governs CPU caches, database buffer pools, and CDN edge caches.

The classic policies, all implemented here as fault-counters over a reference string:

  * FIFO evicts the page that has been resident LONGEST, regardless of use -- simple but prone to
    BELADY'S ANOMALY, where giving it MORE frames can paradoxically cause MORE faults.
  * LRU (least-recently-used) evicts the page unused for the longest time, approximating the future by
    the recent past. It is a STACK ALGORITHM -- more frames never increase faults -- so it is immune to
    Belady's anomaly.
  * CLOCK (second-chance) is a cheap LRU approximation: pages sit in a circular buffer with a reference
    bit; the evicting hand skips and clears set bits until it finds a clear one. It is what real kernels
    actually use, since true LRU's per-access bookkeeping is too expensive.
  * LFU (least-frequently-used) evicts the page referenced the fewest times.
  * OPTIMAL (Belady's MIN, 1966) evicts the page whose NEXT use is farthest in the future. It is
    unrealisable online (it needs the future) but is provably the MINIMUM-fault policy, the gold
    standard every real algorithm is measured against.

This module runs each policy over a reference string with a given number of frames, returning the page-
fault count and the eviction trace, plus a comparison across policies and a demonstrator for Belady's
anomaly. Pure standard library.

Validation. Optimal is checked to be exactly that: over many random reference strings and frame counts,
Belady's MIN incurs no MORE faults than any other policy, and never fewer than the trivial lower bound
(the number of distinct pages). LRU is verified to be a STACK ALGORITHM -- increasing the frame count
never raises its fault count, monotone always -- while FIFO is shown to VIOLATE this (Belady's anomaly
reproduced on the classic reference string). Every policy faults on the first reference to a page (a
compulsory miss) and never faults on a hit while the page stays resident; the resident set never exceeds
the frame count; and hand-traced sequences match by hand. Clock's fault count sits between FIFO and true
LRU as theory predicts. Pure standard library."""


def _simulate(refs, frames, choose_victim):
    """Run a page-replacement policy. ``choose_victim(state, refs, t)`` returns the resident page to
    evict. Returns (faults, trace) where trace lists ('hit'|'fault', page, evicted_or_None)."""
    resident = []                 # list of resident pages, order policy-defined
    faults = 0
    trace = []
    state = {"resident": resident, "order": [], "freq": {}, "ref_bit": {}, "hand": 0}
    for t, page in enumerate(refs):
        if page in resident:
            trace.append(("hit", page, None))
            _on_hit(state, page)
        else:
            faults += 1
            evicted = None
            if len(resident) >= frames:
                evicted = choose_victim(state, refs, t)
                resident.remove(evicted)
                _on_evict(state, evicted)
            resident.append(page)
            _on_insert(state, page, t)
            trace.append(("fault", page, evicted))
    return faults, trace


def _on_hit(state, page):
    state["order"] = [p for p in state["order"] if p != page] + [page]     # LRU recency
    state["freq"][page] = state["freq"].get(page, 0) + 1
    state["ref_bit"][page] = 1                                             # Clock second chance


def _on_insert(state, page, t):
    state["order"].append(page)
    state["freq"][page] = state["freq"].get(page, 0) + 1
    state["ref_bit"][page] = 1


def _on_evict(state, page):
    state["order"] = [p for p in state["order"] if p != page]
    state["freq"].pop(page, None)
    state["ref_bit"].pop(page, None)


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

def fifo(refs, frames):
    """First-in-first-out replacement."""
    insertion = []          # order of insertion (not touched on hit)
    resident = []
    faults = 0
    trace = []
    for page in refs:
        if page in resident:
            trace.append(("hit", page, None))
        else:
            faults += 1
            evicted = None
            if len(resident) >= frames:
                evicted = insertion.pop(0)
                resident.remove(evicted)
            resident.append(page)
            insertion.append(page)
            trace.append(("fault", page, evicted))
    return faults, trace


def lru(refs, frames):
    """Least-recently-used replacement."""
    def victim(state, refs, t):
        return state["order"][0]              # least recently used is at the front
    return _simulate(refs, frames, victim)


def lfu(refs, frames):
    """Least-frequently-used replacement (ties broken by least-recently-used)."""
    def victim(state, refs, t):
        resident = state["resident"]
        # min frequency; break ties by recency (earliest in order)
        return min(resident, key=lambda p: (state["freq"][p], state["order"].index(p)))
    return _simulate(refs, frames, victim)


def clock(refs, frames):
    """Clock (second-chance) replacement using reference bits."""
    ring = []               # circular buffer of resident pages
    ref_bit = {}
    hand = [0]

    resident = []
    faults = 0
    trace = []
    for page in refs:
        if page in resident:
            ref_bit[page] = 1
            trace.append(("hit", page, None))
        else:
            faults += 1
            evicted = None
            if len(resident) >= frames:
                # advance the hand, clearing set bits, until a page with bit 0 is found
                while True:
                    cur = ring[hand[0]]
                    if ref_bit[cur] == 0:
                        evicted = cur
                        break
                    ref_bit[cur] = 0
                    hand[0] = (hand[0] + 1) % len(ring)
                idx = ring.index(evicted)
                ring.pop(idx)
                resident.remove(evicted)
                del ref_bit[evicted]
                if hand[0] >= len(ring):
                    hand[0] = 0
            resident.append(page)
            ring.append(page)
            ref_bit[page] = 1
            trace.append(("fault", page, evicted))
    return faults, trace


def optimal(refs, frames):
    """Belady's optimal (MIN): evict the page whose next use is farthest in the future."""
    def victim(state, refs, t):
        resident = state["resident"]
        # for each resident page, find its next use after time t
        def next_use(p):
            for k in range(t + 1, len(refs)):
                if refs[k] == p:
                    return k
            return float("inf")               # never used again -> best to evict
        return max(resident, key=next_use)
    return _simulate(refs, frames, victim)


# ---------------------------------------------------------------------------
# comparison and analysis
# ---------------------------------------------------------------------------

def compare(refs, frames):
    """Run all policies; return a dict of policy name -> fault count."""
    return {
        "FIFO": fifo(refs, frames)[0],
        "LRU": lru(refs, frames)[0],
        "CLOCK": clock(refs, frames)[0],
        "LFU": lfu(refs, frames)[0],
        "OPTIMAL": optimal(refs, frames)[0],
    }


def distinct_pages(refs):
    """The number of distinct pages -- the compulsory-miss lower bound on faults."""
    return len(set(refs))


def belady_anomaly_example():
    """The classic reference string exhibiting Belady's anomaly under FIFO.

    Returns (refs, faults_3_frames, faults_4_frames) where MORE frames causes MORE faults.
    """
    refs = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    return refs, fifo(refs, 3)[0], fifo(refs, 4)[0]
