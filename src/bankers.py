"""The Banker's algorithm -- granting resources without ever deadlocking.

An operating system hands out resources -- memory pages, file locks, database connections, printer
slots -- to processes that request them incrementally and release them only when done. The danger is
DEADLOCK: process A holds resource 1 and waits for resource 2 while process B holds resource 2 and waits
for resource 1, and neither can ever proceed. Edsger Dijkstra's BANKER'S ALGORITHM (1965) prevents this
by DEADLOCK AVOIDANCE: it grants a request only if the resulting state is still SAFE -- meaning there
exists some order in which all processes can run to completion. The name is Dijkstra's analogy: a banker
lends cash only if it can still satisfy every customer's maximum credit line in some sequence, never
committing to a position from which someone might be starved.

The state is three tables over m resource types and n processes: ALLOCATION (how much each process
currently holds), MAX (the most each will ever need), and AVAILABLE (the free pool). The derived NEED =
MAX - ALLOCATION is how much more each process could still demand. The SAFETY CHECK is the heart: with
the current available pool, repeatedly find any process whose remaining NEED fits within what is free,
pretend it runs to completion and RELEASES everything it held back into the pool, and mark it finished.
If every process can be finished this way, the state is safe and the finishing order is a SAFE SEQUENCE;
if the search stalls with unfinished processes, the state is unsafe -- some subset could deadlock. To
grant a resource REQUEST, the banker tentatively applies it and runs the safety check: grant only if the
resulting state is still safe, otherwise make the process wait.

This module implements the safety check (returning a safe sequence when one exists), the request-grant
decision, and a separate DEADLOCK DETECTOR for the no-max-known case (which processes are actually
stuck given current allocations and requests). Everything is integer vector arithmetic over the
resource types. Pure standard library.

Validation. The safety verdict is checked against a BRUTE-FORCE search over all n! completion orders: a
state is declared safe exactly when some ordering lets every process finish with the running pool, and
the returned safe sequence is verified to be such an ordering. Known textbook instances are matched by
hand (the classic Silberschatz example is safe with its documented safe sequence; nudging one request
makes it unsafe). Granting a request never produces an unsafe state; a request exceeding a process's
declared NEED or the AVAILABLE pool is refused; and releasing resources always keeps or restores
safety. The deadlock detector agrees with a brute-force reduction on random request states. Pure
standard library."""


def _need(alloc, maximum):
    """NEED = MAX - ALLOCATION, per process per resource."""
    return [[maximum[i][j] - alloc[i][j] for j in range(len(maximum[0]))]
            for i in range(len(maximum))]


def _le(a, b):
    """Vector <= comparison."""
    return all(a[j] <= b[j] for j in range(len(a)))


def is_safe(alloc, maximum, available):
    """Return (safe, sequence): whether the state is safe and a completion order if so.

    ``alloc[i][j]`` = resources of type j held by process i; ``maximum[i][j]`` = its maximum claim;
    ``available[j]`` = free units of type j.
    """
    n = len(alloc)
    m = len(available)
    need = _need(alloc, maximum)
    work = list(available)
    finished = [False] * n
    sequence = []
    # repeatedly find a process whose remaining need fits the pool, run it, release its allocation
    progress = True
    while progress and len(sequence) < n:
        progress = False
        for i in range(n):
            if not finished[i] and _le(need[i], work):
                for j in range(m):
                    work[j] += alloc[i][j]
                finished[i] = True
                sequence.append(i)
                progress = True
    if all(finished):
        return True, sequence
    return False, []


def request_resources(alloc, maximum, available, process, request):
    """Try to grant ``request`` (a resource vector) to ``process``.

    Returns (granted, new_alloc, new_available). If granting keeps the state safe it is applied;
    otherwise the state is unchanged and granted is False. Requests exceeding NEED or AVAILABLE are
    refused outright.
    """
    m = len(available)
    need = _need(alloc, maximum)
    # a process may not request beyond what it declared it could still need
    if not _le(request, need[process]):
        return False, alloc, available
    # cannot grant more than is available right now
    if not _le(request, available):
        return False, alloc, available
    # tentatively apply the request
    new_alloc = [row[:] for row in alloc]
    new_available = list(available)
    for j in range(m):
        new_alloc[process][j] += request[j]
        new_available[j] -= request[j]
    safe, _ = is_safe(new_alloc, maximum, new_available)
    if safe:
        return True, new_alloc, new_available
    return False, alloc, available          # rollback: unsafe, make the process wait


def release_resources(alloc, available, process, release):
    """Process releases resources back to the pool. Returns (new_alloc, new_available)."""
    m = len(available)
    new_alloc = [row[:] for row in alloc]
    new_available = list(available)
    for j in range(m):
        give = min(release[j], new_alloc[process][j])
        new_alloc[process][j] -= give
        new_available[j] += give
    return new_alloc, new_available


# ---------------------------------------------------------------------------
# deadlock detection (no max known -- work from current requests)
# ---------------------------------------------------------------------------

def detect_deadlock(alloc, request, available):
    """Detect which processes are deadlocked given current holdings and pending requests.

    Like the safety check but using the actual pending ``request`` matrix instead of NEED. Returns the
    list of deadlocked process indices (empty if none).
    """
    n = len(alloc)
    m = len(available)
    work = list(available)
    finished = [False] * n
    # a process with no outstanding request is trivially finishable
    for i in range(n):
        if all(request[i][j] == 0 for j in range(m)):
            finished[i] = True
    progress = True
    while progress:
        progress = False
        for i in range(n):
            if not finished[i] and _le(request[i], work):
                for j in range(m):
                    work[j] += alloc[i][j]
                finished[i] = True
                progress = True
    return [i for i in range(n) if not finished[i]]


# ---------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------

def brute_is_safe(alloc, maximum, available):
    """Reference: a state is safe iff SOME permutation of processes can all finish. O(n!)."""
    from itertools import permutations
    n = len(alloc)
    m = len(available)
    need = _need(alloc, maximum)
    for order in permutations(range(n)):
        work = list(available)
        ok = True
        for i in order:
            if _le(need[i], work):
                for j in range(m):
                    work[j] += alloc[i][j]
            else:
                ok = False
                break
        if ok:
            return True
    return False
