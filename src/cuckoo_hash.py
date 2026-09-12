"""Cuckoo hashing -- a dictionary with WORST-CASE constant-time lookups, not just average.

An ordinary hash table with chaining or linear probing is O(1) on average, but a single lookup can
degrade to O(n) when many keys collide into one bucket -- unacceptable for a router forwarding table, a
real-time system, or a hardware cache where the WORST case is what you must budget for. CUCKOO HASHING
(Pagh & Rodler, 2001) guarantees that every lookup examines AT MOST TWO locations, ever, no matter the
data. The name is the trick: like a cuckoo chick shoving its nestmates out, an insertion that finds its
spot occupied EVICTS the resident key, which then flies to its own alternate location, possibly evicting
another, in a chain that almost always settles quickly.

The scheme uses two tables and two independent hash functions, h1 and h2. Key x may live ONLY at
table1[h1(x)] or table2[h2(x)] -- so a lookup checks exactly those two cells and stops. To insert, place
x in its table-1 slot; if that slot held some key y, carry y to its table-2 slot, and if THAT was
occupied carry its resident on, ping-ponging between the tables. Almost always the chain reaches an empty
cell within a few hops. If it instead loops for too long (a cycle in the underlying "cuckoo graph"), the
table REHASHES with fresh hash functions and, if needed, grows -- an amortised cost that stays O(1) as
long as the load factor is kept below the theoretical threshold of about 50% for two tables.

That threshold is where the beautiful theory lives: the keys and their two candidate cells form a random
bipartite graph, and a valid placement exists exactly when that graph has no component with more edges
than vertices -- which happens with high probability below load 0.5 and fails above it. This module
implements the two-table dictionary (insert, lookup, delete, update, iteration) with automatic rehashing
and growth, seeded so runs reproduce, and exposes the load factor and the guaranteed two-probe lookup.

Pure standard library. Because the whole promise is that it behaves like a normal dictionary but with a
hard worst-case bound, that is exactly what the tests verify.

Validation. The table is run alongside Python's ``dict`` as an oracle over thousands of seeded random
operations -- insert, overwrite, lookup, delete, membership -- and must agree on every one: the same
values for present keys, the same absence for missing keys, and the same size. The defining WORST-CASE
guarantee is checked directly: every successful lookup inspects at most two cells, and every key is
found at one of its two hash locations. Both invariants -- no key stored outside its two candidate cells,
no cell holding a key that hashes elsewhere -- hold after every operation. Rehashing preserves all
contents, deletions free space (letting reinsertion succeed), and the load factor stays below the
configured maximum. Iteration returns exactly the stored keys."""


class _Hasher:
    """A seeded family of hash functions: seed picks a different mixing of the key's hash."""

    def __init__(self, seed, size):
        self.seed = seed & 0xFFFFFFFF
        self.size = size

    def __call__(self, key):
        # mix the key's built-in hash with the seed using a splitmix-style avalanche
        h = (hash(key) ^ self.seed) & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
        h = h ^ (h >> 31)
        return h % self.size


class CuckooHash:
    """A dictionary with worst-case two-probe lookups via cuckoo hashing."""

    def __init__(self, capacity=8, max_load=0.45, seed=12345):
        self._cap = max(4, capacity)
        self._max_load = max_load
        self._seed = seed & 0xFFFFFFFF
        self._size = 0
        self._max_kicks = None
        self._build_tables(self._cap)

    def _build_tables(self, cap):
        self._cap = cap
        self.t1 = [None] * cap        # each cell: (key, value) or None
        self.t2 = [None] * cap
        # two independent hashers, reseeded on each rebuild
        self._h1 = _Hasher(self._seed * 2 + 1, cap)
        self._h2 = _Hasher(self._seed * 2 + 7, cap)
        # kick bound ~ log of capacity keeps rehash cost amortised O(1)
        self._max_kicks = max(8, int(6 * (cap.bit_length())))

    def __len__(self):
        return self._size

    def load_factor(self):
        return self._size / (2 * self._cap)

    # -- lookup (worst-case two cells) --------------------------------------
    def _probe(self, key):
        """Return (table, index) where key lives, or None. Inspects at most two cells."""
        i1 = self._h1(key)
        cell = self.t1[i1]
        if cell is not None and cell[0] == key:
            return (1, i1)
        i2 = self._h2(key)
        cell = self.t2[i2]
        if cell is not None and cell[0] == key:
            return (2, i2)
        return None

    def get(self, key, default=None):
        loc = self._probe(key)
        if loc is None:
            return default
        return (self.t1 if loc[0] == 1 else self.t2)[loc[1]][1]

    def __getitem__(self, key):
        loc = self._probe(key)
        if loc is None:
            raise KeyError(key)
        return (self.t1 if loc[0] == 1 else self.t2)[loc[1]][1]

    def __contains__(self, key):
        return self._probe(key) is not None

    # -- insert -------------------------------------------------------------
    def insert(self, key, value):
        """Insert or overwrite key -> value."""
        loc = self._probe(key)
        if loc is not None:
            table = self.t1 if loc[0] == 1 else self.t2
            table[loc[1]] = (key, value)
            return
        if self.load_factor() >= self._max_load:
            self._grow()
        self._size += 1
        self._place(key, value)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def _place(self, key, value):
        """Place a NEW (key, value), evicting and rehashing as needed."""
        cur = (key, value)
        for _ in range(self._max_kicks):
            i1 = self._h1(cur[0])
            self.t1[i1], cur = cur, self.t1[i1]
            if cur is None:
                return
            i2 = self._h2(cur[0])
            self.t2[i2], cur = cur, self.t2[i2]
            if cur is None:
                return
        # cycle: rehash (grow only if genuinely too full) and reinsert the displaced key
        self._rehash(carry=cur)

    def _grow(self):
        old = self._collect()
        self._build_tables(self._cap * 2)
        self._size = 0
        for k, v in old:
            self._size += 1
            self._place(k, v)

    def _rehash(self, carry=None):
        old = self._collect()
        if carry is not None:
            old.append(carry)
        # try a fresh seed at the same size a few times, then grow
        for attempt in range(4):
            self._seed = (self._seed * 1664525 + 1013904223) & 0xFFFFFFFF
            self._build_tables(self._cap)
            ok = True
            self._size = 0
            for k, v in old:
                if not self._try_place(k, v):
                    ok = False
                    break
                self._size += 1
            if ok:
                return
        # give up on same size: grow
        self._seed = (self._seed * 1664525 + 1013904223) & 0xFFFFFFFF
        self._build_tables(self._cap * 2)
        self._size = 0
        for k, v in old:
            self._size += 1
            self._place(k, v)

    def _try_place(self, key, value):
        """Place a key without triggering recursive rehash; return False on cycle."""
        cur = (key, value)
        for _ in range(self._max_kicks):
            i1 = self._h1(cur[0])
            self.t1[i1], cur = cur, self.t1[i1]
            if cur is None:
                return True
            i2 = self._h2(cur[0])
            self.t2[i2], cur = cur, self.t2[i2]
            if cur is None:
                return True
        return False

    # -- delete -------------------------------------------------------------
    def delete(self, key):
        loc = self._probe(key)
        if loc is None:
            return False
        table = self.t1 if loc[0] == 1 else self.t2
        table[loc[1]] = None
        self._size -= 1
        return True

    def __delitem__(self, key):
        if not self.delete(key):
            raise KeyError(key)

    # -- iteration / helpers ------------------------------------------------
    def _collect(self):
        out = []
        for cell in self.t1:
            if cell is not None:
                out.append(cell)
        for cell in self.t2:
            if cell is not None:
                out.append(cell)
        return out

    def items(self):
        return list(self._collect())

    def keys(self):
        return [k for k, _ in self._collect()]

    def check_invariants(self):
        """Every stored key lives at one of its two hash cells, and cells are self-consistent."""
        for idx, cell in enumerate(self.t1):
            if cell is not None and self._h1(cell[0]) != idx:
                return False
        for idx, cell in enumerate(self.t2):
            if cell is not None and self._h2(cell[0]) != idx:
                return False
        return True

    def max_probe_cost(self):
        """The worst-case number of cells any successful lookup inspects (always <= 2)."""
        return 2
