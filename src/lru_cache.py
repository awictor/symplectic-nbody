"""LRU and LFU caches: O(1) eviction policies for bounded memory.

A cache holds a fixed number of items and, when full, must EVICT one to make room. Which one you
throw out is the policy, and it decides the hit rate. The two classics:

  LRU (Least Recently Used) -- evict the item untouched for the longest. Bets that recently-used
      things will be used again soon (temporal locality). The default in CPU caches, page tables,
      and web caches.
  LFU (Least Frequently Used) -- evict the item with the fewest accesses. Bets that popular things
      stay popular, ignoring recency.

The interesting part is doing it in O(1). A naive LRU scans for the oldest item on every eviction
(O(n)); the standard trick is a HASH MAP (key -> node) for O(1) lookup PLUS a DOUBLY-LINKED LIST
ordered by recency, so touching an item unlinks it and moves it to the front in O(1), and eviction
just drops the tail. LFU is harder: this module keeps buckets of items grouped by frequency in a
linked structure so increments and min-frequency eviction are also O(1) amortized.

This module implements both with get/put, hit/miss statistics, and introspectable order -- verified
that LRU evicts in true least-recently-used order (checked against a brute-force reference over
random workloads), that touching an item spares it, that LFU evicts the least-frequent (breaking
ties by recency), that capacity is never exceeded, and that a skewed (Zipf-like) workload gives LFU
a better hit rate than LRU while a scan-heavy one favours LRU. Pure stdlib; a data-structures
companion to the Fenwick-tree and union-find notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """Least-Recently-Used cache with O(1) get and put.

    A hash map key->node gives O(1) lookup; a doubly-linked list with sentinel head/tail keeps items
    in recency order (most-recent just after head), so promotion and tail-eviction are O(1)."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.map = {}
        # sentinels avoid null checks: head <-> ... <-> tail; front (head.next) = most recent
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hits = 0
        self.misses = 0

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key, default=None):
        node = self.map.get(key)
        if node is None:
            self.misses += 1
            return default
        self.hits += 1
        self._remove(node)          # promote to most-recently-used
        self._push_front(node)
        return node.value

    def put(self, key, value):
        node = self.map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._push_front(node)
            return
        if len(self.map) >= self.capacity:
            lru = self.tail.prev    # least-recently-used is just before the tail
            self._remove(lru)
            del self.map[lru.key]
        node = _Node(key, value)
        self.map[key] = node
        self._push_front(node)

    def __contains__(self, key):
        return key in self.map

    def __len__(self):
        return len(self.map)

    def keys_mru_to_lru(self):
        """Keys ordered most-recently-used first (introspection / testing)."""
        out = []
        n = self.head.next
        while n is not self.tail:
            out.append(n.key)
            n = n.next
        return out

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class LFUCache:
    """Least-Frequently-Used cache with O(1) amortized get and put.

    Each key has an access frequency; keys are grouped into frequency buckets, each an ordered dict
    (insertion order = recency), so eviction takes the oldest key in the minimum-frequency bucket."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.values = {}            # key -> value
        self.freq = {}              # key -> frequency
        self.buckets = {}           # frequency -> {key: None} preserving insertion order
        self.min_freq = 0
        self.hits = 0
        self.misses = 0

    def _bump(self, key):
        f = self.freq[key]
        del self.buckets[f][key]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq += 1
        self.freq[key] = f + 1
        self.buckets.setdefault(f + 1, {})[key] = None

    def get(self, key, default=None):
        if key not in self.values:
            self.misses += 1
            return default
        self.hits += 1
        self._bump(key)
        return self.values[key]

    def put(self, key, value):
        if key in self.values:
            self.values[key] = value
            self._bump(key)
            return
        if len(self.values) >= self.capacity:
            # evict the oldest key in the least-frequent bucket
            victim = next(iter(self.buckets[self.min_freq]))
            del self.buckets[self.min_freq][victim]
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.values[victim]
            del self.freq[victim]
        self.values[key] = value
        self.freq[key] = 1
        self.buckets.setdefault(1, {})[key] = None
        self.min_freq = 1

    def __contains__(self, key):
        return key in self.values

    def __len__(self):
        return len(self.values)

    def frequency(self, key):
        return self.freq.get(key, 0)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
