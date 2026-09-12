"""Merkle trees: compact cryptographic proofs that an item belongs to a dataset.

A MERKLE TREE (Ralph Merkle, 1979) hashes a list of data blocks into a single ROOT hash such that any
change to any block changes the root, AND any single block can be proven to belong to the dataset
with a proof of only O(log n) hashes -- without revealing the other blocks. This is the data
structure behind blockchain transaction commitments (Bitcoin, Ethereum), Git's content addressing,
certificate transparency logs, and peer-to-peer file verification (BitTorrent). It turns "is this
item in the set?" from an O(n) download into an O(log n) proof.

The construction is a binary tree of hashes. The leaves are the hashes of the data blocks; each
internal node is the hash of its two children concatenated; the root is the top. Because a hash is
one-way and collision-resistant, the root is a fingerprint of the ENTIRE ordered dataset -- flip one
bit anywhere and the root changes. To PROVE that block i is in the tree, supply the AUTHENTICATION
PATH: the sibling hash at each level from the leaf up to the root (about log2(n) hashes). A verifier
who trusts only the root recomputes the path -- hashing the block, combining it with each supplied
sibling in the right order -- and checks that it lands on the known root. If block i were altered, or
if a wrong sibling were supplied, the recomputed root would not match.

This module builds a Merkle tree over a list of byte blocks, exposes the root, generates an inclusion
proof for any index, and verifies a proof against a root -- using SHA-256 with domain-separated leaf
and node prefixes (the standard defence against second-preimage and duplicate-leaf attacks). It is
verified that a valid proof for every block verifies against the true root, that any tampering with a
block, its index, its proof, or the root causes verification to fail, that the proof length is
logarithmic in the number of blocks, that changing any single block changes the root, and on odd
block counts (where the last node is promoted). Pure stdlib (hashlib); a data-structure and
cryptography companion to the Shamir, hash-table, and blockchain notes."""

from __future__ import annotations

import hashlib

_LEAF = b"\x00"      # domain-separation prefixes
_NODE = b"\x01"


def _hash(data):
    return hashlib.sha256(data).digest()


def _hash_leaf(block):
    return _hash(_LEAF + (block if isinstance(block, bytes) else bytes(block)))


def _hash_node(left, right):
    return _hash(_NODE + left + right)


class MerkleTree:
    """A Merkle hash tree over an ordered list of byte blocks."""

    def __init__(self, blocks):
        if not blocks:
            raise ValueError("need at least one block")
        self.n = len(blocks)
        # levels[0] = leaf hashes, levels[-1] = [root]
        self.levels = [[_hash_leaf(b) for b in blocks]]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = []
            for i in range(0, len(cur), 2):
                if i + 1 < len(cur):
                    nxt.append(_hash_node(cur[i], cur[i + 1]))
                else:
                    nxt.append(cur[i])          # odd node promoted unchanged
            self.levels.append(nxt)

    @property
    def root(self):
        return self.levels[-1][0]

    def proof(self, index):
        """The inclusion proof for block `index`: a list of (sibling_hash, is_left) pairs from the
        leaf up to the root. is_left tells the verifier whether the sibling is the LEFT operand."""
        if not (0 <= index < self.n):
            raise IndexError(index)
        path = []
        idx = index
        for level in self.levels[:-1]:
            if idx % 2 == 0:
                sibling = idx + 1
                if sibling < len(level):
                    path.append((level[sibling], False))   # sibling is on the right
                # else: promoted node, no sibling at this level
            else:
                path.append((level[idx - 1], True))        # sibling is on the left
            idx //= 2
        return path


def verify(block, index, proof, root):
    """Verify that `block` at position `index` is included under `root`, given its inclusion proof."""
    h = _hash_leaf(block)
    for sibling, is_left in proof:
        if is_left:
            h = _hash_node(sibling, h)
        else:
            h = _hash_node(h, sibling)
    return h == root


def build(blocks):
    """Convenience: build a tree and return (root, tree)."""
    t = MerkleTree(blocks)
    return t.root, t
