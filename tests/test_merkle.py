"""Tests for merkle: proof validity, tamper detection, logarithmic proofs, odd counts."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from merkle import MerkleTree, verify, build

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def blocks(n):
    return [b"block-%d-data" % i for i in range(n)]


# --- every block's proof verifies against the true root --------------------
for n in [1, 2, 3, 4, 5, 8, 15, 16, 17, 100]:
    bs = blocks(n)
    t = MerkleTree(bs)
    ok = all(verify(bs[i], i, t.proof(i), t.root) for i in range(n))
    check(f"all {n} blocks' proofs verify", ok)

# --- proof length is logarithmic -------------------------------------------
t = MerkleTree(blocks(1000))
proof_lengths = [len(t.proof(i)) for i in range(1000)]
check(f"proof length is about log2(n) (max {max(proof_lengths)} for n=1000)",
      max(proof_lengths) <= math.ceil(math.log2(1000)) + 1)

# --- tampering with the block fails ----------------------------------------
bs = blocks(16)
t = MerkleTree(bs)
check("tampered block fails verification",
      not verify(b"evil-block", 5, t.proof(5), t.root))

# --- tampering with the proof fails ----------------------------------------
p = t.proof(5)
bad_proof = [(h[::-1], flag) for h, flag in p]      # corrupt every sibling hash
check("corrupted proof fails verification", not verify(bs[5], 5, bad_proof, t.root))

# flip a proof direction
if p:
    flipped = [(p[0][0], not p[0][1])] + p[1:]
    check("flipped proof direction fails verification", not verify(bs[5], 5, flipped, t.root))

# --- tampering with the root fails -----------------------------------------
bad_root = bytes([t.root[0] ^ 1]) + t.root[1:]
check("wrong root fails verification", not verify(bs[5], 5, t.proof(5), bad_root))

# --- changing any single block changes the root ----------------------------
root0 = MerkleTree(blocks(8)).root
changed = blocks(8)
changed[3] = b"different"
root1 = MerkleTree(changed).root
check("changing one block changes the root", root0 != root1)

# --- reordering blocks changes the root (order matters) --------------------
bs = blocks(4)
swapped = [bs[1], bs[0], bs[2], bs[3]]
check("reordering blocks changes the root", MerkleTree(bs).root != MerkleTree(swapped).root)

# --- odd block counts (promoted last node) ---------------------------------
for n in [3, 5, 7, 9, 13]:
    bs = blocks(n)
    t = MerkleTree(bs)
    check(f"odd count n={n}: all proofs verify",
          all(verify(bs[i], i, t.proof(i), t.root) for i in range(n)))

# --- a single block ---------------------------------------------------------
t1 = MerkleTree([b"only"])
check("single block: root is its leaf hash and proof is empty",
      verify(b"only", 0, t1.proof(0), t1.root) and t1.proof(0) == [])

# --- leaf/node domain separation: a leaf hash != a node hash of same bytes -
# (second-preimage resistance) -- two blocks whose concatenation could be confused with a node
t = MerkleTree([b"a", b"b"])
# the root is hash(NODE + leaf(a) + leaf(b)); a leaf of the concatenation must differ
from merkle import _hash_leaf
check("leaf and internal-node hashing are domain-separated",
      _hash_leaf(b"ab") != t.root)

# --- second block's proof against first block's data fails -----------------
bs = blocks(8)
t = MerkleTree(bs)
check("block 0's data with block 1's proof fails", not verify(bs[0], 1, t.proof(1), t.root))

# --- build convenience ------------------------------------------------------
root, tree = build(blocks(4))
check("build returns the root and a working tree", root == tree.root)

# --- empty input rejected ---------------------------------------------------
raised = False
try:
    MerkleTree([])
except ValueError:
    raised = True
check("empty block list raises", raised)

# --- proof out of range -----------------------------------------------------
raised = False
try:
    MerkleTree(blocks(4)).proof(10)
except IndexError:
    raised = True
check("out-of-range proof index raises", raised)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all merkle tests passed")
