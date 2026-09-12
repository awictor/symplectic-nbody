"""Tests for elliptic_curve: group axioms, order, ECDH agreement, ECDSA sign/verify."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elliptic_curve import (Curve, curve_f17, curve_toy, inverse_mod,  # noqa: E402
                            generate_keypair, ecdh_shared_secret, ecdsa_sign, ecdsa_verify, _LCG)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def all_points(curve):
    """Every affine point on a small curve, by brute force (plus the identity)."""
    pts = [None]
    for x in range(curve.p):
        for y in range(curve.p):
            if curve.is_on_curve((x, y)):
                pts.append((x, y))
    return pts


def main():
    # ---- 1. modular inverse sanity ----------------------------------------------------
    check("inverse_mod correct", (3 * inverse_mod(3, 17)) % 17 == 1)

    c = curve_f17()

    # ---- 2. every generated point is on the curve ------------------------------------
    pts = all_points(c)
    check("all brute-forced points satisfy the curve equation", all(c.is_on_curve(P) for P in pts))
    # F_17 curve y^2=x^3+2x+2 has 19 points (order 19)
    check("F17 curve has 19 points", len(pts) == 19, f"{len(pts)}")

    # ---- 3. group axioms, checked exhaustively ---------------------------------------
    # identity
    check("identity: None + P == P", all(c.add(None, P) == P for P in pts))
    check("identity: P + None == P", all(c.add(P, None) == P for P in pts))
    # inverses
    check("P + (-P) == identity", all(c.add(P, c.negate(P)) is None for P in pts))
    # commutativity
    comm = all(c.add(P, Q) == c.add(Q, P) for P in pts for Q in pts)
    check("addition is commutative", comm)
    # closure: sum is always on the curve
    closed = all(c.is_on_curve(c.add(P, Q)) for P in pts for Q in pts)
    check("addition is closed (result on curve)", closed)
    # associativity (sample to keep it quick but thorough)
    assoc = True
    for i, P in enumerate(pts):
        for Q in pts:
            for R in pts[:6]:
                if c.add(c.add(P, Q), R) != c.add(P, c.add(Q, R)):
                    assoc = False
    check("addition is associative", assoc)

    # ---- 4. base point order and scalar multiplication --------------------------------
    check("F17 base point order is 19", c.order_of(c.G) == 19)
    check("n*G == identity", c.scalar_mult(c.n, c.G) is None)
    # scalar mult consistent with repeated addition
    acc = None
    consistent = True
    for k in range(1, 20):
        acc = c.add(acc, c.G)
        if c.scalar_mult(k, c.G) != acc:
            consistent = False
    check("k*G equals G added k times", consistent)
    # negative scalar
    check("(-k)*G == -(k*G)", c.scalar_mult(-3, c.G) == c.negate(c.scalar_mult(3, c.G)))

    # ---- 5. ECDH agreement on the toy curve ------------------------------------------
    t = curve_toy()
    agree = 0
    trials = 40
    for seed in range(trials):
        rng_a = _LCG(seed * 7 + 1)
        rng_b = _LCG(seed * 13 + 5)
        da, Qa = generate_keypair(t, rng_a)
        db, Qb = generate_keypair(t, rng_b)
        sa = ecdh_shared_secret(t, da, Qb)
        sb = ecdh_shared_secret(t, db, Qa)
        if sa == sb and sa is not None:
            agree += 1
    check(f"ECDH: both parties derive the same secret ({trials} runs)", agree == trials,
          f"{agree}/{trials}")

    # ---- 6. ECDSA sign/verify --------------------------------------------------------
    good = 0
    tamper_rejected = 0
    wrongkey_rejected = 0
    mangled_rejected = 0
    trials = 40
    for seed in range(trials):
        rng = _LCG(seed * 101 + 3)
        d, Q = generate_keypair(t, rng)
        msg = f"transfer {seed} coins to alice".encode()
        sig = ecdsa_sign(t, d, msg, rng)
        if ecdsa_verify(t, Q, msg, sig):
            good += 1
        # tampered message
        if not ecdsa_verify(t, Q, msg + b"!", sig):
            tamper_rejected += 1
        # wrong public key
        d2, Q2 = generate_keypair(t, _LCG(seed * 999 + 7))
        if Q2 != Q and not ecdsa_verify(t, Q2, msg, sig):
            wrongkey_rejected += 1
        # mangled signature
        if not ecdsa_verify(t, Q, msg, (sig[0], (sig[1] + 1) % t.n)):
            mangled_rejected += 1
    check(f"ECDSA: valid signatures verify ({trials})", good == trials, f"{good}/{trials}")
    check("ECDSA: tampered message rejected", tamper_rejected == trials, f"{tamper_rejected}/{trials}")
    check("ECDSA: wrong key rejected", wrongkey_rejected == trials, f"{wrongkey_rejected}/{trials}")
    check("ECDSA: mangled signature rejected", mangled_rejected == trials, f"{mangled_rejected}/{trials}")

    # ---- 7. singular curve rejected ---------------------------------------------------
    try:
        Curve(a=0, b=0, p=17)   # discriminant 0
        check("singular curve rejected", False)
    except ValueError:
        check("singular curve rejected", True)

    # ---- 8. discrete-log one-wayness illustrated (brute force finds it on small curve)
    d, Q = generate_keypair(t, _LCG(42))
    found = None
    P = None
    for k in range(1, t.n + 1):
        P = t.add(P, t.G)
        if P == Q:
            found = k
            break
    check("brute-force discrete log recovers the private key (small curve)", found == d,
          f"found {found} vs {d}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
