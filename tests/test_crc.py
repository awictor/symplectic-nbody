"""Tests for crc.py -- cyclic redundancy checks.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. CRC values are checked against
the published "123456789" check values and Python's zlib.crc32.
"""

import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import crc  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


CHECK = b"123456789"

# --- published check values -------------------------------------------------
check("CRC-32 of '123456789' is 0xCBF43926", crc.crc_named(CHECK, "CRC-32") == 0xCBF43926)
check("CRC-16-CCITT of '123456789' is 0x29B1", crc.crc_named(CHECK, "CRC-16-CCITT") == 0x29B1)
check("CRC-8 of '123456789' is 0xF4", crc.crc_named(CHECK, "CRC-8") == 0xF4)

# --- agreement with the standard library -----------------------------------
check("CRC-32 matches zlib.crc32 on the check string", crc.crc_named(CHECK, "CRC-32") == zlib.crc32(CHECK))
for msg in (b"", b"a", b"hello world", b"The quick brown fox", bytes(range(256))):
    check(f"CRC-32 matches zlib for {msg[:12]!r}...", crc.crc_named(msg, "CRC-32") == zlib.crc32(msg))

# --- basic properties -------------------------------------------------------
check("CRC width matches the parameter set", crc.crc_named(CHECK, "CRC-8") <= 0xFF)
check("CRC-16 fits in 16 bits", crc.crc_named(CHECK, "CRC-16-CCITT") <= 0xFFFF)
check("identical messages give identical CRCs",
      crc.crc_named(b"repeat", "CRC-32") == crc.crc_named(b"repeat", "CRC-32"))
check("different messages usually give different CRCs",
      crc.crc_named(b"aaaa", "CRC-32") != crc.crc_named(b"aaab", "CRC-32"))
try:
    crc.crc_named(b"x", "CRC-999")
    check("rejects unknown CRC name", False)
except ValueError:
    check("rejects unknown CRC name", True)

# --- reflection helper ------------------------------------------------------
check("reflect of 0b0000_0001 over 8 bits is 0b1000_0000", crc._reflect(0x01, 8) == 0x80)
check("reflect is an involution", crc._reflect(crc._reflect(0xABCD, 16), 16) == 0xABCD)

# --- frame append / check ---------------------------------------------------
for name in ("CRC-8", "CRC-16-CCITT", "CRC-32"):
    frame = crc.append_crc(b"payload data", name)
    check(f"a clean frame passes its check ({name})", crc.check_frame(frame, name))
    nbytes = crc.PARAMS[name]["width"] // 8
    check(f"append adds exactly {nbytes} CRC bytes ({name})",
          len(frame) == len(b"payload data") + nbytes)
    # corrupt one byte -> check fails
    bad = bytearray(frame)
    bad[0] ^= 0xFF
    check(f"a corrupted frame fails its check ({name})", not crc.check_frame(bytes(bad), name))
check("check_frame rejects a too-short frame", not crc.check_frame(b"\x00", "CRC-32"))

# --- error-detection guarantees --------------------------------------------
data = b"Cyclic redundancy check!"
# every single-bit flip must be detected (CRCs always catch single-bit errors)
all_single = all(crc.detects_error(data, crc.flip_bit(data, i), "CRC-32")
                 for i in range(len(data) * 8))
check("every single-bit error is detected (CRC-32)", all_single)
all_single16 = all(crc.detects_error(data, crc.flip_bit(data, i), "CRC-16-CCITT")
                   for i in range(len(data) * 8))
check("every single-bit error is detected (CRC-16)", all_single16)
# a double-bit flip is (almost always) detected too
two = crc.flip_bit(crc.flip_bit(data, 3), 20)
check("a double-bit error is detected", crc.detects_error(data, two, "CRC-32"))
# flip_bit really changes exactly one bit
flipped = crc.flip_bit(data, 5)
diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(data, flipped))
check("flip_bit changes exactly one bit", diff_bits == 1)
# an unchanged message is "not detected" as an error (CRCs equal)
check("no error means CRCs match", not crc.detects_error(data, data, "CRC-32"))

# --- a burst error shorter than the width is caught -------------------------
# flip a contiguous run of bits (a burst) within one region
burst = bytearray(data)
burst[2] ^= 0b0111_0000  # a 3-bit burst well under width 32
check("a short burst error is detected", crc.detects_error(data, bytes(burst), "CRC-32"))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall crc tests passed")
