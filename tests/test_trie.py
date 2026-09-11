"""Tests for trie: insert/search, prefix, autocomplete, delete-with-pruning, suffix substring index."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trie import Trie, SuffixTrie

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- insert / search --------------------------------------------------------
t = Trie()
for w in ["cat", "car", "card", "care", "dog", "do"]:
    t.insert(w)
check("stored word is found", t.search("cat"))
check("a prefix that is not a word is not found", not t.search("ca"))
check("absent word not found", not t.search("cab"))
check("__contains__ works", "dog" in t and "cab" not in t)
check("distinct-word count", len(t) == 6)

# --- prefix membership -----------------------------------------------------
check("starts_with a real prefix", t.starts_with("ca"))
check("starts_with a full word", t.starts_with("card"))
check("starts_with a missing prefix is false", not t.starts_with("xyz"))
check("empty prefix matches (non-empty trie)", t.starts_with(""))

# --- autocomplete -----------------------------------------------------------
check("autocomplete under a prefix", t.autocomplete("car") == ["car", "card", "care"])
check("autocomplete alphabetical", t.autocomplete("c") == ["car", "card", "care", "cat"])
check("autocomplete of a missing prefix is empty", t.autocomplete("z") == [])
check("autocomplete respects a limit", t.autocomplete("c", limit=2) == ["car", "card"])
check("words() returns all, sorted", t.words() == ["car", "card", "care", "cat", "do", "dog"])

# --- insertion frequency ranking -------------------------------------------
t.insert("cat")
t.insert("cat")           # cat now inserted 3x
check("count tracks insertions", t.count("cat") == 3)
check("count of a prefix-only string is 0", t.count("ca") == 0)
check("by-frequency ranking puts the most-inserted first",
      t.autocomplete("c", by_frequency=True)[0] == "cat")
check("frequency ranking does not change the set",
      sorted(t.autocomplete("c", by_frequency=True)) == ["car", "card", "care", "cat"])

# --- longest-prefix matching (IP-routing style) ----------------------------
check("longest stored prefix of a string", t.longest_prefix_of("dogma") == "dog")
check("longest prefix with a shorter match", t.longest_prefix_of("cats") == "cat")
check("no stored prefix returns empty", t.longest_prefix_of("zebra") == "")

# --- deletion with pruning --------------------------------------------------
check("delete returns True for a present word", t.delete("card"))
check("deleted word is gone", not t.search("card"))
check("sibling words survive deletion", t.search("car") and t.search("care"))
check("count drops after delete", len(t) == 5)
check("delete of an absent word returns False", not t.delete("card"))
check("delete of a mere prefix returns False", not t.delete("ca"))
# deleting a word that is a prefix of another keeps the longer word
check("deleting a prefix-word keeps the longer word", t.delete("do") and t.search("dog"))
check("the shorter word is gone", not t.search("do"))

# --- pruning actually removes dead branches (whitebox) ---------------------
t2 = Trie()
t2.insert("abc")
t2.delete("abc")
check("pruning removes the whole dead branch", "a" not in t2.root.children)
check("empty trie after deleting its only word", len(t2) == 0)

# --- re-insertion after deletion -------------------------------------------
t2.insert("abc")
check("re-insert after delete works", t2.search("abc") and len(t2) == 1)

# --- suffix trie substring index -------------------------------------------
st = SuffixTrie("banana")
check("substring present", st.contains("ana"))
check("substring 'nan' present", st.contains("nan"))
check("full text is a substring", st.contains("banana"))
check("single char present", st.contains("a"))
check("absent substring", not st.contains("xyz"))
check("empty pattern is trivially present", st.contains(""))
check("occurrences of 'ana'", st.occurrences("ana") == [1, 3])
check("occurrences of 'a'", st.occurrences("a") == [1, 3, 5])
check("occurrences of a missing pattern", st.occurrences("z") == [])

# --- a trie of many words handles a shared deep prefix ---------------------
t3 = Trie()
deep = ["a" * i for i in range(1, 11)]   # "a", "aa", ..., "aaaaaaaaaa"
for w in deep:
    t3.insert(w)
check("nested prefixes all stored", all(t3.search(w) for w in deep))
check("autocomplete over nested prefixes", len(t3.autocomplete("a")) == 10)
t3.delete("aaaaa")
check("deleting a middle nested word keeps neighbours",
      t3.search("aaaa") and t3.search("aaaaaa") and not t3.search("aaaaa"))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all trie tests passed")
