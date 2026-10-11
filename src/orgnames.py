"""
orgnames.py: organisation names that mean the same thing ("UC Berkeley" = "University of California, Berkeley").

Two uses, both deterministic and small:
    organisation_variants(name)      other ways a page may write the name, for finding it in page text
    same_organisation(a, b)          two names that refer to the same organisation, for comparing names (for example a
                                     sourced institution against an OpenAlex affiliation)

The abbreviation table holds only widely used university abbreviations; anything else must match by words.
"""

from __future__ import annotations

import re
import unicodedata

STOP = {"of", "the", "at", "in", "and", "for", "de", "la", "du"}
GENERIC = {"university", "college", "institute", "technology", "school", "labs", "lab", "inc", "llc", "ltd", "corp",
           "company", "group", "center", "centre", "laboratory", "national", "state"}

# abbreviation -> its long form (tokens)
ABBREVIATIONS = {
    "uc": "university of california", "ut": "university of texas", "univ": "university", "u": "university",
    "mit": "massachusetts institute of technology", "nyu": "new york university", "cmu": "carnegie mellon university",
    "ucl": "university college london", "eth": "eth zurich", "usc": "university of southern california",
    "lab": "laboratory", "caltech": "california institute of technology", "tum": "technical university of munich",
}
# "University of <place>" systems that have a short prefix form: UC Berkeley, UT Austin
PREFIX_FORMS = {"california": "UC", "texas": "UT"}


def _tokens(name: str) -> list[str]:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold().replace("&", " and ")
    return re.findall(r"[a-z0-9]+", text)


def _core(name: str) -> set[str]:
    """Words that identify the organisation: abbreviations expanded, small words dropped."""
    words: list[str] = []
    for t in _tokens(name):
        words += ABBREVIATIONS.get(t, t).split()
    return {w for w in words if w not in STOP}


def _initials(name: str) -> str:
    return "".join(t[0] for t in _tokens(name) if t not in STOP)


def same_organisation(a: str, b: str) -> bool:
    """Equal word sets, or one name's words are all in the other's (and say more than a generic word like
    "university"), or one is the initials of the other ("MIT")."""
    ca, cb = _core(a), _core(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if small <= big and small - GENERIC:
        return True
    ta, tb = _tokens(a), _tokens(b)
    return (len(ta) == 1 and 2 <= len(ta[0]) <= 6 and ta[0] == _initials(b)) or \
           (len(tb) == 1 and 2 <= len(tb[0]) <= 6 and tb[0] == _initials(a))


def organisation_variants(name: str) -> list[str]:
    """The name itself plus the long form of a leading abbreviation ("UC Berkeley" -> "University of California,
    Berkeley") and the short form of a "University of <place>" name ("University of Texas at Austin" -> "UT Austin")."""
    name = " ".join(name.split())
    variants = [name]
    words = name.split()
    if words and words[0].lower() in ABBREVIATIONS:
        long = ABBREVIATIONS[words[0].lower()].title().replace(" Of ", " of ")
        rest = " ".join(words[1:])
        if rest:
            variants += [f"{long}, {rest}", f"{long} {rest}", f"{long} at {rest}"]
        else:
            variants.append(long)
    m = re.match(r"^(?:the\s+)?university of (\w+)(?:,|\s+at)?\s+(.+)$", name, re.I)
    if m and m.group(1).lower() in PREFIX_FORMS:
        variants.append(f"{PREFIX_FORMS[m.group(1).lower()]} {m.group(2)}")
    return list(dict.fromkeys(variants))
