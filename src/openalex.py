"""
openalex.py: a founder's publications from OpenAlex (https://api.openalex.org), called with the standard library.

For one founder: search /authors by name (and by each alias); accept an author only if TWO independent signals hold:
    affiliation        one of their affiliations matches an institution already sourced for that founder (name variants
                       allowed)
    second_affiliation a second, different affiliation also matches a sourced institution
    coauthor           a co-founder is a co-author of one of their works
    topic              most of the works kept have an OpenAlex field that matches the founder's sourced field or the
                       company's sector tag, through the fixed tables FIELD_KEYWORDS and SECTOR_FIELDS (code, never the
                       model's judgement)
A name match alone, or one overlap with a large institution alone, is not enough. Then keep up to 3 works (most cited,
then most recent). The reduced API response (the fields that matter) is returned
as text, so the workflow can store it as the source text and verify.py checks quotes against it as usual; claims from it
are origin `database`, claim type `technical`. At most MAX_REQUESTS requests per founder. The key is the line
openalex_api_key (any case) in .env; it is sent as a parameter and never logged or stored. OpenAlex is free with a key
within its daily allowance, so the logged cost per request is 0.

    find_works(name, institutions, coauthors) -> OpenAlexResult       never raises; errors come back in .error
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.llm import read_env_key
from src.orgnames import same_organisation

BASE = "https://api.openalex.org"
MAX_REQUESTS = 5
CANDIDATES = 5                 # authors looked at per search
KEEP_WORKS = 3
WORKS_PER_REQUEST = 10
WORKS_CHECKS = 3               # candidates whose works are fetched to look for the co-author and topic signals
REQUIRED_SIGNALS = 2
REQUEST_COST_USD = 0.0
TIMEOUT_SECONDS = 30

# OpenAlex "field" names (its 26 fields) that fit what a founder studied or did, from keywords in the founder's sourced
# education, technical, role and recognition claims. Fixed here; the model has no say.
FIELD_KEYWORDS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("computer science", "machine learning", "artificial intelligence", "computer vision", "software", "data science",
      "informatics", "deep learning", "natural language"), ("Computer Science",)),
    (("robotics", "mechanical", "electrical", "mechatronic", "aerospace", "civil", "engineering"), ("Engineering",)),
    (("nuclear",), ("Energy", "Engineering", "Physics and Astronomy")),
    (("physics", "astronomy"), ("Physics and Astronomy",)),
    (("chemistry",), ("Chemistry",)), (("chemical engineering",), ("Chemical Engineering",)),
    (("materials",), ("Materials Science",)),
    (("biology", "biotechnology", "genetic", "bioinformatics", "molecular", "biochemistry"),
     ("Biochemistry, Genetics and Molecular Biology", "Agricultural and Biological Sciences")),
    (("medicine", "medical", "clinical", "health"), ("Medicine", "Health Professions")),
    (("neuroscience",), ("Neuroscience",)), (("pharma",), ("Pharmacology, Toxicology and Pharmaceutics",)),
    (("economics", "finance"), ("Economics, Econometrics and Finance",)),
    (("business", "management", "mba", "entrepreneurship"), ("Business, Management and Accounting",)),
    (("mathematics", "statistics", "operations research"), ("Mathematics", "Decision Sciences")),
    (("energy",), ("Energy",)), (("environmental", "climate", "sustainab"), ("Environmental Science", "Earth and Planetary Sciences")),
    (("political", "sociology", "psychology", "public policy", "communication", "law", "polling"), ("Social Sciences", "Psychology")),
]
# Fields that fit each company sector tag.
SECTOR_FIELDS: dict[str, tuple[str, ...]] = {
    "ai_software": ("Computer Science",), "dev_infra": ("Computer Science",),
    "enterprise_software": ("Computer Science", "Business, Management and Accounting"),
    "fintech": ("Economics, Econometrics and Finance", "Computer Science", "Business, Management and Accounting"),
    "health_bio": ("Medicine", "Biochemistry, Genetics and Molecular Biology", "Neuroscience", "Health Professions",
                   "Pharmacology, Toxicology and Pharmaceutics"),
    "climate_energy": ("Energy", "Environmental Science", "Earth and Planetary Sciences", "Engineering", "Chemical Engineering"),
    "mobility_logistics": ("Engineering", "Computer Science", "Decision Sciences"),
    "aerospace_defense": ("Engineering", "Physics and Astronomy", "Computer Science"),
    "industrial_hardware": ("Engineering", "Materials Science", "Physics and Astronomy", "Computer Science"),
    "consumer": ("Computer Science", "Social Sciences", "Psychology"),
    "commerce_marketplaces": ("Business, Management and Accounting", "Economics, Econometrics and Finance", "Computer Science"),
    "education_work": ("Social Sciences", "Psychology", "Computer Science"),
    "real_estate_construction": ("Engineering", "Business, Management and Accounting", "Environmental Science"),
    "food_agri": ("Agricultural and Biological Sciences", "Environmental Science", "Chemical Engineering"),
}


def allowed_fields(founder_texts: list[str], sector_tags: list[str]) -> tuple[str, ...]:
    """OpenAlex fields that fit the founder's sourced background texts or the company's sector tags."""
    text = " ".join(founder_texts).casefold()
    fields: list[str] = []
    for keywords, names in FIELD_KEYWORDS:
        if any(k in text for k in keywords):
            fields += names
    for tag in sector_tags:
        fields += SECTOR_FIELDS.get(tag, ())
    return tuple(dict.fromkeys(fields))


# GET a URL and return the decoded JSON.
HttpGetJson = Callable[[str], dict]


def _http_get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "founder-dossiers/0.1 (research prototype)"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class Work:
    title: str
    year: Optional[int]
    date: str
    cited: int
    venue: str
    doi: str
    authors: list[str]
    field: str = ""                # the work's OpenAlex field (primary topic), e.g. "Computer Science"


@dataclass
class OpenAlexResult:
    requests: list[str] = field(default_factory=list)          # every request URL, without the key
    candidates: int = 0                                        # distinct authors returned by the searches
    name_mismatch: int = 0                                     # candidates whose name is not a variant of the founder's
    no_overlap: int = 0                                        # name matches with fewer than two signals
    queries: int = 0                                           # author searches run (the name, and each alias)
    accepted_by: str = ""                                      # the signals that decided it, e.g. "affiliation+topic"
    decisions: list[tuple[str, list[str], bool]] = field(default_factory=list)   # (author, signals, accepted) per name match
    author_id: str = ""
    author_name: str = ""
    institutions: list[str] = field(default_factory=list)      # OpenAlex's canonical names for the accepted author
    works: list[Work] = field(default_factory=list)
    source_url: str = ""
    source_text: str = ""
    error: Optional[str] = None

    @property
    def cost_usd(self) -> float:
        return REQUEST_COST_USD * len(self.requests)

    @property
    def rejected(self) -> int:
        return self.name_mismatch + self.no_overlap


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

def _words(name: str) -> list[str]:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    return re.findall(r"[a-z]+", text)


def search_query(name: str) -> str:
    """The author search string: dots in initials become spaces ("A.B. Carter" -> "A B Carter"); the search finds nothing
    for the dotted form."""
    return " ".join(re.sub(r"[.]", " ", name).split())


def same_person_name(a: str, b: str) -> bool:
    """Same last name and the same first name or first initial ("J. Doe" = "Jane Doe"). Middle names are ignored."""
    wa, wb = _words(a), _words(b)
    if len(wa) < 2 or len(wb) < 2 or wa[-1] != wb[-1]:
        return False
    return wa[0] == wb[0] or (len(wa[0]) == 1 and wb[0].startswith(wa[0])) or (len(wb[0]) == 1 and wa[0].startswith(wb[0]))


# --------------------------------------------------------------------------
# The flow
# --------------------------------------------------------------------------

class _Requester:
    def __init__(self, http_get: HttpGetJson, key: str, result: OpenAlexResult, limit: int):
        self.http_get, self.key, self.result, self.limit = http_get, key, result, limit

    def get(self, path: str, params: dict) -> Optional[dict]:
        if len(self.result.requests) >= self.limit:
            return None
        url = f"{BASE}{path}?{urllib.parse.urlencode(params, safe=':,|')}"
        self.result.requests.append(url)                         # logged without the key
        try:
            return self.http_get(url + f"&api_key={urllib.parse.quote(self.key)}")
        except urllib.error.HTTPError as e:
            e.close()
            self.result.error = f"HTTP {e.code}"
        except Exception as e:                                   # network errors, timeouts, bad JSON
            self.result.error = f"{type(e).__name__}: {e}"
        return None


def _institutions(author: dict) -> list[str]:
    names = [((a or {}).get("institution") or {}).get("display_name") for a in author.get("affiliations") or []]
    names += [(i or {}).get("display_name") for i in author.get("last_known_institutions") or []]
    return list(dict.fromkeys(n for n in names if n))


def _works(data: Optional[dict]) -> list[Work]:
    works = []
    for w in (data or {}).get("results") or []:
        location = (w.get("primary_location") or {}).get("source") or {}
        works.append(Work(
            title=w.get("display_name") or w.get("title") or "", year=w.get("publication_year"),
            date=w.get("publication_date") or "", cited=int(w.get("cited_by_count") or 0), venue=location.get("display_name") or "",
            doi=w.get("doi") or "",
            authors=[((a or {}).get("author") or {}).get("display_name") or "" for a in w.get("authorships") or []],
            field=(((w.get("primary_topic") or {}).get("field")) or {}).get("display_name") or ""))
    return [w for w in works if w.title]


def top_works(works: list[Work], n: int = KEEP_WORKS) -> list[Work]:
    """Most cited first, then most recent."""
    return sorted(works, key=lambda w: (w.cited, w.date), reverse=True)[:n]


def source_text(author_name: str, author_id: str, accepted_by: str, institutions: list[str], works: list[Work]) -> str:
    """The reduced response stored as the source text: only fields a claim may quote."""
    return json.dumps({
        "source": "OpenAlex", "author": {"id": author_id, "display_name": author_name, "institutions": institutions},
        "accepted_by": accepted_by,
        "works": [{"title": w.title, "year": w.year, "cited_by_count": w.cited, "venue": w.venue, "doi": w.doi,
                   "field": w.field, "authors": w.authors[:12]} for w in works]}, ensure_ascii=False, indent=1)


def find_works(name: str, institutions: list[str], coauthors: list[str], http_get: HttpGetJson = _http_get,
               key: Optional[str] = None, max_requests: int = MAX_REQUESTS, aliases: tuple[str, ...] = (),
               fields: tuple[str, ...] = ()) -> OpenAlexResult:
    """The accepted author's top works for one founder, or an empty result. `aliases` are other names the founder goes
    by (the name in brackets in the input): each is searched too, and a name match is a match with any of them.
    `fields` are the OpenAlex fields that fit the founder's sourced background and the company's sector (allowed_fields).
    An author is accepted only with two independent signals (see the module docstring)."""
    out = OpenAlexResult()
    try:
        key = key or read_env_key("OPENALEX_API_KEY")
    except RuntimeError as e:
        out.error = str(e)
        return out
    api = _Requester(http_get, key, out, max_requests)
    names = list(dict.fromkeys([name, *aliases]))
    candidates, seen = [], set()
    for n in names:
        data = api.get("/authors", {"search": search_query(n), "per-page": CANDIDATES,
                                    "select": "id,display_name,display_name_alternatives,works_count,affiliations,last_known_institutions"})
        out.queries += 1
        for a in (data or {}).get("results") or []:
            if a.get("id") not in seen:
                seen.add(a.get("id"))
                candidates.append(a)
    out.candidates = len(candidates)
    named = []                                               # candidates whose name is a variant of one of the founder's names
    for a in candidates:
        variants = [a.get("display_name") or ""] + list(a.get("display_name_alternatives") or [])
        if any(same_person_name(v, n) for v in variants for n in names):
            named.append(a)
    out.name_mismatch = len(candidates) - len(named)

    def matched(a: dict) -> list[str]:
        return [i for i in _institutions(a) if any(same_organisation(i, s) for s in institutions)]

    ranked = sorted(named, key=lambda a: (len(matched(a)), a.get("works_count") or 0), reverse=True)
    accepted, accepted_works, accepted_signals = None, [], []
    for position, a in enumerate(ranked):
        signals = []
        overlap = matched(a)
        if overlap:
            signals.append("affiliation")
        if len(overlap) >= 2:
            signals.append("second_affiliation")
        works = []
        if position < WORKS_CHECKS:                          # the works give the co-author and topic signals
            works = _works(api.get("/works", _works_params(a)))
            if any(same_person_name(co, c) for w in works for co in w.authors for c in coauthors):
                signals.append("coauthor")
            kept = top_works(works)
            if fields and kept and sum(w.field in fields for w in kept) * 2 > len(kept):
                signals.append("topic")
        ok = len(signals) >= REQUIRED_SIGNALS
        out.decisions.append((a.get("display_name") or "", signals, ok))
        if ok:
            accepted, accepted_works, accepted_signals = a, works, signals
            break                                            # no more requests once an author is accepted
    out.no_overlap = len(named) - (1 if accepted else 0)
    if accepted is None:
        return out
    out.accepted_by, out.author_id = "+".join(accepted_signals), _short_id(accepted.get("id") or "")
    out.author_name, out.institutions = accepted.get("display_name") or name, _institutions(accepted)
    out.works = top_works(accepted_works)
    if out.works:
        out.source_url = f"{BASE}/works?filter=authorships.author.id:{out.author_id}&sort=cited_by_count:desc"
        out.source_text = source_text(out.author_name, out.author_id, out.accepted_by, out.institutions, out.works)
    return out


def _short_id(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _works_params(author: dict) -> dict:
    return {"filter": f"authorships.author.id:{_short_id(author.get('id') or '')}", "sort": "cited_by_count:desc",
            "per-page": WORKS_PER_REQUEST,
            "select": "id,display_name,publication_year,publication_date,cited_by_count,doi,primary_location,authorships,primary_topic"}
