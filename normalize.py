"""
Text normalization for business names and addresses.

Every rule here is country-agnostic, so France (absent from training)
goes through exactly the same path as the US and India.

Patterns handled (all seen in the Phase 1 examples):
  - literal "null" / "none" / "n/a" inside fields
  - accents (é -> e); non-Latin scripts are dropped
  - website domains and social handles as names ("abc.com", "@abc")
  - former / trade names ("X formerly: Y", "X dba Y")
  - legal suffixes, including typos ("Pdivate", "Limted")
  - "&" vs "and", punctuation, hyphens, brackets, extra spaces
  - digits used as letters ("C0mmunity"), 'the' in any position
  - zero-padded house numbers ("003808" -> "3808")
  - address abbreviations (St/Street/Saint, Rd/Road, Nr/Near, ...)
  - Indian-script words ("லிமிடெட்" -> "limited") via a dictionary learned
    from the training data by script_dict.py (work/script_dict.json)
"""
import json
import os
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz

# Indian scripts: Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil,
# Telugu, Kannada, Malayalam, Sinhala (U+0900 to U+0DFF)
INDIC_RE = re.compile(r"[\u0900-\u0DFF]")
SCRIPT_TOKEN_SPLIT_RE = re.compile(r"[^\w\u0900-\u0DFF\u200c\u200d]+")

_DICT_PATH = Path(os.environ.get(
    "SCRIPT_DICT", Path(__file__).resolve().parents[1] / "work" / "script_dict.json"))
SCRIPT_DICT = (json.loads(_DICT_PATH.read_text(encoding="utf-8"))
               if _DICT_PATH.exists() else {})

NULL_TOKENS_RE = re.compile(r"\b(?:null|none|nan|n/a)\b", re.IGNORECASE)

LEGAL = {
    # English / India
    "pvt", "private", "ltd", "limited", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llc", "llp", "lp", "plc", "pc", "pllc",
    # French
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "societe",
}
# Longer legal words that often appear with typos; matched fuzzily
LEGAL_LONG = ["private", "limited", "corporation", "incorporated", "company", "societe"]
FUZZY_LEGAL_THRESHOLD = 84

ALT_NAME_RE = re.compile(
    r"\b(?:formerly known as|formerly|f/k/a|fka|d/b/a|dba|a/k/a|aka|trading as|t/a)\b\s*:?"
)
WEB_PREFIX_RE = re.compile(r"^(?:https?://)?(?:www\.)?@?")
WEB_SUFFIX_RE = re.compile(r"\.(?:com|net|org|in|co|us|fr|biz|info|io)(?:\.[a-z]{2})?$")
DIGITS_RE = re.compile(r"\d+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# Digits used as letters inside a word ("c0mmunity" -> "community")
LEET_RE = re.compile(r"(?<=[a-z])[013457]+(?=[a-z])")
LEET_MAP = str.maketrans("013457", "oieast")

ADDR_ABBR = {
    # St / Street / Saint are used interchangeably by the noise, so map to one token
    "st": "st", "str": "st", "street": "st", "saint": "st",
    "rd": "rd", "road": "rd",
    "ave": "ave", "av": "ave", "avenue": "ave",
    "blvd": "blvd", "bd": "blvd", "bvd": "blvd", "boulevard": "blvd",
    "dr": "dr", "drive": "dr",
    "ln": "ln", "lane": "ln",
    "ct": "ct", "court": "ct",
    "pl": "pl", "place": "pl",
    "hwy": "hwy", "highway": "hwy",
    "pkwy": "pkwy", "parkway": "pkwy",
    "apt": "apt", "apartment": "apt",
    "bldg": "bldg", "building": "bldg",
    "fl": "fl", "flr": "fl", "floor": "fl",
    "nr": "nr", "near": "nr",
    "opp": "opp", "opposite": "opp",
    "sec": "sector", "sector": "sector",
    "n": "n", "north": "n", "s": "s", "south": "s",
    "e": "e", "east": "e", "w": "w", "west": "w",
}

OUTPUT_COLUMNS = [
    "name_main",     # core name: legal suffixes, 'the', web decorations removed
    "name_alt",      # alternate name after 'formerly'/'dba'/'aka' (often empty)
    "name_all",      # every normalized name token, nothing removed
    "name_nospace",  # name_main with spaces removed ("bnp group" -> "bnpgroup")
    "name_is_web",   # 1 if the name looked like a domain or handle
    "addr_norm",     # normalized address with canonical abbreviations
    "addr_nums",     # sorted unique numbers in the address
    "addr_missing",  # 1 if the address is empty
]


def split_script_tokens(s):
    """Split on punctuation/space while keeping Indian-script words whole."""
    return [t for t in SCRIPT_TOKEN_SPLIT_RE.split(s) if t]


def translate_script(s):
    """Replace known Indian-script words with their English equivalents."""
    if not SCRIPT_DICT or not INDIC_RE.search(s):
        return s
    return " ".join(SCRIPT_DICT.get(t, t) for t in split_script_tokens(s))


def clean_raw(s):
    if s is None:
        return ""
    return NULL_TOKENS_RE.sub(" ", str(s)).strip()


def to_ascii_lower(s):
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode().lower()


def tokenize(s):
    s = s.replace("&", " and ").replace("'", "").replace(".", "")
    return NON_ALNUM_RE.sub(" ", s).split()


def _strip_web(part):
    p = part.strip()
    is_web = False
    for pattern in (WEB_PREFIX_RE, WEB_SUFFIX_RE):
        new = pattern.sub("", p)
        if new != p:
            is_web, p = True, new
    return p, is_web


def _fuzzy_legal(tok):
    return len(tok) >= 5 and any(
        fuzz.ratio(tok, w) >= FUZZY_LEGAL_THRESHOLD for w in LEGAL_LONG
    )


def _core_tokens(toks):
    n = len(toks)
    kept = [
        t for i, t in enumerate(toks)
        if not (t in LEGAL or (i > 0 and i >= n - 2 and _fuzzy_legal(t)))
    ]
    if len(kept) > 1:
        kept = [t for t in kept if t != "the"] or kept  # "Dent Seafood The"
    return kept or toks  # never return an empty name


def normalize_name(raw):
    s = to_ascii_lower(translate_script(clean_raw(raw)))
    parts = [p for p in ALT_NAME_RE.split(s) if p.strip()] or [s]
    is_web = False
    cores, all_toks = [], []
    for part in parts:
        part, web = _strip_web(part)
        is_web = is_web or web
        toks = [LEET_RE.sub(lambda m: m.group().translate(LEET_MAP), t)
                for t in tokenize(part)]
        all_toks.extend(toks)
        core = _core_tokens(toks)
        if core:
            cores.append(" ".join(core))
    main = cores[0] if cores else ""
    alt = cores[1] if len(cores) > 1 else ""
    return main, alt, " ".join(all_toks), main.replace(" ", ""), int(is_web)


def _strip_zeros(t):
    return (t.lstrip("0") or "0") if t.isdigit() else t


def normalize_address(raw):
    s = to_ascii_lower(translate_script(clean_raw(raw)))
    addr = " ".join(_strip_zeros(ADDR_ABBR.get(t, t)) for t in tokenize(s))
    nums = " ".join(sorted({_strip_zeros(n) for n in DIGITS_RE.findall(addr)}, key=int))
    return addr, nums, int(not addr)


def normalize_record(name_and_address):
    """(name, address) -> tuple in OUTPUT_COLUMNS order. Top-level so multiprocessing can use it."""
    name, address = name_and_address
    return normalize_name(name) + normalize_address(address)