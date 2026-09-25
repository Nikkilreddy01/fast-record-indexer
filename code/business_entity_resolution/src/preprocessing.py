"""
Text Preprocessing and Standardization Module for Entity Resolution.
Handles legal suffixes, address abbreviations, landmark patterns, and multilingual noise.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple

# Comprehensive legal suffixes across jurisdictions (US, India, UK, France, Germany, global)
LEGAL_PATTERNS = [
    r"\bprivate\s+limited\b",
    r"\bpvt\s*\.?\s*ltd\s*\.?",
    r"\bltd\s*\.?",
    r"\blimited\b",
    r"\bincorporated\b",
    r"\binc\s*\.?",
    r"\bcorporation\b",
    r"\bcorp\s*\.?",
    r"\bllc\s*\.?",
    r"\bl\.l\.c\s*\.?",
    r"\bllp\s*\.?",
    r"\bl\.l\.p\s*\.?",
    r"\bplc\s*\.?",
    r"\bco\s*\.?\s*ltd\s*\.?",
    r"\bcompany\b",
    r"\bco\s*\.?",
    r"\bsarl\b",
    r"\bsas\b",
    r"\bsa\b",
    r"\bgmbh\b",
    r"\bag\b",
    r"\benterprises?\b",
    r"\bventures?\b",
    r"\bholdings?\b",
    r"\bgroup\b",
    r"\bservices?\b",
    r"\btechnologies?\b",
    r"\bsolutions?\b",
    r"\bindustries?\b",
    r"\blabs?\b",
]

# Standard address abbreviation mappings
ADDRESS_ABBREVIATIONS = {
    r"\bst\b|\bst\.": "street",
    r"\brd\b|\brd\.": "road",
    r"\bave\b|\bave\.|\bav\b|\bav\.": "avenue",
    r"\bblvd\b|\bblvd\.": "boulevard",
    r"\bdr\b|\bdr\.": "drive",
    r"\bln\b|\bln\.": "lane",
    r"\bhwy\b|\bhwy\.": "highway",
    r"\bct\b|\bct\.": "court",
    r"\bpl\b|\bpl\.": "place",
    r"\bsq\b|\bsq\.": "square",
    r"\bfl\b|\bfl\.|\bflr\b": "floor",
    r"\bste\b|\bste\.": "suite",
    r"\bapt\b|\bapt\.": "apartment",
    r"\bbldg\b|\bbldg\.": "building",
    r"\bopp\b|\bopp\.|\bopposite\b": "opposite",
    r"\bnr\b|\bnr\.|\bnear\b": "near",
    r"\badj\b|\badj\.|\badjacent\b": "adjacent",
    r"\bbeside\b|\bbehind\b": "near",
    r"\bpk\b|\bpk\.|\bpark\b": "park",
    r"\bpkwy\b|\bpkwy\.": "parkway",
}

def strip_accents(text: str) -> str:
    """Normalize unicode characters (e.g. é -> e, ô -> o)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def clean_text(text: str) -> str:
    """Basic low-level cleaning and punctuation handling."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text.lower().strip())
    # Standardize ampersand
    text = re.sub(r"\s*&\s*", " and ", text)
    # Replace non-alphanumeric (except commas and hyphens temporarily)
    text = re.sub(r"[^\w\s\-,]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_core_name(name: str) -> Tuple[str, str]:
    """
    Returns (normalized_full_name, core_name_without_legal_suffix).
    """
    cleaned = clean_text(name)
    core = cleaned
    for pattern in LEGAL_PATTERNS:
        core = re.sub(pattern, "", core, flags=re.IGNORECASE)
    # Clean up resulting spaces and trailing punctuation
    core = re.sub(r"[\s\-,]+$", "", core)
    core = re.sub(r"^[\s\-,]+", "", core)
    core = re.sub(r"\s+", " ", core).strip()
    return cleaned, core if core else cleaned

def clean_address(address: str) -> Tuple[str, List[str]]:
    """
    Cleans address, standardizes abbreviations, and extracts numerical tokens (building numbers, postal codes).
    Returns (cleaned_address, list_of_number_strings).
    """
    cleaned = clean_text(address)
    # Standardize abbreviations
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Extract all numbers/digits (e.g. '500', '95113', '400001', '17')
    numbers = re.findall(r"\b\d+\b", cleaned)
    return cleaned, numbers

def preprocess_record(record: Dict[str, str]) -> Dict:
    """
    Enriches a raw record with preprocessed name, core name, cleaned address, and numbers.
    """
    raw_name = record.get("business_name", "")
    raw_addr = record.get("business_address", "")
    country = str(record.get("country", "")).strip()

    cleaned_name, core_name = extract_core_name(raw_name)
    cleaned_addr, addr_numbers = clean_address(raw_addr)

    # First significant word (ignoring stop words like 'the', 'a', etc.)
    words = [w for w in core_name.split() if w not in {"the", "a", "an", "all", "new", "global", "national"}]
    first_word = words[0] if words else (core_name.split()[0] if core_name else "")

    return {
        "entity_id": record["entity_id"],
        "raw_name": raw_name,
        "raw_address": raw_addr,
        "country": country,
        "cleaned_name": cleaned_name,
        "core_name": core_name,
        "first_word": first_word,
        "first_3_chars": core_name[:3] if len(core_name) >= 3 else core_name,
        "cleaned_address": cleaned_addr,
        "addr_numbers": set(addr_numbers),
    }
