"""
Clean, High-Precision Feature Engineering for Business Entity Resolution.
Extracts 17 dense similarity metrics designed for noisy multi-source records:
- Multi-token name similarity (Levenshtein, Partial, Token Sort, Token Set, Jaro-Winkler)
- Substring & concatenation matching (e.g., 'Prime Money' vs 'primemoney.com')
- Missing address resilience (avoids penalizing exact name matches with blank S2/S3 addresses)
- Exact numeric address overlap (PIN codes, house numbers, plot numbers)
- Name-address interaction modeling
"""

import re
import unicodedata
from typing import Dict, List, Set, Any
from rapidfuzz import fuzz, distance

def normalize_string(text: str) -> str:
    """Decomposes Unicode accents and standardizes spacing."""
    if not text:
        return ""
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"\.(com|org|net|in|co|io|fr|us|biz|info)\b", " ", text, flags=re.IGNORECASE)
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(clean.split())

def extract_numbers(text: str) -> Set[str]:
    """Extracts all numeric sequences >= 2 digits (building numbers, postal codes)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d{2,}\b", text))

def extract_pair_features(s1: Dict[str, Any], cand: Dict[str, Any]) -> List[float]:
    """Computes dense feature vector for a candidate pair."""
    s1_name_raw = s1.get("business_name", "")
    c_name_raw = cand.get("business_name", "")
    s1_addr_raw = s1.get("business_address", "")
    c_addr_raw = cand.get("business_address", "")
    c_id = cand.get("entity_id", "")

    s1_name = normalize_string(s1_name_raw)
    c_name = normalize_string(c_name_raw)
    s1_addr = normalize_string(s1_addr_raw)
    c_addr = normalize_string(c_addr_raw)

    # 1. Name String Metrics
    n_lev = fuzz.ratio(s1_name, c_name) / 100.0
    n_part = fuzz.partial_ratio(s1_name, c_name) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    n_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    n_jw = distance.JaroWinkler.similarity(s1_name, c_name)

    # 2. Token Jaccard & Substring
    s1_words = set(s1_name.split())
    c_words = set(c_name.split())
    n_wjac = len(s1_words & c_words) / max(1, len(s1_words | c_words))
    n_sub = 1.0 if (s1_name and c_name and (s1_name in c_name or c_name in s1_name)) else 0.0

    # 3. Concatenation match (e.g. 'Prime Money' vs 'primemoney')
    s1_concat = s1_name.replace(" ", "")
    c_concat = c_name.replace(" ", "")
    concat_match = 1.0 if (s1_concat and c_concat and (s1_concat == c_concat or s1_concat in c_concat or c_concat in s1_concat)) else 0.0

    # 4. Length Ratio
    max_len = max(len(s1_name), len(c_name), 1)
    min_len = min(len(s1_name), len(c_name))
    n_len_ratio = min_len / max_len

    # 5. Address Metrics
    addr_empty = 1.0 if len(c_addr.strip()) < 5 else 0.0
    if addr_empty:
        a_lev = 0.0
        a_sort = 0.0
        a_set = 0.0
        num_inter = 0.0
        num_jac = 0.0
        inter = n_set
    else:
        a_lev = fuzz.ratio(s1_addr, c_addr) / 100.0
        a_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        a_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0

        s1_nums = extract_numbers(s1_addr_raw)
        c_nums = extract_numbers(c_addr_raw)
        shared_nums = s1_nums & c_nums
        all_nums = s1_nums | c_nums
        num_inter = float(len(shared_nums))
        num_jac = len(shared_nums) / max(1, len(all_nums))
        inter = n_set * a_set

    is_s2 = 1.0 if c_id.startswith("S2-") else 0.0

    return [
        n_lev,
        n_part,
        n_sort,
        n_set,
        n_jw,
        n_wjac,
        n_sub,
        concat_match,
        n_len_ratio,
        addr_empty,
        a_lev,
        a_sort,
        a_set,
        num_inter,
        num_jac,
        inter,
        is_s2
    ]
