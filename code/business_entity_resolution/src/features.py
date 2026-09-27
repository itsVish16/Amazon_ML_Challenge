"""
Fast Feature Extraction Module for Entity Matching Pairs.
Includes:
  - RapidFuzz string metrics (Levenshtein, Token Sort, Token Set, Partial)
  - Character n-gram Jaccard
  - Conflict signals (house number conflict, state conflict)
  - Cross-field signals (domain match, dual-anchor indicators)
"""

from rapidfuzz import fuzz

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    if not intersection:
        return 0.0
    return intersection / len(set_a | set_b)


def char_ngrams(text: str, n: int = 3) -> set:
    """Generate character n-grams from a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def compute_pair_features(s1_norm: dict, cand_norm: dict, cand_id: str) -> list:
    """
    Compute dense feature vector between a Source 1 entity and a Candidate entity.
    """
    s1_n = s1_norm['name']
    c_n = cand_norm['name']
    s1_name = s1_n['clean_name']
    c_name = c_n['clean_name']
    s1_tokens = set(s1_n['tokens'])
    c_tokens = set(c_n['tokens'])

    # Name features
    name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    name_sort_ratio = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_set_ratio = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_jaccard = jaccard_similarity(s1_tokens, c_tokens)

    # Character 3-gram Jaccard for names (catches transliteration variations)
    s1_ng = char_ngrams(s1_name, 3)
    c_ng = char_ngrams(c_name, 3)
    name_char_jaccard = jaccard_similarity(s1_ng, c_ng)

    # Exact name indicators
    name_exact = 1.0 if s1_name and s1_name == c_name else 0.0

    # Domain match check (e.g., siiainvestments.com matching Siia Investments)
    domain_match = 0.0
    s1_domain = s1_n.get('domain_base', '')
    c_domain = c_n.get('domain_base', '')
    s1_compact = s1_n.get('compact_name', '')
    c_compact = c_n.get('compact_name', '')

    if c_domain and (c_domain in s1_compact or s1_compact in c_domain):
        domain_match = 1.0
    elif s1_domain and (s1_domain in c_compact or c_compact in s1_domain):
        domain_match = 1.0

    # Address features
    s1_a = s1_norm['addr']
    c_a = cand_norm['addr']
    s1_addr = s1_a['clean_addr']
    c_addr = c_a['clean_addr']
    s1_addr_tokens = set(s1_a['tokens'])
    c_addr_tokens = set(c_a['tokens'])

    cand_addr_empty = 1.0 if not c_addr else 0.0
    s1_addr_empty = 1.0 if not s1_addr else 0.0

    if s1_addr and c_addr:
        addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
        addr_sort_ratio = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        addr_set_ratio = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
        addr_jaccard = jaccard_similarity(s1_addr_tokens, c_addr_tokens)
    else:
        addr_ratio = 0.0
        addr_sort_ratio = 0.0
        addr_set_ratio = 0.0
        addr_jaccard = 0.0

    # Number agreement & Conflict features
    s1_nums = set(s1_a['numbers'])
    c_nums = set(c_a['numbers'])

    if s1_nums and c_nums:
        nums_overlap = jaccard_similarity(s1_nums, c_nums)
        s1_prim = s1_a['primary_number']
        c_prim = c_a['primary_number']
        prim_num_match = 1.0 if (s1_prim and s1_prim == c_prim) else 0.0
        num_conflict = 1.0 if not (s1_nums & c_nums) else 0.0
    elif not s1_nums and not c_nums:
        nums_overlap = 0.5
        prim_num_match = 0.5
        num_conflict = 0.0
    else:
        nums_overlap = 0.0
        prim_num_match = 0.0
        num_conflict = 0.0

    # State agreement & Conflict
    s1_state = s1_a['canonical_state']
    c_state = c_a['canonical_state']
    if s1_state and c_state:
        state_match = 1.0 if s1_state == c_state else -1.0
        state_conflict = 1.0 if s1_state != c_state else 0.0
    else:
        state_match = 0.0
        state_conflict = 0.0

    # Source ID indicator (S2 vs S3)
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0

    # Composite maximum heuristic score
    max_sim = max(name_set_ratio, addr_sort_ratio)

    return [
        name_ratio,
        name_sort_ratio,
        name_set_ratio,
        name_partial_ratio,
        name_jaccard,
        name_char_jaccard,
        name_exact,
        domain_match,
        addr_ratio,
        addr_sort_ratio,
        addr_set_ratio,
        addr_jaccard,
        nums_overlap,
        prim_num_match,
        num_conflict,
        state_match,
        state_conflict,
        cand_addr_empty,
        s1_addr_empty,
        is_s2,
        max_sim
    ]


FEATURE_NAMES = [
    'name_ratio',
    'name_sort_ratio',
    'name_set_ratio',
    'name_partial_ratio',
    'name_jaccard',
    'name_char_jaccard',
    'name_exact',
    'domain_match',
    'addr_ratio',
    'addr_sort_ratio',
    'addr_set_ratio',
    'addr_jaccard',
    'nums_overlap',
    'prim_num_match',
    'num_conflict',
    'state_match',
    'state_conflict',
    'cand_addr_empty',
    's1_addr_empty',
    'is_s2',
    'max_sim'
]
