"""
Expanded Discriminative Feature Extraction for Entity Matching.
24 features designed to discriminate:
  - Strip mall false merges (different businesses at same address)
  - Chain store false merges (same business name at different addresses)
  - Transliteration variations (Indic scripts vs Latin)
  - Address typos and format variations
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
    Compute 24-dimensional dense feature vector between Source 1 entity and Candidate.
    """
    s1_n = s1_norm['name']
    c_n = cand_norm['name']
    s1_name = s1_n['clean_name']
    c_name = c_n['clean_name']
    s1_tokens = set(s1_n['tokens'])
    c_tokens = set(c_n['tokens'])

    # === NAME FEATURES (10) ===
    name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    name_sort_ratio = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_set_ratio = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_jaccard = jaccard_similarity(s1_tokens, c_tokens)

    # Character 3-gram Jaccard (catches transliteration variations)
    s1_ng = char_ngrams(s1_name, 3)
    c_ng = char_ngrams(c_name, 3)
    name_char_jaccard = jaccard_similarity(s1_ng, c_ng)

    # Exact name match
    name_exact = 1.0 if s1_name and s1_name == c_name else 0.0

    # NEW: First meaningful token match (catches "Novak Quinn" vs "Barbara Quinn")
    s1_ftokens = s1_n['tokens']
    c_ftokens = c_n['tokens']
    first_token_match = 0.0
    if s1_ftokens and c_ftokens:
        first_token_match = fuzz.ratio(s1_ftokens[0], c_ftokens[0]) / 100.0

    # NEW: Name token coverage (what fraction of S1 tokens appear in candidate)
    name_token_coverage = 0.0
    if s1_tokens and c_tokens:
        name_token_coverage = len(s1_tokens & c_tokens) / len(s1_tokens)

    # NEW: Name length ratio (catches over-stripped suffixes)
    name_len_ratio = 0.0
    if s1_name and c_name:
        shorter = min(len(s1_name), len(c_name))
        longer = max(len(s1_name), len(c_name))
        name_len_ratio = shorter / longer if longer > 0 else 0.0

    # === DOMAIN FEATURE (1) ===
    domain_match = 0.0
    s1_domain = s1_n.get('domain_base', '')
    c_domain = c_n.get('domain_base', '')
    s1_compact = s1_n.get('compact_name', '')
    c_compact = c_n.get('compact_name', '')

    if c_domain and len(c_domain) >= 4 and (c_domain in s1_compact or s1_compact in c_domain):
        domain_match = 1.0
    elif s1_domain and len(s1_domain) >= 4 and (s1_domain in c_compact or c_compact in s1_domain):
        domain_match = 1.0

    # === ADDRESS FEATURES (4) ===
    s1_a = s1_norm['addr']
    c_a = cand_norm['addr']
    s1_addr = s1_a['clean_addr']
    c_addr = c_a['clean_addr']

    cand_addr_empty = 1.0 if not c_addr else 0.0
    s1_addr_empty = 1.0 if not s1_addr else 0.0

    if s1_addr and c_addr:
        addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
        addr_sort_ratio = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    else:
        addr_ratio = 0.0
        addr_sort_ratio = 0.0

    # === NUMBER FEATURES (3) ===
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

    # === STATE FEATURES (2) ===
    s1_state = s1_a['canonical_state']
    c_state = c_a['canonical_state']
    if s1_state and c_state:
        compatible = (s1_state == c_state) or ({s1_state, c_state} == {'telangana', 'andhra pradesh'})
        state_match = 1.0 if compatible else -1.0
        state_conflict = 0.0 if compatible else 1.0
    else:
        state_match = 0.0
        state_conflict = 0.0

    # === STREET FEATURES (2) - NEW ===
    s1_streets = set(s1_a['street_tokens'])
    c_streets = set(c_a['street_tokens'])
    street_sim = 0.0
    street_conflict = 0.0
    if s1_streets and c_streets:
        if s1_streets & c_streets:
            street_sim = jaccard_similarity(s1_streets, c_streets)
        else:
            street_conflict = 1.0
            # Best fuzzy match between any pair of street tokens
            street_sim = max((fuzz.ratio(a, b) for a in s1_streets for b in c_streets), default=0.0) / 100.0
    elif not s1_streets and not c_streets:
        street_sim = 0.5  # Both missing

    # === SOURCE INDICATOR (1) ===
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0

    # === COMPOSITE (1) ===
    max_sim = max(name_set_ratio, addr_sort_ratio)

    return [
        # Name features (10)
        name_ratio,           # 0
        name_sort_ratio,      # 1
        name_set_ratio,       # 2
        name_partial_ratio,   # 3
        name_jaccard,         # 4
        name_char_jaccard,    # 5
        name_exact,           # 6
        first_token_match,    # 7  NEW
        name_token_coverage,  # 8  NEW
        name_len_ratio,       # 9  NEW
        # Domain (1)
        domain_match,         # 10
        # Address (4)
        addr_ratio,           # 11
        addr_sort_ratio,      # 12
        cand_addr_empty,      # 13
        s1_addr_empty,        # 14
        # Numbers (3)
        nums_overlap,         # 15
        prim_num_match,       # 16
        num_conflict,         # 17
        # State (2)
        state_match,          # 18
        state_conflict,       # 19
        # Street (2) NEW
        street_sim,           # 20
        street_conflict,      # 21
        # Source (1)
        is_s2,                # 22
        # Composite (1)
        max_sim,              # 23
    ]


FEATURE_NAMES = [
    'name_ratio',
    'name_sort_ratio',
    'name_set_ratio',
    'name_partial_ratio',
    'name_jaccard',
    'name_char_jaccard',
    'name_exact',
    'first_token_match',
    'name_token_coverage',
    'name_len_ratio',
    'domain_match',
    'addr_ratio',
    'addr_sort_ratio',
    'cand_addr_empty',
    's1_addr_empty',
    'nums_overlap',
    'prim_num_match',
    'num_conflict',
    'state_match',
    'state_conflict',
    'street_sim',
    'street_conflict',
    'is_s2',
    'max_sim',
]
