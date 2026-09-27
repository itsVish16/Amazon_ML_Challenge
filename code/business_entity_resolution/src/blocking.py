"""
High-Recall Multi-Key Inverted Index Blocking Engine.
Indexes Target records (S2 and S3) by:
1. Address Number + Street / Locality tokens
2. Address Number + State / Region
3. Significant Street token pairs
4. Domain bases & Compact name strings
5. Significant Name token pairs and compact tokens
6. Cross-field keys: Street token + Name token
"""

import re
from collections import defaultdict
from typing import Dict, List, Set
from normalization import normalize_business_name, normalize_address, to_ascii_transliterate, strip_accents


def clean_abbrevs(t: str) -> str:
    """Pre-clean abbreviations before splitting."""
    t = re.sub(r'\bl\.l\.c\b', 'llc', t, flags=re.I)
    t = re.sub(r'\bpvt\.?\s*ltd\.?\b', 'pvt ltd', t, flags=re.I)
    t = re.sub(r'\bl\.t\.d\b', 'ltd', t, flags=re.I)
    t = re.sub(r'\bl\.l\.p\b', 'llp', t, flags=re.I)
    if ' dba ' in t.lower():
        t = t.lower().split(' dba ')[-1]
    elif ' formerly ' in t.lower():
        t = t.lower().split(' formerly ')[-1]
    return t


def generate_blocking_keys(raw_name: str, raw_addr: str, country: str) -> List[str]:
    """Generate high-recall blocking keys for a record."""
    keys = []
    name_clean = clean_abbrevs(raw_name)
    n_info = normalize_business_name(name_clean)
    a_info = normalize_address(raw_addr, country)

    tokens = [w for w in n_info['tokens'] if len(w) > 1]
    if not tokens:
        tokens = n_info['tokens']

    domain = n_info.get('domain_base', '')
    if domain:
        keys.append(f'dom:{domain}')

    # Compact name key
    t_name_ascii = to_ascii_transliterate(name_clean)
    comp_all = re.sub(r'[^a-z0-9]', '', t_name_ascii.lower())
    for suff in ['pvtltd', 'privatelimited', 'ltd', 'limited', 'llc', 'inc', 'corp', 'llp', 'sarl', 'sasu']:
        if comp_all.endswith(suff):
            comp_all = comp_all[:-len(suff)]
            break
    if len(comp_all) >= 4:
        keys.append(f'comp_full:{comp_all[:12]}')

    # Primary name token
    if len(tokens) >= 1:
        if len(tokens[0]) >= 3:
            keys.append(f'n1:{tokens[0]}')

    # Compact prefix & name token pairs
    if len(tokens) >= 2:
        for k in range(2, min(len(tokens) + 1, 4)):
            compact = ''.join(tokens[:k])
            if len(compact) >= 5:
                keys.append(f'comp:{compact}')
        for i in range(min(len(tokens), 3)):
            for j in range(i + 1, min(len(tokens), 3)):
                p = sorted([tokens[i], tokens[j]])
                keys.append(f'name2:{p[0]}_{p[1]}')

    # Address keys
    nums = a_info['numbers']
    sts = a_info['street_tokens']
    state = a_info['canonical_state']

    for n in nums[:3]:
        for s in sts[:3]:
            keys.append(f'n_st:{n}_{s}')
        if state:
            keys.append(f'n_sta:{n}_{state}')

    # Street token pairs (crucial when numbers are omitted or landmark-based)
    if len(sts) >= 2:
        for i in range(min(len(sts), 3)):
            for j in range(i + 1, min(len(sts), 3)):
                p = sorted([sts[i], sts[j]])
                keys.append(f'st2:{p[0]}_{p[1]}')

    # Street + First Name Token (very powerful cross-field anchor)
    if sts and tokens:
        keys.append(f'st_n:{sts[0]}_{tokens[0]}')

    return keys


class CountryBlocker:
    """Inverted Index Blocker for a single country partition."""
    def __init__(self, country: str, max_candidates: int = 40, max_freq: int = 2500):
        self.country = country
        self.max_candidates = max_candidates
        self.max_freq = max_freq
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.key_frequencies: Dict[str, int] = defaultdict(int)

    def add_target_record(self, entity_id: str, raw_name: str, raw_addr: str):
        keys = generate_blocking_keys(raw_name, raw_addr, self.country)
        for k in keys:
            self.index[k].append(entity_id)
            self.key_frequencies[k] += 1

    def retrieve_candidates(self, raw_name: str, raw_addr: str) -> List[str]:
        keys = generate_blocking_keys(raw_name, raw_addr, self.country)
        if not keys:
            return []

        # Sort keys by rarity (lower frequency first)
        scored = [(self.key_frequencies.get(k, 0), k) for k in keys if k in self.index]
        scored.sort(key=lambda x: x[0])

        # A previous implementation stopped as soon as it saw N IDs.  For
        # frequent keys that means the retained candidates depended on source
        # file order, not match evidence.  Score each ID by independent
        # blocking signals, while keeping a small bounded pre-selection pool.
        candidates = defaultdict(float)
        for freq, k in scored:
            if freq > self.max_freq:
                continue
            key_weight = 1.0 / (1.0 + freq)
            for cid in self.index[k]:
                candidates[cid] += key_weight

        return [cid for cid, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:self.max_candidates]]
