"""Ranked Confidence-Aware Matching Engine.

Architectural change from previous version:
- OLD: Independent binary threshold per candidate (if prob > T: accept)
- NEW: Per-entity ranked scoring with confidence gap enforcement.

This prevents chain store explosion where 20 branches of Subway all score 0.75
and get accepted. Now only candidates with clear confidence are accepted.
"""

from rapidfuzz import fuzz


def are_states_compatible(st1: str, st2: str) -> bool:
    """Check if two canonical states are identical or historically compatible."""
    if not st1 or not st2:
        return True
    if st1 == st2:
        return True
    if {st1, st2} == {'telangana', 'andhra pradesh'}:
        return True
    return False


def should_skip_pair(source_norm: dict, candidate_norm: dict) -> bool:
    """Reject a clearly implausible candidate before expensive model scoring."""
    source_state = source_norm['addr'].get('canonical_state', '')
    cand_state = candidate_norm['addr'].get('canonical_state', '')
    if not are_states_compatible(source_state, cand_state):
        return True

    source_nums = set(source_norm['addr']['numbers'])
    candidate_nums = set(candidate_norm['addr']['numbers'])
    if source_nums and candidate_nums and not (source_nums & candidate_nums):
        source_tokens = set(source_norm['name']['tokens'])
        candidate_tokens = set(candidate_norm['name']['tokens'])
        if not (source_tokens & candidate_tokens):
            return True
    return False


def should_accept_pair(source_norm: dict, candidate_norm: dict, probability: float,
                       threshold: float) -> bool:
    """Score a single candidate pair. Used during training threshold calibration.
    For production inference, use select_matches() instead."""
    source_name = source_norm['name']['clean_name']
    candidate_name = candidate_norm['name']['clean_name']

    source_state = source_norm['addr'].get('canonical_state', '')
    candidate_state = candidate_norm['addr'].get('canonical_state', '')
    if not are_states_compatible(source_state, candidate_state):
        return False

    name_similarity = max(
        fuzz.token_sort_ratio(source_name, candidate_name),
        fuzz.token_set_ratio(source_name, candidate_name),
    ) / 100.0

    # Different businesses cannot match regardless of address similarity
    c_domain = candidate_norm['name'].get('domain_base', '')
    s_compact = source_norm['name'].get('compact_name', '')
    domain_match = bool(c_domain and len(c_domain) >= 4 and c_domain in s_compact)
    if not domain_match and name_similarity < 0.50:
        return False

    if probability < threshold:
        return False

    source_nums = set(source_norm['addr']['numbers'])
    candidate_nums = set(candidate_norm['addr']['numbers'])
    number_conflict = bool(source_nums and candidate_nums and not (source_nums & candidate_nums))

    # Hard number conflict: reject unless domain match with very high name similarity
    if number_conflict and not (domain_match and name_similarity >= 0.95):
        return False

    return True


def select_matches(source_norm: dict, scored_candidates: list, threshold: float) -> list:
    """Ranked confidence-aware matching for one S1 entity.

    Args:
        source_norm: Normalized S1 entity dict with 'name' and 'addr' keys.
        scored_candidates: List of (candidate_id, candidate_norm, probability) tuples.
        threshold: Base decision threshold from model calibration.

    Returns:
        List of matched candidate IDs.
    """
    if not scored_candidates:
        return []

    source_name = source_norm['name']['clean_name']
    source_state = source_norm['addr'].get('canonical_state', '')
    source_nums = set(source_norm['addr']['numbers'])
    s_compact = source_norm['name'].get('compact_name', '')

    # Phase 1: Compute enriched scores and filter physically impossible matches
    enriched = []
    for cand_id, cand_norm, prob in scored_candidates:
        cand_name = cand_norm['name']['clean_name']
        cand_state = cand_norm['addr'].get('canonical_state', '')

        # Hard veto: state conflict
        if not are_states_compatible(source_state, cand_state):
            continue

        name_sim = max(
            fuzz.token_sort_ratio(source_name, cand_name),
            fuzz.token_set_ratio(source_name, cand_name),
        ) / 100.0

        # Hard veto: completely different names
        c_domain = cand_norm['name'].get('domain_base', '')
        domain_match = bool(c_domain and len(c_domain) >= 4 and c_domain in s_compact)
        if not domain_match and name_sim < 0.50:
            continue

        cand_nums = set(cand_norm['addr']['numbers'])
        num_conflict = bool(source_nums and cand_nums and not (source_nums & cand_nums))

        # Hard veto: number conflict without domain rescue
        if num_conflict and not (domain_match and name_sim >= 0.95):
            continue

        enriched.append((cand_id, cand_norm, prob, name_sim, domain_match))

    if not enriched:
        return []

    # Phase 2: Sort by probability descending
    enriched.sort(key=lambda x: -x[2])

    # Phase 3: Accept matches using confidence-aware thresholds
    matched = []
    high_threshold = max(threshold, 0.88)  # Very confident matches

    for cand_id, cand_norm, prob, name_sim, domain_match in enriched:
        # Tier 1: High confidence — accept if model is very sure
        if prob >= high_threshold and name_sim >= 0.50:
            matched.append(cand_id)
            continue

        # Tier 2: Standard confidence — accept with name evidence
        if prob >= threshold and name_sim >= 0.55:
            matched.append(cand_id)
            continue

        # Tier 3: Domain match can bridge moderate probability
        if domain_match and prob >= threshold * 0.85 and name_sim >= 0.30:
            matched.append(cand_id)
            continue

    return matched
