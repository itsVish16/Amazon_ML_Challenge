"""Shared, precision-first pair decision rules.

Keeping these rules in one module prevents the validation threshold from being
calibrated against different behaviour than the production pipeline.
"""

from rapidfuzz import fuzz


def should_skip_pair(source_norm: dict, candidate_norm: dict) -> bool:
    """Reject a clearly implausible candidate before expensive model scoring."""
    source_nums = set(source_norm['addr']['numbers'])
    candidate_nums = set(candidate_norm['addr']['numbers'])
    if source_nums and candidate_nums and not (source_nums & candidate_nums):
        source_tokens = set(source_norm['name']['tokens'])
        candidate_tokens = set(candidate_norm['name']['tokens'])
        return not (source_tokens & candidate_tokens)
    return False


def should_accept_pair(source_norm: dict, candidate_norm: dict, probability: float,
                       threshold: float) -> bool:
    """Apply the exact production policy for one scored candidate pair."""
    source_name = source_norm['name']['clean_name']
    candidate_name = candidate_norm['name']['clean_name']
    source_addr = source_norm['addr']['clean_addr']
    candidate_addr = candidate_norm['addr']['clean_addr']
    source_nums = set(source_norm['addr']['numbers'])
    candidate_nums = set(candidate_norm['addr']['numbers'])
    number_conflict = bool(source_nums and candidate_nums and not (source_nums & candidate_nums))

    source_streets = set(source_norm['addr']['street_tokens'])
    candidate_streets = set(candidate_norm['addr']['street_tokens'])
    street_conflict = bool(source_streets and candidate_streets and not (source_streets & candidate_streets))

    source_state = source_norm['addr'].get('canonical_state', '')
    candidate_state = candidate_norm['addr'].get('canonical_state', '')
    state_conflict = bool(source_state and candidate_state and source_state != candidate_state)

    # Rejection: If both entities have conflicting states AND conflicting streets, they are physically disjoint.
    if state_conflict and street_conflict:
        return False

    candidate_domain = candidate_norm['name'].get('domain_base', '')
    source_compact = source_norm['name'].get('compact_name', '')
    domain_match = bool(candidate_domain and len(candidate_domain) >= 5 and candidate_domain in source_compact)

    # Anchor 1: Exact clean name + compatible address (no number conflict, no street conflict, no state conflict)
    if source_name and source_name == candidate_name and len(source_name) >= 4:
        if not number_conflict and not street_conflict and not state_conflict:
            return True

    # Anchor 2: Domain match with compatible address
    if domain_match and not number_conflict and not (state_conflict and street_conflict):
        return True

    if probability < threshold:
        return False

    name_similarity = max(
        fuzz.token_sort_ratio(source_name, candidate_name),
        fuzz.token_set_ratio(source_name, candidate_name),
    ) / 100.0

    # Rule 1: A hard address number conflict cannot be accepted without domain match
    if number_conflict and not (domain_match and name_similarity >= 0.95):
        return False

    # Rule 2: Street conflict on different streets requires extremely high name similarity and no number conflict
    if street_conflict and (number_conflict or name_similarity < 0.95):
        return False

    # Rule 3: Missing address requires high name similarity
    if (not source_addr or not candidate_addr) and name_similarity < 0.75:
        return False

    # Rule 4: Low name similarity requires matching address numbers
    if name_similarity < 0.40 and not (source_nums and candidate_nums and (source_nums & candidate_nums)):
        return False

    return True
