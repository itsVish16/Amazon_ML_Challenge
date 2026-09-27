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

    # These are high-precision anchors, independent of the model score.
    if source_name and source_name == candidate_name and len(source_name) >= 4 and not number_conflict:
        return True

    candidate_domain = candidate_norm['name'].get('domain_base', '')
    source_compact = source_norm['name'].get('compact_name', '')
    if candidate_domain and len(candidate_domain) >= 5 and candidate_domain in source_compact and not number_conflict:
        return True

    if probability < threshold:
        return False

    name_similarity = max(
        fuzz.token_sort_ratio(source_name, candidate_name),
        fuzz.token_set_ratio(source_name, candidate_name),
    ) / 100.0

    # A hard address-number conflict is only acceptable with an exceptionally
    # similar name. Missing addresses need the same conservative treatment.
    if number_conflict and name_similarity < 0.80:
        return False
    if (not source_addr or not candidate_addr) and name_similarity < 0.75:
        return False
    if name_similarity < 0.40 and not (source_nums and candidate_nums and (source_nums & candidate_nums)):
        return False
    return True
