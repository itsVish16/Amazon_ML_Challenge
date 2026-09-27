"""
Text and Address Normalization Module for Business Entity Resolution.
Optimized for US, India, and France business entities with script transliteration,
accent stripping, domain handling, and address canonicalization.
"""

import re
import unicodedata
import anyascii

# Compiled Regex Patterns
RE_ACCENTS = re.compile(r'[\u0300-\u036f]')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_LEADING_ZEROS = re.compile(r'\b0+([0-9]+)\b')
RE_NOISE_PREFIX = re.compile(r'^[#<>\-\*\@\s]+')
RE_PMB_POBOX = re.compile(r'\b(pmb|p\s*o\s*box|po\s*box)\s*[0-9a-z\-]+', re.IGNORECASE)
RE_UNIT_APARTMENT = re.compile(r'\b(unit|apt|apartment|suite|ste|flr|floor|bldg|building|kh\s*no\.?|house\s*no\.?)\b', re.IGNORECASE)

# Legal Suffixes across US, India, and France
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'llc', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'pvt ltd', 'pvt limited', 'private limited', 'pvt',
    'llp', 'associates', 'holdings', 'enterprises', 'trading', 'center',
    'services', 'partners', 'group', 'sarl', 'sasu', 'sas', 'eurl',
    'sa', 'sci', 'snc', 'fils', 'et fils'
}

# Standard Street Suffix Canonicalization
STREET_MAP = {
    'rd': 'road', 'road': 'road',
    'st': 'street', 'str': 'street', 'street': 'street',
    'ave': 'avenue', 'av': 'avenue', 'avenue': 'avenue',
    'blvd': 'boulevard', 'boul': 'boulevard', 'boulevard': 'boulevard', 'bd': 'boulevard',
    'dr': 'drive', 'drive': 'drive',
    'ln': 'lane', 'lane': 'lane',
    'ct': 'court', 'court': 'court',
    'pl': 'place', 'place': 'place',
    'pkwy': 'parkway', 'parkway': 'parkway',
    'hwy': 'highway', 'highway': 'highway',
    'cir': 'circle', 'circle': 'circle',
    'way': 'way',
    'ter': 'terrace', 'terrace': 'terrace',
    'rue': 'rue', 'r': 'rue',
    'voie': 'voie', 'allee': 'allee', 'passage': 'passage',
}

# US State Bidirectional Mapping
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee',
    'tx': 'texas', 'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington',
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia'
}
US_STATE_NAMES = {v: k for k, v in US_STATES.items()}

# Indian State Abbreviation & Local Script Mapping
INDIA_STATES = {
    'mh': 'maharashtra', 'tn': 'tamil nadu', 'ka': 'karnataka', 'dl': 'delhi',
    'up': 'uttar pradesh', 'gj': 'gujarat', 'wb': 'west bengal', 'rj': 'rajasthan',
    'hr': 'haryana', 'mp': 'madhya pradesh', 'pb': 'punjab', 'ap': 'andhra pradesh',
    'ts': 'telangana', 'kl': 'kerala', 'or': 'odisha', 'br': 'bihar', 'as': 'assam',
    'ch': 'chandigarh', 'uttr prdes': 'uttar pradesh', 'mharastr': 'maharashtra',
    'tmilnatu': 'tamil nadu', 'dilli': 'delhi', 'hriyana': 'haryana',
    'gujrat': 'gujarat', 'pnb': 'punjab', 'bengal': 'west bengal'
}

ORDINALS = {
    '1st': '1', 'first': '1', '2nd': '2', 'second': '2',
    '3rd': '3', 'third': '3', '4th': '4', 'fourth': '4',
    '5th': '5', 'fifth': '5', '6th': '6', 'sixth': '6',
    '7th': '7', 'seventh': '7', '8th': '8', 'eighth': '8',
    '9th': '9', 'ninth': '9', '10th': '10', 'tenth': '10'
}


def strip_accents(text: str) -> str:
    """Strip diacritics and accents."""
    text_nfkd = unicodedata.normalize('NFKD', text)
    return RE_ACCENTS.sub('', text_nfkd)


def to_ascii_transliterate(text: str) -> str:
    """Transliterate Indic or non-Latin scripts to ASCII representation."""
    if not text:
        return ""
    if any(ord(c) > 127 for c in text):
        return anyascii.anyascii(text)
    return text


def clean_text_basic(text: str) -> str:
    """General clean text lowercase and normalize spaces."""
    if not text:
        return ""
    t = to_ascii_transliterate(text)
    t = strip_accents(t).lower()
    t = RE_LEADING_ZEROS.sub(r'\1', t)
    t = RE_NON_ALPHANUM.sub(' ', t)
    t = RE_MULTI_SPACE.sub(' ', t).strip()
    return t


def normalize_business_name(raw_name: str) -> dict:
    """
    Comprehensive business name normalization.
    """
    if not raw_name:
        return {
            'clean_name': '', 'tokens': [], 'sorted_tokens': [],
            'acronym': '', 'domain_base': '', 'stripped_name': '',
            'compact_name': ''
        }

    t = to_ascii_transliterate(raw_name)
    t = strip_accents(t).lower().strip()
    t = RE_NOISE_PREFIX.sub('', t).strip()

    # Pre-clean abbreviations before splitting
    t = re.sub(r'\bl\.l\.c\b', 'llc', t, flags=re.I)
    t = re.sub(r'\bpvt\.?\s*ltd\.?\b', 'pvt ltd', t, flags=re.I)
    t = re.sub(r'\bl\.t\.d\b', 'ltd', t, flags=re.I)
    t = re.sub(r'\bl\.l\.p\b', 'llp', t, flags=re.I)

    # Handle "formerly ..." or "dba ..."
    if ' formerly ' in t:
        t = t.split(' formerly ')[-1].strip()
    elif ' dba ' in t:
        t = t.split(' dba ')[-1].strip()

    # Domain detection (e.g., siiainvestments.com, @shoshanarude)
    domain_base = ''
    domain_match = re.search(r'([a-z0-9]+)\.(com|in|org|net|co|io|fr|us)\b', t)
    if domain_match:
        domain_base = domain_match.group(1)
    elif t.startswith('@'):
        domain_base = t[1:].strip()

    clean = RE_NON_ALPHANUM.sub(' ', t)
    tokens = [w for w in clean.split() if w]

    # Filter legal suffixes
    filtered_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and f"{tokens[i]} {tokens[i+1]}" in LEGAL_SUFFIXES:
            i += 2
        elif tokens[i] in LEGAL_SUFFIXES:
            i += 1
        else:
            filtered_tokens.append(tokens[i])
            i += 1

    if not filtered_tokens:
        filtered_tokens = tokens

    clean_name = ' '.join(tokens)
    stripped_name = ' '.join(filtered_tokens)
    sorted_tokens = sorted(filtered_tokens)
    acronym = ''.join(w[0] for w in filtered_tokens if w)
    compact_name = ''.join(filtered_tokens)

    return {
        'clean_name': clean_name,
        'tokens': filtered_tokens,
        'sorted_tokens': sorted_tokens,
        'acronym': acronym,
        'domain_base': domain_base,
        'stripped_name': stripped_name,
        'compact_name': compact_name
    }


def normalize_address(raw_addr: str, country: str = "") -> dict:
    """
    Comprehensive address normalization with robust digit substring extraction.
    """
    if not raw_addr:
        return {
            'clean_addr': '', 'numbers': [], 'primary_number': '',
            'tokens': [], 'canonical_state': '', 'street_tokens': []
        }

    t = to_ascii_transliterate(raw_addr)
    t = strip_accents(t).lower().strip()
    t = RE_NOISE_PREFIX.sub('', t).strip()
    t = RE_PMB_POBOX.sub(' ', t)

    # Clean non-alphanumeric except spaces
    t_clean = RE_NON_ALPHANUM.sub(' ', t)
    words = t_clean.split()

    normalized_words = []
    canonical_state = ''
    country_upper = country.upper() if country else ""

    for w in words:
        if w in ORDINALS:
            w = ORDINALS[w]
        if w in STREET_MAP:
            w = STREET_MAP[w]
        if country_upper == 'US':
            if w in US_STATES:
                canonical_state = w
                w = US_STATES[w]
            elif w in US_STATE_NAMES:
                canonical_state = US_STATE_NAMES[w]
        elif country_upper == 'INDIA':
            if w in INDIA_STATES:
                canonical_state = INDIA_STATES[w]
                w = canonical_state

        normalized_words.append(w)

    clean_addr = ' '.join(normalized_words)

    # Extract all digit sequences (up to 6 digits) and strip leading zeros
    raw_digits = re.findall(r'\b\d{1,6}\b', re.sub(r'\b0+(\d+)', r'\1', clean_addr))
    numbers = []
    for d in raw_digits:
        if d not in numbers:
            numbers.append(d)

    primary_number = numbers[0] if numbers else ''

    stop_addr = {'road', 'street', 'avenue', 'boulevard', 'lane', 'drive', 'court',
                 'place', 'parkway', 'highway', 'way', 'rue', 'unit', 'apartment',
                 'floor', 'suite', 'bldg', 'near', 'opp', 'sector', 'block', 'plot',
                 'null', 'city', 'township', 'region', 'state', 'hq', 'no'}
    street_tokens = [w for w in normalized_words if w not in stop_addr and len(w) > 2 and not w.isdigit()]

    return {
        'clean_addr': clean_addr,
        'numbers': numbers,
        'primary_number': primary_number,
        'tokens': normalized_words,
        'canonical_state': canonical_state,
        'street_tokens': street_tokens
    }
