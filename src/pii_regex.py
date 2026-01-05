import regex
import ipaddress
import luhn
import datetime

from pii_pattern import (
    ip_pattern, email_pattern, credit_card_pattern, gender_pattern,
    key_pattern)

year_patterns = [
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/][1-2][0-9]{3})(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # yyyy-yyyy or yyyy/yyyy
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/.][0-3][0-9][\p{Pd}/.][0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # yyyy-mm-dd or yyyy-dd-mm or yyyy/mm/dd or yyyy/dd/mm or yyyy.mm.dd or yyyy.dd.mm
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/.][0-3][0-9][\p{Pd}/.](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # mm-dd-yyyy or dd-mm-yyyy or mm/dd/yyyy or dd/mm/yyyy or mm.dd.yyyy or dd.mm.yyyy or the same but with yy instead of yyyy
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # mm-yyyy or mm/yyyy or the same but with yy
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}-[0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # yyyy-mm or yyyy/mm
]

# regex

# regex.MULTILINE:
# This allows ^ and $ to match at the start and end of each line, not just the whole string.
# Use this when you want to match patterns in multi-line text.

# regex.VERBOSE:
# This lets you write the regex in multiple lines with spaces and comments.
# It ignores extra spaces and newlines, making the regex easier to read.

# regex.IGNORECASE:
# This makes the regex ignore case (uppercase vs lowercase).
# For example, "hello" can match "Hello" or "HELLO" when this flag is used.

# regex
ip_regex = regex.compile(ip_pattern, flags=regex.MULTILINE)
email_regex = regex.compile(email_pattern, flags=regex.MULTILINE | regex.VERBOSE)
gender_regex = regex.compile(gender_pattern, flags=regex.MULTILINE | regex.IGNORECASE)
credit_card_regex = regex.compile(credit_card_pattern, flags=regex.MULTILINE | regex.VERBOSE)
api_key_regex = regex.compile(key_pattern, flags=regex.MULTILINE)


def ip_has_digit(matched_str):
    """Checks to make sure the PII span is not just :: or whatever that may
    accidentally be picked up by making sure there are digits."""
    return any(map(str.isdigit, matched_str))

def matches_date_pattern(matched_str):
    # Screen out date false positives
    for year_regex in year_patterns:
        if year_regex.match(matched_str):
            return True
    return False

def filter_versions(matched_str, context):
    """Filter addresses in this format x.x.x.x  and the words dns/server
    don't appear in the neighboring context, usually they are just versions"""
    # count occurence of dots 
    dot_count = matched_str.count('.')
    exclude = (dot_count == 3 and len(matched_str) == 7) 
    if exclude:
        if "dns" in context.lower() or "server" in context.lower():
            return False
    return exclude

def not_ip_address(matched_str):
    """ make sure the string has a valid IP address format
    e.g: 33.01.33.33 is not a valid IP address because of the 0 in front of 1
    TODO: fix this directly in the regex"""
    try:
        ipaddress.ip_address(matched_str)
        return False
    except ValueError:
        return True
def is_luhn_valid(card_number):
    if luhn.verify(card_number):
        return True
    return False

def get_regex_pattern(pii_type):
    """Returns the regex pattern for the specified PII type."""
    patterns = {
        "ip_address": ip_regex,
        "email": email_regex,
        "gender":gender_regex,
        "key": api_key_regex,
        "credit_card": credit_card_regex,
    }
    return patterns.get(pii_type, None)

def find_pii_by_regex(content, pii_type):
    """Detects patterns in the given content using the provided regex pattern."""
    regex_pattern = get_regex_pattern(pii_type)
    matches = []
    if not regex_pattern:
        raise ValueError(f"No regex pattern found for PII type: {pii_type}")
    matches_tmp = regex_pattern.finditer(content)

    for match in matches_tmp:
        # if pii_type == "key":
        #     value = match.group(0)  # For API keys, we want the whole match
        #     start, end = match.span(0)
        if pii_type == "ip_address":
            value = match.group(1)  # For ip_address, use the ip_address itself
            start, end = match.span(1)
        else:
            value = match.group(0)
            start, end = match.span(0)
        if value:
            if pii_type == "ip_address":
                # Filter out false positive IPs
                if not ip_has_digit(value) :
                    continue
                if matches_date_pattern(value):
                    continue
                if filter_versions(value, content[start-100:end+100]) or  not_ip_address(value):
                    continue
            elif pii_type == "credit_card":
                # Filter out invalid credit card numbers
                if not is_luhn_valid(value):
                    continue

            matches.append({
                "piiId": -1,  # Placeholder for PII ID, to be set later
                "piiType": pii_type,
                "location_start": start,
                "location_end": end,
                "value": value,
                "detectedBy": "regex",
                "timestamp": datetime.datetime.now().isoformat(),
                "notes": None,
                "isHumanReviewed": False,
                "confidenceScore": -1.0
            })
    
    return matches
