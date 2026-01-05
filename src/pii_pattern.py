
"""
some patterns are borrowed from paper: SANTACODER: DON’T REACH FOR THE STARS (https://arxiv.org/abs/2301.03988)

"""

# ------------------- Regex Patterns -------------------
ipv4_pattern = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}"
ipv6_pattern = r"(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
ip_pattern = (
    r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])("
    + r"|".join([ipv4_pattern, ipv6_pattern])
    + r")(?:$|[\s@,?!;:'\"(.\p{Han}])"
)
email_pattern = r'''
    (?<= ^ | [\b\s@,?!;:)('".\p{Han}<] )
    (
      [^\b\s@?!;,:)('"<]+
      @
      [^\b\s@!?;,/]*
      [^\b\s@?!;,/:)('">.]
      \.
      \p{L} \w{1,}
    )
    (?= $ | [\b\s@,?!;:)('".\p{Han}>] )
'''

# written by hua 
gender_pattern = r"\b(male|female|man|woman|boy|girl|non[- ]?binary|gender[- ]?queer|trans(gender)?|cis(gender)?)\b"

credit_card_pattern = r'''
\b(
    # Visa
    4[0-9]{12}(?:[0-9]{3})? |
    # MasterCard
    5[1-5][0-9]{14} |
    2(?:2[2-9][0-9]{12}|[3-7][0-9]{13}) |  # MasterCard 2017+ range (2221-2720)
    # American Express
    3[47][0-9]{13} |
    # Diners Club
    3(?:0[0-5]|[68][0-9])[0-9]{11} |
    # Discover
    6(?:011|5[0-9]{2})[0-9]{12} |
    # JCB
    (?:2131|1800|35\d{3})\d{11}
)\b
'''

APIs = [
    {"Domain": "Social Media", "Provider": "Meta", "Secret type": "facebook_access_token", "Regex": r"EAACEdEose0cBA[0-9A-Za-z]+", "Risks": "D,M"},
    {"Domain": "Communication", "Provider": "Slack", "Secret type": "slack_api_token", "Regex": r"xox[p|b|o|a]-[0-9]{12}-[0-9]{12}-[0-9]{12}-[a-z0-9]{32}", "Risks": "D,M"},
    {"Domain": "Communication", "Provider": "Slack", "Secret type": "slack_incoming_webhook_url", "Regex": r"https:\/\/hooks.slack.com\/services\/[A-Za-z0-9+\/]{44,46}", "Risks": "D,M"},
    {"Domain": "Communication", "Provider": "Sendinblue", "Secret type": "sendinblue_api_key", "Regex": r"xkeysib-[a-f0-9]{64}-[a-zA-Z0-9]{16}", "Risks": "D,M"},
    {"Domain": "IaaS", "Provider": "Alibaba Cloud", "Secret type": "alibaba_cloud_access_key_id", "Regex": r"LTAI[a-zA-Z0-9]{20}", "Risks": "D,F"},
    {"Domain": "IaaS", "Provider": "Amazon Web Services (AWS)", "Secret type": "aws_access_key_id", "Regex": r"AKIA[0-9A-Z]{16}", "Risks": "D,F"},
    {"Domain": "IaaS", "Provider": "Tencent Cloud", "Secret type": "tencent_cloud_secret_id", "Regex": r"AKID[0-9a-zA-Z]{32}", "Risks": "D,F"},
    {"Domain": "SaaS", "Provider": "Google", "Secret type": "google_api_key", "Regex": r"AIza[0-9A-Za-z\-_]{35}", "Risks": "D,F"},
    {"Domain": "SaaS", "Provider": "Google", "Secret type": "google_oauth_client_id", "Regex": r"[0-9]{11,13}-[a-z0-9]{32}\.apps\.googleusercontent\.com", "Risks": "D,F"},
    {"Domain": "SaaS", "Provider": "Google", "Secret type": "google_oauth_client_secret", "Regex": r"GOCSPX-[a-zA-Z0-9]{28}", "Risks": "D,F"},
    {"Domain": "Payment", "Provider": "Midtrans", "Secret type": "midtrans_sandbox_server_key", "Regex": r"SB-Mid-server-[0-9a-zA-Z_-]{24}", "Risks": "D,F"},
    {"Domain": "Payment", "Provider": "Flutterwave", "Secret type": "flutterwave_live_secret_key", "Regex": r"FLWPUBK_TEST-[0-9a-f]{32}-X", "Risks": "D,F"},
    {"Domain": "Payment", "Provider": "Flutterwave", "Secret type": "flutterwave_test_api_secret_key", "Regex": r"FLWSECK_TEST-[0-9a-f]{32}-X", "Risks": "D,F"},
    {"Domain": "Payment", "Provider": "Stripe", "Secret type": "stripe_live_secret_key", "Regex": r"sk_live_[0-9a-zA-Z]{24}", "Risks": "D,F"},
    {"Domain": "Payment", "Provider": "Stripe", "Secret type": "stripe_test_secret_key", "Regex": r"sk_test_[0-9a-zA-Z]{24}", "Risks": "D,F"},
    {"Domain": "EC", "Provider": "eBay", "Secret type": "ebay_production_client_id", "Regex": r"[a-zA-Z0-9_\-]{8}-[a-zA-Z0-9_\-]{8}PRD-[a-z0-9]{9}-[a-z0-9]{8}", "Risks": "D"},
    {"Domain": "DevOps", "Provider": "GitHub", "Secret type": "github_personal_access_token", "Regex": r"ghp_[0-9a-zA-Z]{36}", "Risks": "D"},
    {"Domain": "DevOps", "Provider": "GitHub", "Secret type": "github_oauth_access_token", "Regex": r"gho_[0-9a-zA-Z]{36}", "Risks": "D"},
]

# Initialize an empty list to hold individual regex patterns
all_individual_patterns = []

for api in APIs:
    # Append each regex, wrapped in a non-capturing group (?:...)
    all_individual_patterns.append(f"(?:{api['Regex']})")

# Join all collected patterns with the OR operator `|`
key_pattern = "|".join(all_individual_patterns)

# fake credit card number for testing
credit_card_number_test = ["4111111111111111",
"4012888888881881",
"4222222222222",
"5555555555554444",
"5105105105105100",
"2223000048450010",
"378282246310005",
"371449635398431",
"6011111111111117",
"6011000990139424",
"3566111111111113"]

# Test IP addresses for private and special ranges
import ipaddress

def is_informative_ip(ip_str: str) -> bool:
    """
    Returns True if an IP (v4 or v6) is 'informative'—i.e., a valid, publicly routable address.
    Returns False for:
      - private
      - loopback
      - link-local
      - unspecified
      - multicast
      - reserved
      - IPv4 broadcast (255.255.255.255)
      - invalid IP strings
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False  # invalid, thus not informative

    # Exclude broadcast explicitly (only IPv4)
    if ip.version == 4 and ip_str == "255.255.255.255":
        return False

    # ip.is_global covers public (not private, not reserved, not loopback, etc.)
    return getattr(ip, "is_global", not (
        ip.is_private or
        ip.is_loopback or
        ip.is_link_local or
        ip.is_unspecified or
        ip.is_multicast or
        ip.is_reserved
    ))

import math
from collections import Counter

def shannon_entropy(s: str) -> float:
    """
    Compute the Shannon entropy of a string.
    Returns a float between 0 and log2(len(set(s))) (max entropy).
    """
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
    return entropy

import re
def is_valid_key(token: str) -> bool:
    """
    Heuristic check: does this look like a valid API key?
    Returns False if suspicious characters are found.
    """
    # Reject if contains any suspicious characters
    # TODO: could clean prefix and suffix, will do later
    if re.search(r'[\\/,;\'\"<>{}\[\]\(\)\s]', token):
        return False
    # Optionally: enforce length and structure
    if len(token) < 16:  # too short
        return False
    # set of unique characters
    unique_chars = set(token)
    if len(unique_chars) < 8:
        return False
    # entropy
    if shannon_entropy(token) < 3.0:
        return False
    return True

import string
# Define allowed characters (standard secure charset)
ALLOWED_CHARS = set(string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}<>?,./")

def looks_like_password(password: str, min_length: int = 8) -> bool:
    """
    Check if a password meets security requirements:
    - Minimum length
    - All characters within allowed charset
    - Sufficient entropy
    """
    if len(password) < min_length:
        return False
    if any(c not in ALLOWED_CHARS for c in password):
        return False
    # if shannon_entropy(password) < min_entropy:
    #     return False
    # if "password" in password.lower():
    #     return False
    # TODO: looks like a function, e.g. "rs.getString"

    return True

def clean_prefix_suffix(raw_string: str, location_start=-1, location_end=-1, clean_strs = r'[\s:：;,.。\'\"<>\[\]\(\)\{\}]+'):
    """
    Clean common leading and trailing characters from a string
    and update the location indices accordingly.

    Args:
        raw_string (str): The original raw string (may include prefix/suffix symbols).
        location_start (int): The start index of the raw string in the source text.
        location_end (int): The end index of the raw string in the source text.
        clean_strs (str): Regex pattern for characters to clean from start/end.

    Returns:
        Tuple[str, int, int]: A tuple containing the cleaned string,
                              the updated start index,
                              and the updated end index.
    """
    # Match prefix and calculate its length
    prefix_match = re.match(r'^' + clean_strs, raw_string)
    prefix_len = len(prefix_match.group()) if prefix_match else 0

    # Match suffix and calculate its length
    suffix_match = re.search(clean_strs + r'$', raw_string)
    suffix_len = len(suffix_match.group()) if suffix_match else 0

    # Clean the string by trimming the matched prefix and suffix
    cleaned = raw_string[prefix_len: len(raw_string) - suffix_len if suffix_len else None]

    # Update the original character indices to reflect the cleaned result
    new_start = location_start + prefix_len if location_start != -1 else -1
    new_end = location_end - suffix_len if location_end != -1 else -1

    return cleaned, new_start, new_end

import re

def is_full_english_or_spanish_name(value: str) -> bool:
    """
    Determines if the input name is a "full English or Spanish name" containing at least two parts.
    Allowed separators: space, period (.), hyphen (-)
    Allowed characters: English letters, Spanish characters (áéíóúüñÁÉÍÓÚÜÑ)

    Args:
        value (str): The name string to be checked

    Returns:
        bool: Returns True if it is a valid English/Spanish full name, False otherwise
    """
    value = value.strip()

    # at least two words, connected by space, period (.), or hyphen (-)
    multi_word_pattern = re.compile(
        r"^([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\.?|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)([ .-]([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\.?|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+))+$"
    )
    
    # disallow characters other than letters and these separators
    invalid_char_pattern = re.compile(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ .-]")

    if not multi_word_pattern.match(value):
        return False
    if invalid_char_pattern.search(value):
        return False

    return True

def is_username(value: str) -> bool:
    """
    Check if the input string is a valid username.
    """
    # check [^a-zA-Z0-9_-]
    if re.search(r"[^a-zA-Z0-9_-]", value):
        return False, "contains invalid characters"
    # check length
    if len(value) < 3:
        return False, "too short"
    false_username = [
        "admin",
        "root",
        "user",
        "guest",
        "test",
        "testuser",
        "username",
        "Administrator",
        "usernam",
        "use",
        "nickName",
        "getUserNam",
        "name",
        "userNam",
        
        ]
    if value.lower() in false_username:
        return False, "in false username list"
    return True, "valid username"
    









