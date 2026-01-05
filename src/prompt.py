import re
import requests
import ipaddress
import math
import string
from collections import Counter
from typing import Tuple, Optional, Set

# Compiled regex patterns for efficiency
USERNAME_INVALID_CHARS = re.compile(r'[^a-zA-Z0-9_-]')
KEY_SUSPICIOUS_CHARS = re.compile(r'[\\/,;\'\"<>{}\[\]()\s]')
CREDIT_CARD_PATTERN = re.compile(r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|2(?:2[2-9][0-9]{12}|[3-7][0-9]{13})|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})$')
PHONE_PATTERN = re.compile(r'\b(?:\+?1[-\s]?)?\(?[0-9]{3}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b')
SSN_PATTERN = re.compile(r'^[0-9]{3}-[0-9]{2}-[0-9]{4}$')
GENDER_PATTERN = re.compile(r'\b(male|female|man|woman|boy|girl|non[- ]?binary|gender[- ]?queer|trans(gender)?|cis(gender)?)\b', re.IGNORECASE)
DATE_PATTERN = re.compile(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?[0-9]{2}\b')

# Define allowed characters (standard secure charset)
ALLOWED_CHARS = set(string.ascii_letters + string.digits + string.punctuation)

# Global false username list - pre-computed as set for O(1) lookup
FALSE_USERNAMES = [
    "admin", "root", "user", "guest", "test", "testuser", "username",
    "Administrator", "nickName", "getUserName", "name", "userName",
]
false_username = {name.lower() for name in FALSE_USERNAMES}

FALSE_PASSWORDS = [ "password", "admin", "123456", "123456789", "12345678", "12345", "1234", "111111", "000000", "password1", "password123", "passw0rd", "qwerty", "qwerty123", \
"abc123", "1q2w3e", "123123", "admin", "user", "letmein", "welcome", "default", "changeme", "test", "test123", "asdf", "asdfgh", "iloveyou", "secret", "my_password", "yourpassword", \
"<password>", "pwd", "toor", "root", "guest", "login", "access", "temp123"
]
false_password = {password.lower() for password in FALSE_PASSWORDS}

def find_quotes(full_s, value, location_start, location_end, search_size = 10):
    left_start = max(0, location_start-search_size)
    left_end = location_start + int(len(value) /2)
    right_end = min(len(full_s), location_end+search_size)
    right_start = location_end - int(len(value) /2)

    # 在左边搜索最后一个 quote
    left_quote_pos = None
    left_quote_char = None
    for i in range(left_end - 1, left_start - 1, -1):
        if full_s[i] in ("'", '"'):
            left_quote_pos = i
            left_quote_char = full_s[i]
            break

    # 在右边搜索第一个 quote
    right_quote_pos = None
    right_quote_char = None
    for i in range(right_start, right_end):
        if full_s[i] in ("'", '"'):
            right_quote_pos = i
            right_quote_char = full_s[i]
            break
    
    # 如果两边都找到 且 引号一致
    if (left_quote_pos is not None and right_quote_pos is not None 
        and left_quote_char == right_quote_char):
        new_value = full_s[left_quote_pos + 1:right_quote_pos]
        return new_value, left_quote_pos + 1, right_quote_pos
    else:
        # 什么都不做
        return value, location_start, location_end






def is_informative_ip(ip_str: str) -> bool:
    """
    Returns True if an IP (v4 or v6) is 'informative'—i.e., a valid, publicly routable address.
    Returns False for private, loopback, link-local, unspecified, multicast, reserved, or invalid IPs.
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

def is_valid_key(token: str):
    """
    Heuristic check: does this look like a valid API key?
    Returns False if suspicious characters are found.
    """
    # Use pre-compiled regex for efficiency
    if KEY_SUSPICIOUS_CHARS.search(token):
        return False, "Key contains suspicious characters"
    if len(token) < 16:  # too short
        return False, "Key is too short"
    # Check character diversity
    unique_chars = set(token)
    if len(unique_chars) < 8:
        return False, "Key has less than 8 unique characters"
    # Check entropy
    if shannon_entropy(token) < 3.0:
        return False, "Key has low entropy"
    return True, "Key passed quick check"

def looks_like_password(password: str, min_length: int = 8) -> bool:
    """
    Check if a password meets security requirements:
    - Minimum length
    - All characters within allowed charset
    """
    if len(password) < min_length:
        return False, "Password is too short"
    if any(c not in ALLOWED_CHARS for c in password):
        return False, "Password contains invalid characters"
    if password.lower() in false_password:
        return False, "Password is in false password list"
    return True, "Password passed quick check"

# Name regex patterns
NAME_MULTI_WORD = re.compile(r'^([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\.?|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)([ .-]([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\.?|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+))+$')
NAME_INVALID_CHARS = re.compile(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ .-]')
def is_full_english_or_spanish_name(value: str) -> bool:
    """
    Determines if the input name is a "full English or Spanish name" containing at least two parts.
    Allowed separators: space, period (.), hyphen (-)
    Allowed characters: English letters, Spanish characters (áéíóúüñÁÉÍÓÚÜÑ)
    """
    value = value.strip()
    
    # Use pre-compiled regex patterns for efficiency
    return bool(NAME_MULTI_WORD.match(value) and not NAME_INVALID_CHARS.search(value))

NAME_MULTI_WORD_ENGLISH = re.compile(
    r'^([a-zA-Z]\.?|[a-zA-Z]+)(([ .-]|, )([a-zA-Z]\.?|[a-zA-Z]+))+$'
)
NAME_INVALID_CHARS_ENGLISH = re.compile(r'[^a-zA-Z .,-]')

def is_full_english(value: str) -> bool:
    """
    Determines if the input is a full English name with at least two parts.
    Allowed characters: a-z, A-Z, space, period (.), hyphen (-), and comma (,) 
    Accepts formats like:
    - First Last
    - Last, First
    - First-Last
    - C. Last
    """
    value = value.strip()
    return (
        bool(NAME_MULTI_WORD_ENGLISH.fullmatch(value))
        and not NAME_INVALID_CHARS_ENGLISH.search(value)
    )
# def is_valid_credit_card(card_number: str) -> bool:
#     """Validate credit card number format and Luhn check"""
#     clean_card = card_number.replace(' ', '').replace('-', '')
#     return bool(CREDIT_CARD_PATTERN.match(clean_card) and is_luhn_valid(clean_card))

def is_valid_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    return bool(PHONE_PATTERN.search(phone))

def is_valid_ssn(ssn: str) -> bool:
    """Validate SSN format"""
    return bool(SSN_PATTERN.match(ssn))

def is_valid_gender(gender: str) -> bool:
    """Validate gender term"""
    return bool(GENDER_PATTERN.search(gender))

def is_valid_date_of_birth(date: str) -> bool:
    """Validate date format for date of birth"""
    return bool(DATE_PATTERN.search(date))

def is_valid_username(value: str) -> bool:
    """Check if the input string is a valid username."""
    if USERNAME_INVALID_CHARS.search(value):
        return False
    if len(value) < 6:
        return False
    if value.lower() in false_username:
        return False
    return True

def load_iana_tlds() -> Set[str]:
    """Load IANA TLD list with error handling"""
    try:
        url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            return {line.lower() for line in lines if not line.startswith("#")}
        else:
            print(f"Warning: Failed to download IANA TLD list (status: {response.status_code})")
            return set()
    except Exception as e:
        print(f"Warning: Could not load IANA TLD list: {e}")
        return set()

# Lazy loading to avoid import-time failures
iana_tlds = load_iana_tlds()

def remove_prefix_suffix(value: str,
                        location_start: int,
                        location_end: int,
                        prefix: str = r'^[\s:：;,.。\'\"<>\[\]\(\)\{\}]+', 
                        suffix: str = r'[\s:：;,.。\'\"<>\[\]\(\)\{\}]+$') -> Tuple[Optional[str], int, int]:
    """Remove prefix and suffix from value and adjust location indices"""
    # Match prefix and calculate its length
    prefix_match = re.match(prefix, value)
    prefix_len = len(prefix_match.group()) if prefix_match else 0

    # Match suffix and calculate its length
    suffix_match = re.search(suffix, value)
    suffix_len = len(suffix_match.group()) if suffix_match else 0

    # Clean the string by trimming the matched prefix and suffix
    cleaned = value[prefix_len: len(value) - suffix_len if suffix_len else None]

    # Update the original character indices to reflect the cleaned result
    new_start = location_start + prefix_len 
    new_end = location_end - suffix_len 

    # Return None if cleaned value is invalid
    if new_start < 0 or new_end <= new_start or not cleaned.strip():
        return None, location_start, location_end

    return cleaned, new_start, new_end

def quick_check(value: str, location_start: int, location_end: int, piiType: str, content: str) -> Tuple[Optional[str], int, int, str]:
    """
    Perform a quick check on the value based on its PII type.
    Returns: (cleaned_value_or_None, start_location, end_location, explanation)
    """
    if piiType == "email":
        # remove common leading and trailing characters
        result = remove_prefix_suffix(value, location_start, location_end)
        if result[0] is None:
            return None, location_start, location_end, "Value is not a valid PII value"
        value, location_start, location_end = result
        
        if '..' in value:
            return None, location_start, location_end, "Email contains '..'"
        if value.startswith('.') or value.endswith('.'):
            return None, location_start, location_end, "Email starts or ends with '.'"
        if value.count('@') != 1:
            return None, location_start, location_end, "Email contains less or more than one '@' symbol"
        
        local, domain = value.split('@')
        if not local or not domain:
            return None, location_start, location_end, "Email contains no local or domain part"
        if domain.startswith('.') or domain.endswith('.'):
            return None, location_start, location_end, "Domain starts or ends with '.'"
        # Check for specific invalid patterns in email
        if ' ' in value or '/' in value or '(' in value or ')' in value:
            return None, location_start, location_end, "Email contains invalid characters"
        else:
            # check is value has valid tld
            tld = domain.strip().split('.')[-1].lower()
            if iana_tlds and tld not in iana_tlds:
                explanation = "Email has invalid TLD."
                return None, location_start, location_end, explanation
        
        return value, location_start, location_end, "Email passed quick check"
    
    elif piiType == "ip_address":
        result = remove_prefix_suffix(value, location_start, location_end)
        if result[0] is None:
            return None, location_start, location_end, "Value is not a valid PII value"
        value, location_start, location_end = result
        
        if not is_informative_ip(value):
            return None, location_start, location_end, "Value is an uninformative IP address"
        return value, location_start, location_end, "IP address passed quick check"
    
    elif piiType == "key":
        result = remove_prefix_suffix(value, location_start, location_end)
        if result[0] is None:
            return None, location_start, location_end, "Value is not a valid PII value"
        value, location_start, location_end = result

        value, location_start, location_end = find_quotes(content, value, location_start, location_end)
        
        is_valid, explanation = is_valid_key(value)
        if not is_valid:
            return None, location_start, location_end, explanation
        return value, location_start, location_end, "Key passed quick check"
    
    elif piiType == "name":
        result = remove_prefix_suffix(value, location_start, location_end)
        if result[0] is None:
            return None, location_start, location_end, "Value is not a valid PII value"
        value, location_start, location_end = result
        
        if not is_full_english(value):
            return None, location_start, location_end, "Value is not a full English name"
        return value, location_start, location_end, "Name passed quick check"
    
    elif piiType == "password":
        result = remove_prefix_suffix(value, location_start, location_end, 
                                    prefix=r'^\s+', suffix=r'\s+$')
        if result[0] is None:
            return None, location_start, location_end, "Value is not a valid PII value"
        value, location_start, location_end = result

        value, location_start, location_end = find_quotes(content, value, location_start, location_end)
        
        is_valid, explanation = looks_like_password(value)
        if not is_valid:
            return None, location_start, location_end, explanation
        return value, location_start, location_end, explanation
    
    elif piiType == "username":
        result = remove_prefix_suffix(value, location_start, location_end, 
                                    prefix=r'^\s+', suffix=r'\s+$')
        if result[0] is None:
            return None, location_start, location_end, "Value is not a valid PII value"
        value, location_start, location_end = result
        
        if not is_valid_username(value):
            return None, location_start, location_end, "Value is not a valid username"
        return value, location_start, location_end, "Username passed quick check"
    
    # elif piiType == "credit_card":
    #     result = remove_prefix_suffix(value, location_start, location_end)
    #     if result[0] is None:
    #         return None, location_start, location_end, "Value is not a valid PII value"
    #     value, location_start, location_end = result
        
    #     if not is_valid_credit_card(value):
    #         return None, location_start, location_end, "Value is not a valid credit card"
    #     return value, location_start, location_end, "Credit card passed quick check"
    
    # elif piiType == "phone_number":
    #     result = remove_prefix_suffix(value, location_start, location_end)
    #     if result[0] is None:
    #         return None, location_start, location_end, "Value is not a valid PII value"
    #     value, location_start, location_end = result
        
    #     if not is_valid_phone_number(value):
    #         return None, location_start, location_end, "Value is not a valid phone number format"
    #     return value, location_start, location_end, "Phone number passed quick check"
    
    # elif piiType in ["ID", "ssn"]:
    #     result = remove_prefix_suffix(value, location_start, location_end)
    #     if result[0] is None:
    #         return None, location_start, location_end, "Value is not a valid PII value"
    #     value, location_start, location_end = result
        
    #     if not is_valid_ssn(value):
    #         return None, location_start, location_end, "Value is not a valid SSN format"
    #     return value, location_start, location_end, "SSN passed quick check"
    
    # elif piiType == "gender":
    #     result = remove_prefix_suffix(value, location_start, location_end)
    #     if result[0] is None:
    #         return None, location_start, location_end, "Value is not a valid PII value"
    #     value, location_start, location_end = result
        
    #     if not is_valid_gender(value):
    #         return None, location_start, location_end, "Value is not a recognized gender term"
    #     return value, location_start, location_end, "Gender passed quick check"
    
    # elif piiType == "date_of_birth":
    #     result = remove_prefix_suffix(value, location_start, location_end)
    #     if result[0] is None:
    #         return None, location_start, location_end, "Value is not a valid PII value"
    #     value, location_start, location_end = result
        
    #     if not is_valid_date_of_birth(value):
    #         return None, location_start, location_end, "Value is not a valid date format"
    #     return value, location_start, location_end, "Date of birth passed quick check"
    
    # Default case for unsupported PII types
    return value, location_start, location_end, f"No validation available for PII type: {piiType}"

PROMPT_TEMPLATES = {
  "email": {
    "system": "You are a privacy inspection assistant that evaluates whether a VALUE is a sensitive email address.",
    "user": "The following is the VALUE and its context. Please determine whether the VALUE is a sensitive email address:\nContext:\n{context}\nVALUE: {value}"
  },

  "password": {
    "system": "You are a privacy inspection assistant that evaluates whether a VALUE is a sensitive password.",
    "user": "The following is the VALUE and its context. Please determine whether the VALUE is a sensitive password:\nContext:\n{context}\nVALUE: {value}"
  },

  "name": {
    "system": "You are a privacy inspection assistant that evaluates whether a VALUE is a sensitive personal name.",
    "user": "The following is the VALUE and its context. Please determine whether the VALUE is a sensitive person's name:\nContext:\n{context}\nVALUE: {value}"
  },

  "username": {
    "system": "You are a privacy inspection assistant that evaluates whether a VALUE is a sensitive username.",
    "user": "The following is the VALUE and its context. Please determine whether the VALUE is a sensitive username:\nContext:\n{context}\nVALUE: {value}"
  },

  "ip_address": {
    "system": "You are a privacy inspection assistant that evaluates whether a VALUE is a sensitive IP address.",
    "user": "The following is the VALUE and its context. Please determine whether the VALUE is a sensitive IP address:\nContext:\n{context}\nVALUE: {value}"
  },

  "key": {
    "system": "You are a privacy inspection assistant that evaluates whether a VALUE is a sensitive key (e.g., API key, token, secret).",
    "user": "The following is the VALUE and its context. Please determine whether the VALUE is a sensitive key:\nContext:\n{context}\nVALUE: {value}"
  }
}

INSTRUCTIONS = {
    "email": (
        "You are a privacy inspection assistant. Your task is to determine whether a given VALUE (from GitHub) is a sensitive email address.\n"
        "Score from 1 to 100 (100 means highly sensitive) using these criteria:\n"
        "1. Format: a invalid email (name@domain.tld) with invalid tld or invalid domain should receive a low score\n"
        "2. Context: If the VALUE is used in a test case, such as in a Test function or @Test annotation, it should be given a low score.\n"
        "3. Realness: If the email is a a placeholder like test@example.com, it should be given a low score.\n"
        "Example low-sensitivity patterns such as \"foo@bar.com\", \"example.com\", or redacted addresses should be given a low score.\n"
        "Output only a JSON object:\n{ \"score\": int, \"reason\": str }"
    ),
    "password": (
        "You are a privacy inspection assistant. Your task is to determine whether a given VALUE (from GitHub) is a sensitive password.\n"
        "Score from 1 to 100 (100 means highly sensitive) using these criteria:\n"
        "1. Format:Dummy passwords (e.g., '123456', 'password') or invalid formats should receive a low score. If the VALUE is only part of a password, it should also receive a low score.\n"
        "2. Context:If the VALUE is a sensitive password (as a string literal or in a comment) and is assigned to fields such as `password`, `pwd`, or `secret` in code or configuration files, it should receive a high score.\n"
        "If the VALUE appears in a test case (e.g., inside a test function or in a `@Test` annotation), it should receive a low score.\n"
        "If the VALUE is a token, it should receive a low score.\n"
        "If the VALUE is a method name, function name, variable, or getter method (even if related to password), it should receive a low score.\n"
        "3. Realness: A true sensitive password is not a placeholder, test, or dummy value. It has high entropy and is often directly hard-coded as a string literal in the source code.\n"
        "Output only a JSON object:\n{ \"score\": int, \"reason\": str}"
    ),
    "name": (
        "You are a privacy inspection assistant. Your task is to determine whether a given VALUE (from GitHub) is sensitive name of a person.\n"
        "Score from 1 to 100 (100 means highly sensitive) based on:\n"
        "1. Format: a full name (e.g., two capitalized words) should receive a high score, otherwise a low score\n"
        "2. Context: if the name is used in author, user, or PII-related fields, it should be given a high score\n If the VALUE is used in a test case, such as in a Test function or @Test annotation, it should be given a low score.\n"
        "3. Realness: if the name is a placeholder like 'John Doe' or 'Test User' or in a test case, it should be given a low score\n"
        "Common false positives such as: Strings that resemble organization names, class/method names, common placeholders, etc should be given a low score\n"
        "Output only a JSON object:\n{ \"score\": int, \"reason\": str }"
    ),
    "username": (
        "You are a privacy inspection assistant. Your task is to determine whether a given VALUE (from GitHub) is a sensitive username.\n"
        "Score from 1 to 100 (100 means highly sensitive) based on:\n"
        "1. Format: it should only contain alphanumeric characters, space, underscore, or hyphen, otherwise it should be given a low score\n"
        "2. Context: if the username is used in author, user, or PII-related fields, it should be given a high score\n If the VALUE is used in a test case, such as in a Test function or @Test annotation, it should be given a low score.\n"
        "3. Realness: if the username is a placeholder like 'admin', 'test_user' or 'username', it should be given a low score\n"
        "Common false positives, Strings that resemble organization names, class/method names, common placeholders, etc should be given a low score.\n"
        "If the VALUE is only part of the username, it should be given a low score.\n"
        "If the VALUE is nickname or only first name or last name, it should be given a low score.\n"
        "If the VALUE is simply a chinese name or a single word or a number, it should be given a low score.\n"
        "Output only a JSON object:\n{ \"score\": int, \"reason\": str }"
    ),
    "ip_address": (
        "You are a privacy inspection assistant. Your task is to determine whether a given VALUE (from GitHub) is a sensitive IP address.\n"
        "Score from 1 to 100 (100 means highly sensitive) based on:\n"
        "1. Format: if it is not a valid IPv4 or IPv6 address, it should be given a low score\n"
        "2. Context: if the VALUE is used in logs or connection metadata, it should be given a high score\n If the VALUE is used in a test case, such as in a Test function or @Test annotation, it should be given a low score.\n"
        "3. Realness: if the VALUE is a local/test address like 127.0.0.1 or 192.168.x.x, it should be given a low score\n"
        "Output only a JSON object:\n{ \"score\": int, \"reason\": str }"
    ),
    "key": (
        "You are a privacy inspection assistant. Your task is to determine whether a given VALUE (from GitHub) is a sensitive key (e.g., API key, secret token).\n"
        "Score from 1 to 100 (100 means highly sensitive) using:\n"
        "1. Format: if it is a high-entropy or having well-known prefix for keys (e.g., 'sk-', 'AKIA', 'ghp_', etc), it should be given a high score\n"
        "2. Context: if it is used in field like 'key', 'token', 'secret', it should be given a high score\n If the VALUE is used in a test case, such as in a Test function or @Test annotation, it should be given a low score.\n"
        "3. Realness: if it is a dummy string like 'XXXXX', 'your_api_key_here', it should be given a low score\n"
        "If the VALUE is only part of the key, it should be given a low score.\n"
        "Output only a JSON object:\n{ \"score\": int, \"reason\": str }"
    )
}



    


