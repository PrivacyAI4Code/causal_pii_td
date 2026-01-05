from collections import defaultdict

CodeFile_KEYS = {
    "fileId", "blobId", "directoryId", "path", "contentId",
    "detectedLicenses", "licenseType", "repoName", "githubId",
    "language", "content", "length_bytes", "extension", "piiRecords"
}

PiiRecord_KEYS = {
    "piiId", "piiType", "location_start", "location_end", "value",
    "detectedBy", "timestamp", "notes", "isHumanReviewed", "confidenceScore"
}

def CodeFile(**kwargs):
    invalid_keys = set(kwargs.keys()) - CodeFile_KEYS
    if invalid_keys:
        print(f"Invalid key(s) passed to CodeFile: {invalid_keys}")

    d = defaultdict(lambda: None)
    d.update(kwargs)
    return {k: d[k] for k in CodeFile_KEYS}

def PiiRecord(**kwargs):
    invalid_keys = set(kwargs.keys()) - PiiRecord_KEYS
    if invalid_keys:
        print(f"Invalid key(s) passed to PiiRecord_KEYS: {invalid_keys}")
    
    d = defaultdict(lambda: None)
    d.update(kwargs)
    return {k: d[k] for k in PiiRecord_KEYS}

pii_types_list = [
    "name",
    "email",
    "ip_address",
    "gender",
    "phone_number",
    "address", # no regex found for this
    "social_media", # keywords is easy but what info should be collected?
    "date_of_birth",
    "ID",  # e.g., SSN/ real IDs
    "Medical_record",
    "bank_statement",
    "education",
    "political_affiliation",
    "username",
    "password",
    "key",  # e.g., RSA private key
    "credit_card",
    "biometric_data",  # e.g., Facial Recognition data
    "tokens"  # e.g., Authorization Bearer tokens
]