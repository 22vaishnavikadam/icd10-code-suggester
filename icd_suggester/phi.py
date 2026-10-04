"""Rule-based PHI redaction (HIPAA Safe Harbor-style, subset).

Regex redaction is a first line of defence only. Production systems should add
a clinical NER model (e.g. Presidio / AWS Comprehend Medical) and audit logging.
"""
import re

_PATTERNS = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    ("MRN", re.compile(r"\b(?:MRN|Member ID|Policy)\s*[:#]?\s*[A-Z0-9-]{5,}\b", re.I)),
    ("NAME", re.compile(r"(?:(?<=Patient: )|(?<=Patient )|(?<=Mr\. )|(?<=Mrs\. )|(?<=Ms\. )|(?<=Dr\. ))[A-Z][a-z]+(?:\s[A-Z][a-z]+)?")),
]


def redact_phi(text: str):
    """Return (redacted_text, counts_by_type)."""
    counts = {}
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts
