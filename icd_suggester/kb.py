"""Tiny knowledge base for retrieval: one passage per code.

Coding notes are short, original paraphrases of general coding practice. For production,
index the full ICD-10-CM tabular list and your payer / coding-guideline documents.
"""
from .data import CODES

NOTES = {
    "E11.9": "Use when type 2 diabetes is documented and no complication is mentioned. If neuropathy, nephropathy or retinopathy is documented, a more specific code applies.",
    "I10": "Use for a documented diagnosis of high blood pressure without heart or kidney disease. A single elevated reading alone is not a diagnosis.",
    "J45.909": "Asthma without a documented exacerbation. Wheeze, night-time breathlessness and inhaler use are typical supporting findings.",
    "J18.9": "Pneumonia when the organism is not specified. Needs clinical or imaging support such as an infiltrate on chest x-ray.",
    "N39.0": "Urinary tract infection with no site specified. Typically burning on urination, frequency and a positive urinalysis.",
    "M54.50": "Low back pain not otherwise specified. A more specific code applies when sciatica or a disc disorder is documented.",
    "J02.9": "Acute sore throat without a named cause. If strep infection is confirmed, a more specific code applies.",
    "K21.9": "Reflux symptoms such as heartburn and regurgitation without esophagitis. If esophagitis is documented, a different K21 code applies.",
    "E78.5": "Elevated blood lipids with no type specified. More specific codes exist for pure high cholesterol or mixed hyperlipidemia.",
    "F32.9": "Depressive episode without severity specified. Severity, recurrence or psychotic features change the code.",
    "I48.91": "Atrial fibrillation with type not specified. Paroxysmal, persistent and permanent types have their own codes.",
    "J44.9": "COPD with no exacerbation or infection documented. Chronic cough, smoking history and reduced FEV1 are typical.",
    "G43.909": "Migraine with no further specification. Aura, intractability or status migrainosus change the code.",
    "R07.9": "Chest pain with no cause established. Use only when no more definitive diagnosis, such as angina or reflux, is documented.",
}

KB = [
    {
        "code": code,
        "description": desc,
        "note": NOTES[code],
        "text": f"{desc}. {NOTES[code]} Typical findings: {', '.join(phrases)}.",
    }
    for code, (desc, phrases) in CODES.items()
]
