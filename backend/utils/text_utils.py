import re

route_map = {
    "cap": "oral", "tab=taz=tas": "oral", "tablet": "oral", "capsule": "oral",
    "syrup": "oral", "injection": "injectable", "ointment": "topical",
    "paint": "topical", "mouth wash": "oral rinse", "mouthwash": "oral rinse",
    "gel": "topical", "eyedrop": "ocular", "eye drop": "ocular", "drops": "ocular",
    "brush": "dental", "toothpaste": "dental", "paste": "dental", "tas": "oral", "taz": "oral"
}

term_normalizer = {
    "tas": "tab", "taz": "tab", "tab.": "tab", "drip": "drops",
    "dryps": "drops", "drips": "drops", "cap.": "cap", "inj.": "injection",
    "oint.": "ointment", "syp": "syrup", "mouthwash": "mouth wash", "eyedrop": "eye drop"
}

OCR_NORMALIZATION = {
    "8F": "BF", "B/F": "BF", "A/F": "AF", "AF.": "AF", "E0T": "EOT", "EOD": "EOT"
}

ABBREVIATION_MAP = {
    "BF": "before food", "AF": "after food", "AC": "before meals", "PC": "after meals",
    "HS": "at bedtime", "SOS": "as needed", "STAT": "immediately",
    "OD": "once daily", "BD": "twice daily", "TDS": "three times daily",
    "QID": "four times daily", "EOD": "every other day", "EOT": "every other day",
    "PRN": "as needed", "PO": "by mouth", "SL": "sublingual",
    "IM": "intramuscular", "IV": "intravenous", "SC": "subcutaneous",
    "OU": "both eyes", "OD-eye": "right eye", "OS": "left eye",
    "QHS": "every night at bedtime", "QAM": "every morning", "QN": "every night"
}

FREQUENCY_MAP = {
    "1-0-1": "once in the morning and once at night",
    "1-1-1": "morning, afternoon, and night",
    "0-0-1": "once at night",
    "1-0-0": "once in the morning",
    "0-1-0": "once in the afternoon",
    "1-1-0": "morning and afternoon",
    "0-1-1": "afternoon and night"
}

def normalize_prescription_abbreviations(text):
    if not text:
        return text
    for wrong, correct in OCR_NORMALIZATION.items():
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", correct, text, flags=re.IGNORECASE)
    return text

def expand_medical_abbreviations(text):
    if not text:
        return text
    for abbr, full in ABBREVIATION_MAP.items():
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text, flags=re.IGNORECASE)
    return text

def expand_frequency_codes(text):
    if not text:
        return text
    for code, phrase in FREQUENCY_MAP.items():
        text = re.sub(r"\b" + re.escape(code) + r"\b", phrase, text, flags=re.IGNORECASE)
    text = re.sub(r"x\s*(\d+)\s*days", r"for \1 days", text, flags=re.IGNORECASE)
    return text

def convert_digits(text, language='hindi'):
    digit_maps = {
        'hindi':    {'0':'०','1':'१','2':'२','3':'३','4':'४','5':'५','6':'६','7':'७','8':'८','9':'९'},
        'marathi':  {'0':'०','1':'१','2':'२','3':'३','4':'४','5':'५','6':'६','7':'७','8':'८','9':'९'},
        'kannada':  {'0':'೦','1':'೧','2':'೨','3':'೩','4':'೪','5':'೫','6':'೬','7':'೭','8':'೮','9':'೯'},
        'tamil':    {'0':'௦','1':'௧','2':'௨','3':'௩','4':'௪','5':'௫','6':'௬','7':'௭','8':'௮','9':'௯'},
        'telugu':   {'0':'౦','1':'౧','2':'౨','3':'౩','4':'౪','5':'౫','6':'౬','7':'౭','8':'౮','9':'౯'},
        'malayalam':{'0':'൦','1':'൧','2':'൨','3':'൩','4':'൪','5':'൫','6':'൬','7':'൭','8':'൮','9':'൯'},
        'gujarati': {'0':'૦','1':'૧','2':'૨','3':'૩','4':'૪','5':'૫','6':'૬','7':'૭','8':'૮','9':'૯'},
        'bengali':  {'0':'০','1':'১','2':'২','3':'৩','4':'৪','5':'৫','6':'৬','7':'৭','8':'৮','9':'৯'},
        'urdu':     {'0':'۰','1':'۱','2':'۲','3':'۳','4':'۴','5':'۵','6':'۶','7':'۷','8':'۸','9':'۹'}
    }
    digit_map = digit_maps.get(language.lower(), {})
    return ''.join(digit_map.get(ch, ch) for ch in text)