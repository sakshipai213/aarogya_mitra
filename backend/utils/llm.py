import os
import re
import json
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def extract_json_array(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    multi_flat = re.match(r'^\s*(\[.*\])\s*,\s*(\[.*)', text, re.DOTALL)
    if multi_flat:
        wrapped = "[" + text[:text.rfind("]") + 1].strip().rstrip(",") + "]"
        try:
            result = json.loads(wrapped)
            if isinstance(result, list) and all(isinstance(x, list) for x in result):
                return [
                    {"medicine": x[0], "route": x[1] if len(x) > 1 else "oral",
                     "dosage and duration": x[2] if len(x) > 2 else ""}
                    for x in result if x and isinstance(x[0], str)
                ]
        except Exception:
            pass
    idx_start = text.find('[')
    idx_end = text.rfind(']')
    if idx_start != -1 and idx_end > idx_start:
        try:
            result = json.loads(text[idx_start:idx_end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    idx_obj_start = text.find('{')
    idx_obj_end = text.rfind('}')
    if idx_obj_start != -1 and idx_obj_end > idx_obj_start:
        try:
            obj = json.loads(text[idx_obj_start:idx_obj_end + 1])
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON array found")

def normalize_llm_entries(raw_list):
    if not isinstance(raw_list, list) or len(raw_list) == 0:
        return []
    if all(isinstance(x, dict) and "medicine" in x for x in raw_list):
        return raw_list
    if all(isinstance(x, str) for x in raw_list):
        return [{"medicine": raw_list[0],
                 "route": raw_list[1] if len(raw_list) > 1 else "oral",
                 "dosage and duration": raw_list[2] if len(raw_list) > 2 else ""}]
    results = []
    i = 0
    while i < len(raw_list):
        if isinstance(raw_list[i], str):
            entry = {"medicine": raw_list[i], "route": "oral", "dosage and duration": ""}
            i += 1
            while i < len(raw_list) and isinstance(raw_list[i], dict):
                entry.update(raw_list[i])
                i += 1
            results.append(entry)
        else:
            i += 1
    if results:
        return results
    return [x for x in raw_list if isinstance(x, dict) and "medicine" in x]

def extract_meds_from_llm(text):
    prompt = (
        "Extract ALL medicines from the prescription below.\n"
        "Output ONLY a valid JSON array. No explanation, no markdown, no extra text.\n"
        "Each element must have exactly these three keys:\n"
        '  "medicine"  : exact name from the text\n'
        '  "route"     : oral / injectable / topical / ocular / dental\n'
        '  "dosage and duration": e.g. "1 tablet morning and night for 5 days"\n\n'
        "Rules:\n"
        "- Extract EVERY medicine name you can find, even if OCR is messy.\n"
        "- Keep medicine name EXACTLY as written in the text.\n"
        "- Do NOT repeat medicines.\n"
        "- Output the JSON array and nothing else.\n\n"
        f"Prescription text:\n{text}\n\nJSON array:"
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"⚠ Groq API error: {e}")
        return []
    try:
        raw = extract_json_array(content)
        return normalize_llm_entries(raw)
    except Exception:
        print("⚠ Groq did not return valid JSON. Skipping.")
        return []
    