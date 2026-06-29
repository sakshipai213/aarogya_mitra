import pandas as pd
import re
from flask import Blueprint, request, jsonify
from rapidfuzz import process, fuzz
from utils.ocr import get_ocr_output_single_image
from utils.llm import extract_meds_from_llm
from utils.text_utils import (
    normalize_prescription_abbreviations,
    expand_medical_abbreviations,
    expand_frequency_codes,
    route_map,
    term_normalizer
)

generate_bp = Blueprint('generate', __name__)

MASTER_CSV = "Medicine_Details_revised.csv"
UPLOAD_PATH = "uploaded_rx.png"

def clean_medicine_name(name):
    if not name:
        return ""
    name = str(name).lower()
    name = re.sub(r'\b\d+(\.\d+)?\s*(mg|ml|mcg|g|kg|l|%)\b', '', name)
    name = re.sub(r'\b\d+(\.\d+)?/\d+(\.\d+)?\b', '', name)
    name = re.sub(r'\b(tablet|tab|cap|capsule|syrup|injection|ointment|cream|solution|drops|gel|powder|spray|pill|kit|paint|mouthwash|brush)\b', '', name)
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fuzzy_correct_med_name(name, master_list, threshold=0.55, prefer_tablet=True):
    cleaned_input = clean_medicine_name(name)
    if not cleaned_input:
        return name, 0.0
    cleaned_master = [clean_medicine_name(m) for m in master_list]
    best_match = process.extractOne(
        cleaned_input, cleaned_master,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=int(threshold * 100)
    )
    if not best_match:
        return name, 0.0
    _, score, idx = best_match
    candidate = master_list[idx]
    normalized_score = score / 100.0
    if prefer_tablet:
        tablet_indices = [i for i, m in enumerate(master_list) if "tablet" in m.lower()]
        tablet_cleaned = [cleaned_master[i] for i in tablet_indices]
        if tablet_cleaned:
            tablet_match = process.extractOne(cleaned_input, tablet_cleaned, scorer=fuzz.token_sort_ratio)
            if tablet_match and tablet_match[1] >= score - 2:
                candidate = master_list[tablet_indices[tablet_match[2]]]
                normalized_score = tablet_match[1] / 100.0
    return candidate, normalized_score

def remove_duplicate_entries(entries):
    seen = set()
    unique = []
    for entry in entries:
        key = entry.get("medicine", "").strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique

def run_pipeline_and_return_json(input_image_path, master_csv, match_threshold=0.55):
    print("📄 Running OCR...")
    ocr_text = get_ocr_output_single_image(input_image_path)
    if not ocr_text.strip():
        return []
    ocr_text = normalize_prescription_abbreviations(ocr_text)
    ocr_text = expand_medical_abbreviations(ocr_text)
    ocr_text = expand_frequency_codes(ocr_text)

    master_df = pd.read_csv(master_csv, low_memory=False)
    master_meds = master_df['Medicine Name'].dropna().astype(str).tolist()
    raw_entries = extract_meds_from_llm(ocr_text)

    all_results = []
    for entry in raw_entries:
        raw_name = entry.get("medicine", "").strip()
        if not raw_name:
            continue
        matched_name, score = fuzzy_correct_med_name(raw_name, master_meds, threshold=match_threshold)
        if score < 0.60:
            continue
        route = entry.get("route", "oral")
        corrected_lower = matched_name.lower()
        for wrong, right in term_normalizer.items():
            corrected_lower = corrected_lower.replace(wrong, right)
        for key, val in route_map.items():
            if corrected_lower.endswith(key) or (" " + key) in corrected_lower:
                route = val
                break
        all_results.append({
            "medicine": matched_name,
            "route": route,
            "dosage_and_duration": (entry.get("dosage and duration") or "").strip()
        })

    all_results = remove_duplicate_entries(all_results)
    return [{"medicine": m["medicine"], "route": m["route"], "dosage": m["dosage_and_duration"]} for m in all_results]

@generate_bp.route('/generate', methods=['POST'])
def generate_medicine_info():
    if 'image' not in request.files:
        return jsonify({"error": "Image file missing"}), 400
    img_file = request.files['image']
    img_file.save(UPLOAD_PATH)
    try:
        results = run_pipeline_and_return_json(UPLOAD_PATH, MASTER_CSV)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500