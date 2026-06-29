import pandas as pd
from flask import Blueprint, request, jsonify
from rapidfuzz import process

substitute_bp = Blueprint('substitute', __name__)

df_a = pd.read_csv("Medicine_Details_revised.csv")
df_b = pd.read_csv("PESURF_new_medicine_dataset_revised.csv", low_memory=False)

def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

meds_a = df_a["Medicine Name"].dropna().apply(normalize).unique().tolist()
set_meds_a = set(meds_a)

sub_columns = ["substitute0", "substitute1", "substitute2", "substitute3", "substitute4"]
subs_dict = {}
for _, row in df_b.iterrows():
    med_name = normalize(row["name"])
    substitutes = [normalize(row[col]) for col in sub_columns if pd.notna(row[col])]
    subs_dict[med_name] = [s for s in substitutes if s]

def fuzzy_match(query, choices, threshold=95, return_match=False):
    if not query:
        return None if return_match else False
    match_result = process.extractOne(query, choices)
    if match_result:
        match, score, _ = match_result
        if score >= threshold:
            return match if return_match else True
    return None if return_match else False

@substitute_bp.route("/find_substitute", methods=["POST"])
def find_substitute():
    data = request.json
    user_input = normalize(data.get("medicine_name", ""))
    matched_med = fuzzy_match(user_input, subs_dict.keys(), return_match=True)
    if not matched_med:
        return jsonify({"status": "error", "message": f"No match found for '{user_input}'."})
    substitutes = subs_dict.get(matched_med, [])
    valid_subs = []
    for sub in substitutes:
        match_in_a = fuzzy_match(sub, set_meds_a, return_match=True)
        if match_in_a:
            valid_subs.append(f"{sub} ➜ matched in A as '{match_in_a}'")
    return jsonify({"status": "success", "matched_med": matched_med, "valid_substitutes": valid_subs})