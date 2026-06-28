#!/usr/bin/env python
# coding: utf-8




# In[ ]:


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pyngrok import conf, ngrok
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from openai import OpenAI
import os

from rapidfuzz import process


import pandas as pd
import pytesseract
from PIL import Image as PILImage
import requests
import re
from gtts import gTTS
from moviepy.editor import *
from langdetect import detect
from collections import Counter
from difflib import SequenceMatcher
from difflib import get_close_matches
from IPython.display import Audio
import textdistance
import io
from moviepy.video.fx.all import loop
from moviepy.video.fx.all import resize


import cv2
import re
import json
from google.cloud import vision
#from PIL import Image


# In[ ]:


# Initialize Flask app
app = Flask(__name__)
CORS(app)



# Load CSV only once at app start
import pandas as pd
csv_path = "Medicine_Details_revised.csv"
df = pd.read_csv(csv_path)
df.fillna("", inplace=True)


# In[ ]:


GCP_CRED_PATH = "vision_api.json"
if not os.path.exists(GCP_CRED_PATH):
    raise FileNotFoundError("Google Vision credentials missing.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CRED_PATH
vision_client = vision.ImageAnnotatorClient()

LLM_MODEL = "mistralai/mistral-7b-instruct"
OPENROUTER_API_KEY = "your_api_key_here"  # 🔒 Replace this with your actual OpenRouter API key

route_map = {
    "cap": "oral", "tab=taz=tas": "oral", "tablet": "oral", "capsule": "oral",
    "syrup": "oral", "injection": "injectable", "ointment": "topical",
    "paint": "topical", "mouth wash": "oral rinse", "mouthwash": "oral rinse",
    "gel": "topical", "eyedrop": "ocular", "eye drop": "ocular", "drops": "ocular",
    "brush": "dental", "toothpaste": "dental", "paste": "dental","tas":"oral","taz":"oral"
}
term_normalizer = {
    "tas": "tab",
    "taz": "tab",
    "tab.": "tab",
    "drip": "drops",
    "dryps": "drops",
    "drips": "drops",
    "cap.": "cap",
    "inj.": "injection",
    "oint.": "ointment",
    "syp": "syrup",
    "mouthwash": "mouth wash",
    "eyedrop": "eye drop"
}


def extract_bracket_regions(image_path, output_dir="bracket_groups"):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
    closed1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)
    contours1, _ = cv2.findContours(closed1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dilated = cv2.dilate(edges, None, iterations=2)
    contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def filter_contours(contours, method=1):
        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / (w + 1e-5)
            area = cv2.contourArea(cnt)
            if method == 1:
                if h > 80 and aspect_ratio > 3 and area > 200:
                    results.append((x, y, w, h))
            else:
                if h > 60 and 1.5 < aspect_ratio < 20 and area > 150:
                    if x < img_w * 0.25 or x > img_w * 0.75:
                        results.append((x, y, w, h))
        return results

    def dedupe_brackets(brackets, iou_thresh=0.4):
        final = []
        for (x, y, w, h) in brackets:
            add = True
            for (fx, fy, fw, fh) in final:
                xa, ya = max(x, fx), max(y, fy)
                xb, yb = min(x + w, fx + fw), min(y + h, fy + fh)
                inter_area = max(0, xb - xa) * max(0, yb - ya)
                union_area = w * h + fw * fh - inter_area
                iou = inter_area / float(union_area + 1e-5)
                if iou > iou_thresh:
                    add = False
                    break
            if add:
                final.append((x, y, w, h))
        return final

    brackets = dedupe_brackets(filter_contours(contours1, 1) + filter_contours(contours2, 2))

    crops = []
    for i, (x, y, w, h) in enumerate(brackets):
        top = max(0, y - 10)
        bottom = min(img_h, y + h + 10)
        crop = image[top:bottom, :]
        fname = f"{output_dir}/bracket_{i+1}.png"
        cv2.imwrite(fname, crop)
        crops.append(fname)

    used_y = sorted([(y, y + h) for (_, y, _, h) in brackets])
    merged = []
    for start, end in used_y:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    unused = []
    prev_end = 0
    for start, end in merged:
        if start - prev_end > 30:
            unused.append((prev_end, start))
        prev_end = end
    if img_h - prev_end > 30:
        unused.append((prev_end, img_h))

    for i, (uy1, uy2) in enumerate(unused):
        region = image[uy1:uy2, :]
        if region.shape[0] > 30:
            fname = f"{output_dir}/leftover_{i+1}.png"
            cv2.imwrite(fname, region)
            crops.append(fname)

    return crops

def get_ocr_output(folder):
    output = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if os.path.isdir(fpath) or fname.startswith('.'):
            continue
        with open(fpath, "rb") as img_file:
            content = img_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        text = response.text_annotations[0].description if response.text_annotations else ""
        output.append((fname, text.strip()))
    return output

def extract_meds_from_llm(text, api_key):
    prompt = f"""You are a medical NLP assistant. Extract medicine entries from the following prescription text.

Your task is to extract *only unique* medicine entries in the format below:
[
  {{
    "medicine": "Name of the medicine found in the csv file",
    "route": "oral, injectable, topical, etc.",
    "dosage and duration": "e.g. 1 tablet in the morning and night for 5 days"
  }}
]

Rules:
- Convert frequency codes: "1-0-1" → "morning and night", "1-1-1" → "morning, afternoon, and night", etc.
- REFER TO THE ROUTE MAP FOR ROUTE
- DO NOT CHANGE THE FORMAT
- Apply duration to all meds if mentioned once.
- Remove brand names.
- Output valid JSON only.
{text}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, json=payload)
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content']

    try:
        start = content.find('[')
        end = content.rfind(']') + 1
        return json.loads(content[start:end])
    except:
        print("⚠ Failed to parse JSON from LLM response:\n", content)
        return [{"medicine": "Unidentified", "route": None, "dosage and duration": None}]

def refine_with_mistral(entries, api_key):
    meds_for_llm = [{
        "medicine": e["corrected_medicine"],
        "route": e.get("route", ""),
        "dosage and duration": e.get("dosage and duration", "")
    } for e in entries]

    prompt = f"""You are a medical NLP assistant. Given the following extracted and corrected medicines, return a clean, valid JSON array with:
- Only unique generic medicine names
- Route (oral, topical, etc.)
- DO NOT CHANGE THE FORMAT OF JSON, JUST MEDICINE,ROUTE , DOSAGE AND DURATION

Input:
{json.dumps(meds_for_llm, indent=2)}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        start = content.find("[")
        end = content.rfind("]") + 1
        return json.loads(content[start:end])
    except Exception as e:
        print(f"⚠ Final Mistral refinement failed: {e}")
        return entries



# === HELPERS ===
def clean_medicine_name(name):
    name = name.lower()
    name = re.sub(r'\b\d+(\.\d+)?\s*(mg|ml|mcg|g|kg|l|%)\b', '', name)
    name = re.sub(r'\b\d+(\.\d+)?/\d+(\.\d+)?\b', '', name)
    name = re.sub(r'\b(tablet|tab|cap|capsule|syrup|injection|ointment|cream|solution|drops|gel|powder|spray|pill|kit|paint|mouthwash|brush)\b', '', name)
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fuzzy_correct_med_name(name, master_list, threshold=0.80):
    cleaned_input = clean_medicine_name(name)
    best_match = name
    best_score = 0
    for candidate in master_list:
        candidate_clean = clean_medicine_name(candidate)
        score = textdistance.jaro_winkler.normalized_similarity(cleaned_input, candidate_clean)
        if score > best_score:
            best_match = candidate
            best_score = score
    return best_match if best_score >= threshold else name

def filter_llm_output_against_ocr(raw_entries, ocr_text):
    ocr_text_lower = ocr_text.lower()
    return [entry for entry in raw_entries if entry.get("medicine", "").lower() in ocr_text_lower]

def remove_duplicate_entries(entries):
    seen = set()
    unique_entries = []
    for entry in entries:
        name_key = entry.get("corrected_medicine", "").strip().lower()
        if name_key not in seen:
            seen.add(name_key)
            unique_entries.append(entry)
    return unique_entries


'''def format_meds_as_text(meds):
    return json.dumps([{
        "medicine": m.get("corrected_medicine", ""),
        "route": m.get("route", ""),
        "dosage and duration": m.get("dosage and duration", "")
    } for m in meds], indent=2)'''


# In[ ]:


# Constants
MASTER_CSV = "Medicine_Details_revised.csv"
UPLOAD_PATH = "uploaded_rx.png"
OPENROUTER_API_KEY = "your_api_key_here"  # 🔒 Replace this


@app.route('/generate', methods=['POST'])
def generate_medicine_info():
    if 'image' not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    img_file = request.files['image']
    img_file.save(UPLOAD_PATH)

    try:
        results = run_pipeline_and_return_json(
            input_image_path=UPLOAD_PATH,
            master_csv=MASTER_CSV,
            api_key=OPENROUTER_API_KEY
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_pipeline_and_return_json(input_image_path, master_csv, api_key):
    extract_bracket_regions(input_image_path)
    ocr_output = get_ocr_output("bracket_groups")

    master_df = pd.read_csv(master_csv)
    master_meds = master_df['Medicine Name'].str.lower().tolist()

    all_results = []
    for _, text in ocr_output:
        if not text.strip():
            continue
        raw_entries = extract_meds_from_llm(text, api_key)
        filtered_entries = filter_llm_output_against_ocr(raw_entries, text)
        for entry in filtered_entries:
            raw_name = entry.get("medicine", "")
            corrected = fuzzy_correct_med_name(raw_name, master_meds)
            entry["corrected_medicine"] = corrected

            route = entry.get("route", "").lower()
            corrected_lower = corrected.lower()

            for wrong, right in term_normalizer.items():
                corrected_lower = corrected_lower.replace(wrong, right)

            for key, val in route_map.items():
                if corrected_lower.endswith(key):
                    entry["route"] = val
                    break
            else:
                if not route:
                    entry["route"] = "unspecified"

            all_results.append(entry)

    all_results = remove_duplicate_entries(all_results)
    return [ {
        "medicine": m.get("corrected_medicine", ""),
        "route": m.get("route", ""),
        "dosage": m.get("dosage and duration", "")
    } for m in all_results ]



# In[ ]:


# backend.py



# --- Load CSVs ---
df_a = pd.read_csv("Medicine_Details_revised.csv")
df_b = pd.read_csv("PESURF_new_medicine_dataset_revised.csv")

# --- Normalize helper ---
def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

# --- Preprocess Dataset A ---
meds_a = df_a["Medicine Name"].dropna().apply(normalize).unique().tolist()
set_meds_a = set(meds_a)

# --- Prepare substitute dictionary from Dataset B ---
sub_columns = ["substitute0", "substitute1", "substitute2", "substitute3", "substitute4"]
subs_dict = {}
for _, row in df_b.iterrows():
    med_name = normalize(row["name"])
    substitutes = [normalize(row[col]) for col in sub_columns if pd.notna(row[col])]
    subs_dict[med_name] = [sub for sub in substitutes if sub]

# --- Fuzzy match function ---
def fuzzy_match(query, choices, threshold=95, return_match=False):
    if not query:
        return None if return_match else False
    match_result = process.extractOne(query, choices)
    if match_result:
        match, score, _ = match_result
        if score >= threshold:
            return match if return_match else True
    return None if return_match else False

# --- API endpoint ---
@app.route("/find_substitute", methods=["POST"])
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

    return jsonify({
        "status": "success",
        "matched_med": matched_med,
        "valid_substitutes": valid_subs
    })

#if __name__ == "__main__":
#   app.run(debug=True)


# In[ ]:


api_key = "your_api_key_here"  # 🔁 Replace with your actual Nebius key
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ["OPENAI_API_KEY"]
)


model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def convert_digits(text, language='hindi'):
    digit_maps = {
        'hindi':    {'0': '०', '1': '१', '2': '२', '3': '३', '4': '४', '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'},
        'marathi':  {'0': '०', '1': '१', '2': '२', '3': '३', '4': '४', '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'},
        'kannada':  {'0': '೦', '1': '೧', '2': '೨', '3': '೩', '4': '೪', '5': '೫', '6': '೬', '7': '೭', '8': '೮', '9': '೯'},
        'tamil':    {'0': '௦', '1': '௧', '2': '௨', '3': '௩', '4': '௪', '5': '௫', '6': '௬', '7': '௭', '8': '௮', '9': '௯'},
        'telugu':   {'0': '౦', '1': '౧', '2': '౨', '3': '౩', '4': '౪', '5': '౫', '6': '౬', '7': '౭', '8': '౮', '9': '౯'},
        'malayalam':{'0': '൦', '1': '൧', '2': '൨', '3': '൩', '4': '൪', '5': '൫', '6': '൬', '7': '൭', '8': '൮', '9': '൯'},
        'gujarati': {'0': '૦', '1': '૧', '2': '૨', '3': '૩', '4': '૪', '5': '૫', '6': '૬', '7': '૭', '8': '૮', '9': '૯'},
        'bengali':  {'0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪', '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'},
        'urdu':     {'0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴', '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'}
    }
    digit_map = digit_maps.get(language.lower(), {})
    return ''.join(digit_map.get(ch, ch) for ch in text)


# In[ ]:


# STEP 3: Define /generate route

@app.route('/generate-text', methods=['POST'])
def generate():
    data = request.json
    structured_input = data['structured_input']
    language = data['language'].lower()

    # Step 1: Prompt Meta LLaMA-3.1 (Nebius)
    prompt = (
    "You are a medical assistant. Convert the structured medicine instructions into very simple, spoken-style English. "
    "Present each instruction as a numbered list starting from 1, with no introductions, no summaries, and no extra comments. "
    "Only output the instructions themselves.\n"
    "Example:\n"
    "Input: paracetamol 500mg;orally;morning and night x3 weeks\n"
    "Output: Take 1 Paracetamol 500 mg tablet by mouth in the morning and again at night and keep doing this for 3 weeks.\n"
    "Input: Phexin 500mg Capsule;oral;1-0-1 X for 5 days\n"
    "Output: Take 1 Phexin 500 mg tablet by mouth in the morning and at night and keep doing this for 5 days.\n"
    "Input: Zerodol-P tablet; orally; 1-0-1 for 5 days\n"
    "Output: Take 1 Zerodol-p tablet by mouth in the morning and at night and keep doing this for 5 days.\n"
    "Input:Stolin gum paint; topical; not specified for 5 days.\n"
    "Output: Use stolin gum paint on your gums for 5 days.\n"
    "Input:Colgate Plax mouthwash; oral rinse; not specified for 5 day.\n"
    "Output:Use Colgate plax mouthwash to rinse your mouth for 5 days.\n"
    "Input: Oral B Pro 2; ;not specified for 1 week.\n"
    "Output: Use Oral B Pro 2 for 1 week.\n"
    "Input: ciprofloxacin;eye;2 drops four times a day x7 days\n"
    "Output: Put 2 drops of Ciprofloxacin into your eyes four times a day. Keep doing this for 7 days.\n"
    "Input: chlorhexidine;mouth wash;10ml twice a day\n"
    "Output: Rinse your mouth with Chlorhexidine mouthwash 10 ml twice a day.\n"
    "Input: asthalin;inhaler; every 6hrs x10 days\n"
    "Output: Use the Asthalin inhaler by placing the open end in your mouth, pressing it once as you breathe in slowly and do this every 6 hours for 10 days.\n"
    "Input: avamys;nasal spray;twice a day x7 days\n"
    "Output: Spray Avamys nasal spray into each nostril twice a day. Do this for 7 days.\n"
    "Input: amorfine;cream; three times a day x7 days\n"
    "Output: Apply Amorfine cream three times a day and do this for 7 days.\n"
    "Input:phexin 500mg capsule; oral; 1 tablet in the morning and night for 5 days\n"
    "Output: Take 1 Phexin 500 mg tablet by mouth in the morning and again at night and keep doing this for 5 days.\n"
    "Input: zerodol tablet; oral; 1 tablet in the morning and night for 5 days\n"
    "Output: Take 1 Zeodol tablet by mouth in the morning and again at night and keep doing this for 5 days.\n"
    "Input: stolin gum paint; topical; used for 5 days\n"
    "Output: Apply Stolin gum paint on your gums for 5 days.\n"
    "Input: colgate plax mouthwash; oral rinse; for 1 week\n"
    "Output: Use Colgate Plax mouthwash to rinse your mouth for 1 week.\n"
    "Input: oral b pro electric brush; dental; not specified\n"
    "Output: Use Oral B Pro electric brush to gently brush your teeth.\n"

    f"Input: {structured_input}\nOutput:"
)


    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.6,
        top_p=0.9
    )

    english_instruction = response.choices[0].message.content.strip()

    # Step 2: Translate to Indian Language (NLLB)
    lang_code_map = {
        'hindi': 'hin_Deva',
        'marathi': 'mar_Deva',
        'gujarati': 'guj_Gujr',
        'bengali': 'ben_Beng',
        'urdu': 'urd_Arab',
        'tamil': 'tam_Taml',
        'kannada': 'kan_Knda',
        'telugu': 'tel_Telu',
        'malayalam': 'mal_Mlym',
    }


    tgt_lang = lang_code_map.get(language, 'hin_Deva')  # default to Hindi
    tokenizer.src_lang = "eng_Latn"

    english_instruction_lower = english_instruction.lower()

    inputs = tokenizer(english_instruction_lower, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    translated_text = convert_digits(translated_text, language=language)

    return jsonify({
        "english_instruction": english_instruction,
        "translated_instruction": translated_text
    })


# In[ ]:


@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        #  1. Get uploaded image file
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        #image = PILImage.open(file.stream).convert("RGB")
        # Save uploaded image to disk
        uploaded_pil = PILImage.open(file.stream).convert("RGB")
        image_path = "uploaded_prescription.jpg"
        uploaded_pil.save(image_path)

# Now you can use image_path as needed later
        image = uploaded_pil  # If you still want to use it as a PIL Image


        #  2. OCR + Language Detection (same as your code)
        ocr_text = pytesseract.image_to_string(image,lang='eng+hin+tam+tel+kan+mal+guj+ben+mar+pan+urd')
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        detected_langs = []
        for line in lines:
            try:
                lang = detect(line)
                detected_langs.append(lang)
            except:
                pass


        lang_counter = Counter(detected_langs)
        non_english_langs = [lang for lang in detected_langs if lang != 'en']
        primary_native_lang = Counter(non_english_langs).most_common(1)[0][0] if non_english_langs else 'en'

        def is_english(text):
            return all(ord(char) < 128 or char in "।।“”‘’" for char in text)

        english_lines = [line.strip() for line in ocr_text.split('\n') if is_english(line) and line.strip()]
        instruction_lines = [line for line in english_lines if re.match(r'^\d+[\).]', line)]

        ner_pipeline = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")



        def extract_instruction_from_ner(text):
            result = {'quantity': None, 'unit': None, 'medicine': None, 'route': None, 'frequency': None, 'duration': None}

            number_words = {
                "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
            }

            lowered = text.lower()

            #  1. Quantity + Unit (digit or word-number)
            qty_unit_pattern = r'\b(\d+|' + '|'.join(number_words.keys()) + r')\s*(drops?|ml|tablets?|capsules?|pills?|puffs?|injections?)\b'
            qty_unit = re.search(qty_unit_pattern, lowered)
            if qty_unit:
                qty = qty_unit.group(1)
                result['quantity'] = number_words.get(qty, qty)
                result['unit'] = qty_unit.group(2)

    # 2. Match standalone units after medicine name (even if no quantity)
            unit_keywords = ['tablet', 'capsule', 'inhaler', 'cream', 'spray', 'gel', 'ointment', 'shampoo',
                            'suspension', 'solution', 'mouthwash', 'lotion', 'drops?', 'patch']
            unit_regex = r'\b(' + '|'.join(unit_keywords) + r')\b'
            unit_match = re.search(unit_regex, lowered)
            if not result['unit'] and unit_match:
                result['unit'] = unit_match.group(1)

    # 3. Route
            for route in ['mouth', 'ear', 'eye', 'eyes', 'nose', 'skin', 'nostril', 'nostrils']:
                if route in lowered:
                    result['route'] = route.rstrip("s")
                    break

    #  4. Frequency
            freq_match = re.search(r'(every\s+\d+\s+\w+|once a day|twice a day|thrice a day|\d+\s+times a day)', lowered)
            if freq_match:
                result['frequency'] = freq_match.group(1)

    # 5. Duration
            dur_match = re.search(r'(?:for|do this for|keep doing this for)\s+(\d+)\s*(days?|weeks?|months?)', lowered)
            if dur_match:
                result['duration'] = f"{dur_match.group(1)} {dur_match.group(2)}"

            return result

# Load your CSV only once
        df = pd.read_csv('Medicine_Details.csv')  # Replace with actual filename
        df.fillna("", inplace=True)

# Build lowercase sets for matching
        medicine_names = df['Medicine Name'].str.lower().unique().tolist()
        composition_words = df['Composition'].str.lower().str.split('[,+/]')  # split multiple compositions
        composition_flat = set(word.strip() for sublist in composition_words for word in sublist if word.strip())

# Combine both into a searchable list
        all_possible_names = set(medicine_names) | composition_flat


        def extract_possible_medicine(text):
            text_lower = text.lower()

    # Priority 1: After "of"
            match = re.search(r'\bof\s+([a-zA-Z][a-zA-Z0-9-]+)', text_lower)
            if match:
                token = match.group(1)
                #print(f"🔍 Trying after 'of': '{token}'")
                matches = get_close_matches(token, all_possible_names, n=1, cutoff=0.6)
                if matches:
                    print(f"Matched: {token} → {matches[0]}")
                    return matches[0]

    # Priority 2: After 'take', 'put', 'apply', etc.
            match = re.search(r'\b(take|put|apply|insert|use)\s+([a-zA-Z][a-zA-Z0-9-]+)', text_lower)
            if match:
                token = match.group(2)
                #print(f"🔍 Trying after '{match.group(1)}': '{token}'")
                matches = get_close_matches(token, all_possible_names, n=1, cutoff=0.6)
                if matches:
                    print(f"Matched: {token} → {matches[0]}")
                    return matches[0]

    # Optional fallback: longest capitalized token
            capitalized_tokens = re.findall(r'\b([A-Z][a-zA-Z0-9-]{2,})\b', text)
            for token in capitalized_tokens:
                #print(f"🔍 Trying capitalized token: '{token}'")
                matches = get_close_matches(token.lower(), all_possible_names, n=1, cutoff=0.6)
                if matches:
                    #print(f" Matched: {token} → {matches[0]}")
                    return matches[0]

            #print(" No medicine match found.")
            return None

# ✅ Cleaning function for names
        def clean_medicine_name(name):
            name = name.lower()

    # Remove dosage-related suffixes like 500mg, 5 ml, 2mg/5ml, etc.
            name = re.sub(r'\b\d+(\.\d+)?\s*(mg|ml|mcg|g|kg|l|%)\b', '', name)
            name = re.sub(r'\b\d+(\.\d+)?\s*/\s*\d+(\.\d+)?\s*(mg|ml|mcg|%)?\b', '', name)
            name = re.sub(r'\b\d+\b', '', name)  # Remove standalone numbers (e.g., 25, 500)

    # Remove form keywords like tablet, capsule, etc.
            name = re.sub(r'\b(tablet|syrup|capsule|injection|suspension|ointment|cream|solution|drops|gel|powder|spray|pill|kit)\b', '', name)

    # Normalize whitespace
            name = re.sub(r'\s+', ' ', name).strip()

            return name

# ✅ Clean and prepare CSV
        df['clean_name'] = df['Medicine Name'].astype(str).apply(clean_medicine_name)
        cleaned_medicine_names = df['clean_name'].tolist()
        original_names = df['Medicine Name'].tolist()

# ✅ Damerau-Levenshtein matching logic


        def get_best_match_damerau_levenshtein(ocr_text, dataset_list, score_threshold=0.6):
            ocr_clean = clean_medicine_name(ocr_text)
            best_score = -1
            best_match = None

            for candidate in dataset_list:
                candidate_clean = clean_medicine_name(candidate)
                sim_score = textdistance.damerau_levenshtein.normalized_similarity(ocr_clean, candidate_clean)
                if sim_score > best_score:
                    best_score = sim_score
                    best_match = candidate

            if best_score >= score_threshold:
                return best_match
            else:
                return ocr_text  # fallback

# ✅ Wrapper for integrations
        def fuzzy_match_medicine_from_csv(raw_name, route, df):
            return get_best_match_damerau_levenshtein(raw_name, df['Medicine Name'].tolist())


        def fetch_medicine_image(med_name, df):
            match_row = df[
                df['Medicine Name'].str.lower().str.contains(med_name.lower()) |
                df['Composition'].str.lower().str.contains(med_name.lower())
            ]
            if not match_row.empty:
                image_url = match_row.iloc[0]['Image URL']
                img_data = requests.get(image_url).content
                with open("med_img.png", "wb") as f:
                    f.write(img_data)
                return "med_img.png"
            return None

        nllb_model_id = "facebook/nllb-200-1.3B"
        nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_id)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_id)


        # ✅ Set NLLB + gTTS language
        lang_code_map = {
            "hi": {"nllb": "hin_Deva", "gtts": "hi"},
            "ta": {"nllb": "tam_Taml", "gtts": "ta"},
            "kn": {"nllb": "kan_Knda", "gtts": "kn"},
            "te": {"nllb": "tel_Telu", "gtts": "te"},
            "ml": {"nllb": "mal_Mlym", "gtts": "ml"},
            "mr": {"nllb": "mar_Deva", "gtts": "mr"},
            "gu": {"nllb": "guj_Gujr", "gtts": "gu"},
            "bn": {"nllb": "ben_Beng", "gtts": "bn"},
            "ur": {"nllb": "urd_Arab", "gtts": "ur"},
        }

        if primary_native_lang in lang_code_map:
            nllb_lang = lang_code_map[primary_native_lang]["nllb"]
            tts_lang = lang_code_map[primary_native_lang]["gtts"]
        else:
            return jsonify({"error": f"Unsupported or undetected language: {primary_native_lang}"}), 400

        def clean_instruction(instr):
            return re.sub(r'^\d+[\).\s]+', '', instr).strip()

        final_clips = []



        # ✅ Inject the rest of your code BELOW (from cleaning English lines to building final video)
        # Make sure to replace:
        # - `image_path = "/content/z15.png"` → use `image` from above
        # - `df` → already loaded globally
        # - `instruction_lines` → extract from ocr_text like you already do

        # 🔽 Your full core logic here — no changes needed to main logic
        # Just ensure it uses `image` and `df`, and at the end:

        for i, instr in enumerate(instruction_lines):
            #print(f"\n Processing Instruction {i+1}: {instr}")

            cleaned_instr = clean_instruction(instr)

            # 1. Extract NER
            parsed = extract_instruction_from_ner(instr)

            # 2. Medicine name
            raw_med_name = extract_possible_medicine(instr)
            if raw_med_name:
                corrected = fuzzy_match_medicine_from_csv(raw_med_name, parsed['route'], df)
                parsed['medicine'] = corrected
            else:
                parsed['medicine'] = "Unknown"

            #print(" Parsed:", parsed)

            # 3. Translate using NLLB
            '''inputs = nllb_tokenizer(instr, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tokenizer.lang_code_to_id[nllb_lang]
            )
            translated_text = nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            print(f" Translated ({primary_native_lang}): {translated_text}")'''

            cleaned_instr = clean_instruction(instr.strip().replace("\n", " "))


            # Translate using NLLB
            inputs = nllb_tokenizer(cleaned_instr, return_tensors="pt")
            translated_tokens = nllb_model.generate(
                **inputs,
                max_length=256,
                no_repeat_ngram_size=2,
                forced_bos_token_id=nllb_tokenizer.lang_code_to_id[nllb_lang]
            )
            translated_text = nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            #print(f" Translated ({primary_native_lang}): {translated_text}")


            # 4. gTTS Audio from translated instruction
            tts = gTTS(text=translated_text, lang=tts_lang)
            audio_path = f"audio_{i}.mp3"
            tts.save(audio_path)
            audio = AudioFileClip(audio_path)

            # 5. Choose animation
            if parsed['unit']:
                unit = parsed['unit'].lower()
                if 'drop' in unit:
                    if parsed['route'] == 'ear':
                        video_path = "ear_drop.mp4"
                    elif parsed['route'] == 'eye':
                        video_path = "eye_drop.mp4"
                    else:
                        video_path = "generic_drop.mp4"
                elif any(u in unit for u in ['tablet', 'pill', 'capsule']):
                    video_path = "pill.mp4"
                elif any(u in unit for u in ['cream', 'ointment', 'gel']):
                    video_path = "cream_application.mp4"
                elif any(u in unit for u in ['inhaler', 'puff', 'spray']):
                    video_path = "inhaler.mp4"
                else:
                    video_path = "default.mp4"
            else:
                video_path = "default.mp4"

            # 6. Fetch medicine image
            med_image_path = fetch_medicine_image(parsed['medicine'], df)

            # 7. Build animation sequence
            clip = VideoFileClip(video_path).resize(height=400)
            

        #  Step 1: Determine how many times to repeat to match/exceed audio duration
            n = int(parsed['quantity']) if parsed['quantity'] else 1
            min_required_duration = audio.duration
            base_clip = clip.without_audio()

        # If repeating clip isn't enough, loop to match audio length
            repeated_clip = concatenate_videoclips([base_clip] * n)
            if repeated_clip.duration < min_required_duration:
                repeated_clip = loop(base_clip, duration=min_required_duration)

        # Match durations exactly and set audio
            anim_sequence = repeated_clip.set_duration(min_required_duration).set_audio(audio)


            # 8. Image on left
            if med_image_path:
                med_img = PILImage.open(med_image_path).convert('RGB').resize((int(clip.w), clip.h))
                med_img.save("resized_med.jpg")
                left_clip = ImageClip("resized_med.jpg").set_duration(audio.duration)
            else:
                left_clip = ColorClip(size=(clip.w, clip.h), color=(255, 255, 255)).set_duration(audio.duration)

            # 9. Combine image + animation
            combined = clips_array([[left_clip, anim_sequence]])
            final_clips.append(combined)

        final_video = concatenate_videoclips(final_clips, method="compose")
        final_video.write_videofile("final_instruction_video.mp4", fps=24, codec="libx264", audio_codec="aac")

        return send_file("final_instruction_video.mp4", mimetype="video/mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 500








if __name__ == "__main__":
    app.run(port=5000, debug=True)
