import os
import re
import requests
import pandas as pd
import pytesseract
import textdistance

from flask import Blueprint, request, jsonify, send_file
from PIL import Image as PILImage
from gtts import gTTS
from langdetect import detect
from collections import Counter
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, ColorClip,
    concatenate_videoclips, clips_array
)
from moviepy.video.fx.all import loop

generate_video_bp = Blueprint('generate_video', __name__)

# ── Load NLLB model once at startup (not inside the route) ──────────────────
nllb_model_id = "facebook/nllb-200-1.3B"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_id)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_id)

LANG_CODE_MAP = {
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

# ── Helper functions ─────────────────────────────────────────────────────────

def is_english(text):
    return all(ord(char) < 128 or char in "।।""''" for char in text)

def clean_instruction(instr):
    return re.sub(r'^\d+[\).\s]+', '', instr).strip()

def clean_medicine_name(name):
    name = name.lower()
    name = re.sub(r'\b\d+(\.\d+)?\s*(mg|ml|mcg|g|kg|l|%)\b', '', name)
    name = re.sub(r'\b\d+(\.\d+)?\s*/\s*\d+(\.\d+)?\s*(mg|ml|mcg|%)?\b', '', name)
    name = re.sub(r'\b\d+\b', '', name)
    name = re.sub(r'\b(tablet|syrup|capsule|injection|suspension|ointment|cream|solution|drops|gel|powder|spray|pill|kit)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

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
    return best_match if best_score >= score_threshold else ocr_text

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

def extract_instruction_from_ner(text):
    result = {'quantity': None, 'unit': None, 'medicine': None, 'route': None, 'frequency': None, 'duration': None}
    number_words = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    lowered = text.lower()

    qty_unit_pattern = r'\b(\d+|' + '|'.join(number_words.keys()) + r')\s*(drops?|ml|tablets?|capsules?|pills?|puffs?|injections?)\b'
    qty_unit = re.search(qty_unit_pattern, lowered)
    if qty_unit:
        qty = qty_unit.group(1)
        result['quantity'] = number_words.get(qty, qty)
        result['unit'] = qty_unit.group(2)

    unit_keywords = ['tablet', 'capsule', 'inhaler', 'cream', 'spray', 'gel', 'ointment', 'shampoo',
                     'suspension', 'solution', 'mouthwash', 'lotion', 'drops?', 'patch']
    unit_match = re.search(r'\b(' + '|'.join(unit_keywords) + r')\b', lowered)
    if not result['unit'] and unit_match:
        result['unit'] = unit_match.group(1)

    for route in ['mouth', 'ear', 'eye', 'eyes', 'nose', 'skin', 'nostril', 'nostrils']:
        if route in lowered:
            result['route'] = route.rstrip("s")
            break

    freq_match = re.search(r'(every\s+\d+\s+\w+|once a day|twice a day|thrice a day|\d+\s+times a day)', lowered)
    if freq_match:
        result['frequency'] = freq_match.group(1)

    dur_match = re.search(r'(?:for|do this for|keep doing this for)\s+(\d+)\s*(days?|weeks?|months?)', lowered)
    if dur_match:
        result['duration'] = f"{dur_match.group(1)} {dur_match.group(2)}"

    return result

def extract_possible_medicine(text, all_possible_names):
    text_lower = text.lower()

    match = re.search(r'\bof\s+([a-zA-Z][a-zA-Z0-9-]+)', text_lower)
    if match:
        token = match.group(1)
        matches = get_close_matches(token, all_possible_names, n=1, cutoff=0.6)
        if matches:
            return matches[0]

    match = re.search(r'\b(take|put|apply|insert|use)\s+([a-zA-Z][a-zA-Z0-9-]+)', text_lower)
    if match:
        token = match.group(2)
        matches = get_close_matches(token, all_possible_names, n=1, cutoff=0.6)
        if matches:
            return matches[0]

    for token in re.findall(r'\b([A-Z][a-zA-Z0-9-]{2,})\b', text):
        matches = get_close_matches(token.lower(), all_possible_names, n=1, cutoff=0.6)
        if matches:
            return matches[0]

    return None

# ── Route ────────────────────────────────────────────────────────────────────

@generate_video_bp.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # 1. Save uploaded image
        file = request.files['image']
        uploaded_pil = PILImage.open(file.stream).convert("RGB")
        image_path = "uploaded_prescription.jpg"
        uploaded_pil.save(image_path)
        image = uploaded_pil

        # 2. OCR + language detection
        ocr_text = pytesseract.image_to_string(
            image, lang='eng+hin+tam+tel+kan+mal+guj+ben+mar+pan+urd'
        )
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        detected_langs = []
        for line in lines:
            try:
                detected_langs.append(detect(line))
            except:
                pass

        non_english_langs = [lang for lang in detected_langs if lang != 'en']
        primary_native_lang = Counter(non_english_langs).most_common(1)[0][0] if non_english_langs else 'en'

        english_lines = [line.strip() for line in ocr_text.split('\n') if is_english(line) and line.strip()]
        instruction_lines = [line for line in english_lines if re.match(r'^\d+[\).]', line)]

        # 3. Load CSV + build medicine name sets
        df = pd.read_csv('Medicine_Details.csv')
        df.fillna("", inplace=True)
        df['clean_name'] = df['Medicine Name'].astype(str).apply(clean_medicine_name)

        medicine_names = df['Medicine Name'].str.lower().unique().tolist()
        composition_words = df['Composition'].str.lower().str.split('[,+/]')
        composition_flat = set(word.strip() for sublist in composition_words for word in sublist if word.strip())
        all_possible_names = set(medicine_names) | composition_flat

        # 4. NER pipeline
        ner_pipeline = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

        # 5. Language check
        if primary_native_lang not in LANG_CODE_MAP:
            return jsonify({"error": f"Unsupported or undetected language: {primary_native_lang}"}), 400
        nllb_lang = LANG_CODE_MAP[primary_native_lang]["nllb"]
        tts_lang = LANG_CODE_MAP[primary_native_lang]["gtts"]

        # 6. Build video clips
        final_clips = []

        for i, instr in enumerate(instruction_lines):
            parsed = extract_instruction_from_ner(instr)

            raw_med_name = extract_possible_medicine(instr, all_possible_names)
            if raw_med_name:
                parsed['medicine'] = fuzzy_match_medicine_from_csv(raw_med_name, parsed['route'], df)
            else:
                parsed['medicine'] = "Unknown"

            cleaned_instr = clean_instruction(instr.strip().replace("\n", " "))

            # Translate
            inputs = nllb_tokenizer(cleaned_instr, return_tensors="pt")
            translated_tokens = nllb_model.generate(
                **inputs,
                max_length=256,
                no_repeat_ngram_size=2,
                forced_bos_token_id=nllb_tokenizer.lang_code_to_id[nllb_lang]
            )
            translated_text = nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

            # Audio
            tts = gTTS(text=translated_text, lang=tts_lang)
            audio_path = f"audio_{i}.mp3"
            tts.save(audio_path)
            audio = AudioFileClip(audio_path)

            # Choose animation video
            if parsed['unit']:
                unit = parsed['unit'].lower()
                if 'drop' in unit:
                    video_path = "ear_drop.mp4" if parsed['route'] == 'ear' else \
                                 "eye_drop.mp4" if parsed['route'] == 'eye' else "generic_drop.mp4"
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

            # Fetch medicine image
            med_image_path = fetch_medicine_image(parsed['medicine'], df)

            # Build clip
            clip = VideoFileClip(video_path).resize(height=400)
            n = int(parsed['quantity']) if parsed['quantity'] else 1
            min_required_duration = audio.duration
            base_clip = clip.without_audio()

            repeated_clip = concatenate_videoclips([base_clip] * n)
            if repeated_clip.duration < min_required_duration:
                repeated_clip = loop(base_clip, duration=min_required_duration)

            anim_sequence = repeated_clip.set_duration(min_required_duration).set_audio(audio)

            if med_image_path:
                med_img = PILImage.open(med_image_path).convert('RGB').resize((int(clip.w), clip.h))
                med_img.save("resized_med.jpg")
                left_clip = ImageClip("resized_med.jpg").set_duration(audio.duration)
            else:
                left_clip = ColorClip(size=(clip.w, clip.h), color=(255, 255, 255)).set_duration(audio.duration)

            combined = clips_array([[left_clip, anim_sequence]])
            final_clips.append(combined)

        final_video = concatenate_videoclips(final_clips, method="compose")
        final_video.write_videofile("final_instruction_video.mp4", fps=24, codec="libx264", audio_codec="aac")

        return send_file("final_instruction_video.mp4", mimetype="video/mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 500