from flask import Blueprint, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.llm import groq_client
from utils.text_utils import convert_digits

generate_text_bp = Blueprint('generate_text', __name__)

model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

LANG_CODE_MAP = {
    'hindi': 'hin_Deva', 'marathi': 'mar_Deva', 'gujarati': 'guj_Gujr',
    'bengali': 'ben_Beng', 'urdu': 'urd_Arab', 'tamil': 'tam_Taml',
    'kannada': 'kan_Knda', 'telugu': 'tel_Telu', 'malayalam': 'mal_Mlym',
}

@generate_text_bp.route('/generate-text', methods=['POST'])
def generate():
    data = request.json
    structured_input = data['structured_input']
    language = data['language'].lower()

    prompt = (
        "You are a medical assistant. Convert the structured medicine instructions into very simple, spoken-style English. "
        "Present each instruction as a numbered list starting from 1, with no introductions, no summaries, and no extra comments. "
        "Only output the instructions themselves.\n"
        "Example:\n"
        "Input: paracetamol 500mg;orally;morning and night x3 weeks\n"
        "Output: Take 1 Paracetamol 500 mg tablet by mouth in the morning and again at night and keep doing this for 3 weeks.\n"
        "Input: Phexin 500mg Capsule;oral;1-0-1 X for 5 days\n"
        "Output: Take 1 Phexin 500 mg tablet by mouth in the morning and at night and keep doing this for 5 days.\n"
        f"Input: {structured_input}\nOutput:"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.6,
        top_p=0.9
    )
    english_instruction = response.choices[0].message.content.strip()

    tgt_lang = LANG_CODE_MAP.get(language, 'hin_Deva')
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(english_instruction.lower(), return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated_tokens = model.generate(
        **inputs, forced_bos_token_id=forced_bos_token_id,
        max_length=128, num_beams=4, early_stopping=True
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    translated_text = convert_digits(translated_text, language=language)

    return jsonify({"english_instruction": english_instruction, "translated_instruction": translated_text})
