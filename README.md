# Aarogya Mitra 


**Aarogya Mitra** turns a photo of a handwritten prescription into instructions patients can actually understand  in their own language, spoken aloud, or shown as a short instructional video. It's built for low-literacy and non-English-speaking patients in India, with a companion tool for pharmacists to verify, edit, and find substitute medicines.

---
## Team Project
 This repository is maintained on my GitHub as a portfolio copy of a collaborative project developed by a two-member team.

## How It Works

1. A prescription photo is captured or uploaded.
2. **Google Cloud Vision OCR** extracts the raw text.
3. A **Groq-hosted LLaMA 3.3 70B** model parses the text into structured medicine entries (name, route, dosage & duration), and prescription shorthand (`BD`, `HS`, `1-0-1`, etc.) is expanded into plain English.
4. Extracted medicine names are fuzzy-matched against a medicine database to correct OCR errors.
5. Instructions are translated into the patient's language using **Meta's NLLB-200** translation model, and can be turned into speech (**gTTS**) or an animated instructional video (**MoviePy**).
6. Pharmacists get a review screen to edit extracted data and look up cheaper/available substitute medicines.

---

## Architecture
<img width="1238" height="698" alt="image" src="https://github.com/user-attachments/assets/f0f55417-8c69-4913-b42f-6cf9321067f1" />


## Features

- **Prescription OCR** — reads handwritten or printed prescriptions via Google Vision's document text detection.
- **AI medicine extraction** — LLM-based parsing of medicine name, route, dosage, and duration from messy OCR text.
- **Medical abbreviation expansion** — converts shorthand (`OD`, `TDS`, `PRN`, `1-1-1`, etc.) into full instructions.
- **Multilingual translation** — Hindi, Marathi, Gujarati, Bengali, Urdu, Tamil, Kannada, Telugu, and Malayalam, including native-script digit conversion.
- **Instructional video generation** — per-medicine narrated video clips (pill, drops, cream, inhaler) assembled with the translated audio, for patients who can't read.
- **Medicine substitute finder** — fuzzy-matches a medicine against a substitutes dataset and cross-checks availability.
- **Two guided flows** — a simple Patient flow (capture → video) and a detailed Pharmacist flow (capture → edit → translate/download).

---

## Tech Stack

**Backend:** Python, Flask, Flask-CORS
**Frontend:** React 19, React Router

**AI / ML:**
- Google Cloud Vision API — OCR
- Groq API (LLaMA 3.3 70B) — medicine extraction & instruction simplification
- Hugging Face Transformers — `facebook/nllb-200-1.3B` (translation), `dslim/bert-base-NER` (named entity recognition)
- RapidFuzz / textdistance — fuzzy string matching
- gTTS — text-to-speech
- MoviePy — video assembly
- pytesseract + langdetect — secondary OCR and language detection for the video pipeline

---

## Project Structure

```text
aarogya_mitra/
├── backend/
│   ├── app.py                       # Flask app entry point, registers all blueprints
│   ├── routes/
│   │   ├── generate.py              # POST /generate           — OCR + LLM medicine extraction
│   │   ├── generate_text.py         # POST /generate-text       — simplify + translate instructions
│   │   ├── generate_video.py        # POST /generate-video      — full video instruction pipeline
│   │   └── substitute.py            # POST /find_substitute     — medicine substitute lookup
│   ├── utils/
│   │   ├── ocr.py                   # Google Vision OCR wrapper
│   │   ├── llm.py                   # Groq client + JSON response parsing
│   │   └── text_utils.py            # Abbreviation/frequency maps, Indic digit conversion
│   ├── Medicine_Details.csv
│   ├── Medicine_Details_revised.csv
│   └── Medicine_Details_revised_new.csv
│
└── my-app/                          # React frontend
    └── src/
        ├── App.jsx                  # Route definitions
        ├── components/Translator.jsx # Lightweight i18n context
        ├── i18n/                    # en.json, hi.json UI strings
        └── pages/
            ├── Language.jsx         # Language selection (entry screen)
            ├── Website.jsx          # Patient / Pharmacist role picker
            ├── PatientCamera.jsx    # Patient flow: capture/upload → generated video
            └── Pharmacist.jsx       # Pharmacist flow: capture/upload → edit → substitutes → translate
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide, with language packs for the Indian languages used (`hin`, `tam`, `tel`, `kan`, `mal`, `guj`, `ben`, `mar`, `pan`, `urd`)
- A [Google Cloud Vision API](https://cloud.google.com/vision) key
- A [Groq API](https://groq.com/) key

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install flask flask-cors python-dotenv pandas rapidfuzz requests \
            transformers torch gtts langdetect pytesseract moviepy \
            textdistance groq pillow
```

Create a `.env` file inside `backend/`:

```env
GROQ_API_KEY=your_groq_api_key
VISION_API_KEY=your_google_vision_api_key
```

Run the server:

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

> **Note:** The `/generate-video` endpoint expects a `PESURF_new_medicine_dataset_revised.csv` substitutes dataset and a set of short instructional video clips (`pill.mp4`, `eye_drop.mp4`, `ear_drop.mp4`, `generic_drop.mp4`, `cream_application.mp4`, `inhaler.mp4`, `default.mp4`) in the `backend/` directory. These are excluded from version control and need to be added locally before that route will work.

### Frontend Setup

```bash
cd my-app
npm install
npm start
```

The app will run at `http://localhost:3000` and expects the backend to be running on `http://127.0.0.1:5000`.

---

## API Endpoints

| Method | Endpoint            | Description                                                        |
|--------|----------------------|----------------------------------------------------------------------|
| POST   | `/generate`          | Upload a prescription image → returns extracted medicines as JSON  |
| POST   | `/generate-text`     | Convert structured medicine input → simplified + translated text   |
| POST   | `/generate-video`    | Upload a prescription image → returns a narrated instructional video |
| POST   | `/find_substitute`   | Look up substitute medicines for a given medicine name             |
---

## My Contributions

- Built the prescription data pipeline by collecting handwritten prescriptions from public datasets and local pharmacies.
- Curated and annotated prescription data for OCR and information extraction tasks.
- Evaluated multiple OCR engines through comparative experiments to identify the best-performing solution for handwritten prescriptions.
- Developed preprocessing and post-processing pipelines to clean noisy OCR outputs.
- Researched and experimented with multiple NER techniques and LLM-based approaches for extracting structured prescription information.
- Implemented fuzzy string matching to accurately map extracted medicine names to standardized medicine datasets.
- Designed a pipeline to transform raw OCR output into structured JSON, enabling downstream translation, text-to-speech, and pharmacist verification modules.
  


