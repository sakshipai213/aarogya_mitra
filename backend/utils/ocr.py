import base64
import requests
import os

VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"

def get_ocr_output_single_image(image_path):
    with open(image_path, "rb") as img_file:
        content = base64.b64encode(img_file.read()).decode()
    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
        }]
    }
    response = requests.post(VISION_URL, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    text = ""
    if "responses" in result and result["responses"]:
        first = result["responses"][0]
        if "error" in first:
            raise RuntimeError(f"Vision API error: {first['error'].get('message', first['error'])}")
        if "fullTextAnnotation" in first:
            text = first["fullTextAnnotation"].get("text", "")
        elif "textAnnotations" in first and first["textAnnotations"]:
            text = first["textAnnotations"][0].get("description", "")
    return text.strip()
