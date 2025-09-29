import os
import re
from flask import Flask, request, render_template_string
from PIL import Image
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
POPPLER_PATH = r'C:\poppler-25.07.0\Library\bin'

# --- Basic Flask App Setup ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Verification Logic ---

def verify_physical_scan(doc_image, text):
    """
    Final version using Histogram Equalization for robust watermark detection.
    """
    report = {"Document Type Detected": "Physical Scan"}
    passed_checks = 0
    total_checks_to_pass = 3  # All 3 checks: Format, Logo, Watermark

    # --- Check 1: Format Checker ---
    required_keywords = ["statement of grades", "enrollment number", "sgpa"]
    found_keywords = sum(1 for kw in required_keywords if kw in text.lower())
    if found_keywords >= 2:
        report["Document Format"] = "✅ Correct"
        passed_checks += 1
    else:
        report["Document Format"] = f"❌ Incorrect (Found {found_keywords}/{len(required_keywords)} keywords)"

    # --- Visual Feature Verification ---
    templates = {
        "University Logo": "university_logo.png",
        "Background Watermark": "watermark_template.png"
    
    }

    for feature_name, template_path in templates.items():
        if not os.path.exists(template_path):
            report[feature_name] = "⚠️ Template missing on server"
            continue

        template = cv2.imread(template_path, 0)
        if template is None:
            report[feature_name] = "❌ Template could not be read"
            continue

        # --- Image Processing Strategy ---
        # For the high-contrast logo, use simple blurring.
        if "Logo" in feature_name:
            processed_doc = cv2.GaussianBlur(doc_image, (7, 7), 0)
            processed_template = cv2.GaussianBlur(template, (7, 7), 0)
            threshold = 0.60  # A reliable threshold for the logo

        # For the low-contrast watermark, use Histogram Equalization.
        else:
            # This enhances contrast, making the faint watermark stand out.
            processed_doc = cv2.equalizeHist(doc_image)
            processed_template = cv2.equalizeHist(template)
            threshold = 0.35  # A good starting threshold for this powerful method

        # --- Resizing and Matching ---
        doc_h, doc_w = processed_doc.shape[:2]
        scale_factor = 0.15 if "Logo" in feature_name else 0.50
        target_w = int(doc_w * scale_factor)

        # Ensure target_w is not zero
        if target_w == 0:
            report[feature_name] = "❌ Document image is too small"
            continue

        ratio = target_w / processed_template.shape[1]
        target_h = int(processed_template.shape[0] * ratio)
        
        # Ensure target_h is not zero
        if target_h == 0:
            report[feature_name] = "❌ Template resize resulted in zero height"
            continue
            
        resized_template = cv2.resize(processed_template, (target_w, target_h))

        if resized_template.shape[0] > doc_h:
            report[feature_name] = "❌ Template resize failed"
            continue

        res = cv2.matchTemplate(processed_doc, resized_template, cv2.TM_CCOEFF_NORMED)

        if np.max(res) > threshold:
            report[feature_name] = f"✅ Found (Confidence: {np.max(res):.2f})"
            passed_checks += 1
        else:
            report[feature_name] = f"❌ Not Found (Confidence: {np.max(res):.2f})"

    # --- Final Verdict ---
    if passed_checks >= total_checks_to_pass:
        report['Overall Result'] = "VERIFIED"
    else:
        report['Overall Result'] = "NOT VERIFIED"

    return report

def verify_digital_document(text):
    """Rules for verifying a digitally generated document."""
    report = {"Document Type Detected": "Digital (DigiLocker)"}
    # For digital documents, keyword checks are sufficient
    if "Indira Gandhi Delhi Technical University".lower() in text.lower() and "digilocker" in text.lower():
        report['Overall Result'] = "VERIFIED"
        report["Source"] = "✅ DigiLocker and University name found"
    else:
        report['Overall Result'] = "NOT VERIFIED"
        report["Source"] = "❌ DigiLocker or University name missing"
        
    return report

def run_verification(filepath):
    """Detects document type and runs the appropriate verification checks."""
    text = extract_text_from_file(filepath)
    
    if not text:
        return {"Overall Result": "ERROR: Could not extract any text from the document."}

    if "digilocker" in text.lower():
        return verify_digital_document(text)
    else:
        try:
            if filepath.lower().endswith('.pdf'):
                images = convert_from_path(filepath, poppler_path=POPPLER_PATH)
                if not images: return {"Overall Result": "ERROR: PDF is empty or unreadable."}
                doc_image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
            else:
                doc_image = cv2.imread(filepath, 0)
            
            if doc_image is None: return {"Overall Result": "ERROR: Could not read file as an image."}
            return verify_physical_scan(doc_image, text)
        except Exception as e:
            return {"Overall Result": f"ERROR: Image processing failed: {e}"}

def extract_text_from_file(filepath):
    """Extracts text from any supported file type."""
    text = ""
    file_ext = os.path.splitext(filepath)[1].lower()
    try:
        if file_ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                full_text = "".join(page.extract_text() or "" for page in pdf.pages)
            if len(full_text.strip()) < 100:
                images = convert_from_path(filepath, poppler_path=POPPLER_PATH)
                text = "".join(pytesseract.image_to_string(img) for img in images)
            else: 
                text = full_text
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            text = pytesseract.image_to_string(Image.open(filepath))
    except Exception: 
        text = "" 
    return text

# --- HTML Template with Detailed Report ---
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Transcript Verification</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f4f4f9; }
        .container { max-width: 800px; margin: auto; background: white; padding: 2.5em; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .result { margin-top: 2em; padding: 1.5em; border-radius: 8px; }
        .result.success { border: 1px solid #28a745; background-color: #e9f7ec; }
        .result.fail { border: 1px solid #dc3545; background-color: #fbebed; }
        h2, h3, h4 { color: #333; }
        ul { list-style-type: none; padding: 0; }
        li { padding: 0.5em 0; border-bottom: 1px solid #eee; }
        li:last-child { border-bottom: none; }
        strong { color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h2>IGDTUW Alumni Transcript Verification</h2>
        <p>Upload a transcript to verify its authenticity.</p>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file" required><br><br>
          <input type="submit" value="Verify">
        </form>
        
        {% if report %}
          <div class="result {{ 'success' if 'VERIFIED' in report.get('Overall Result', '') else 'fail' }}">
              <h3>Final Result: {{ report.get('Overall Result', 'INCONCLUSIVE') }}</h3>
              <hr>
              <h4>Verification Details:</h4>
              <ul>
                  {% for key, value in report.items() if key != 'Overall Result' %}
                      <li><strong>{{ key }}:</strong> {{ value }}</li>
                  {% endfor %}
              </ul>
          </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def verify_transcript():
    report = None
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            report = run_verification(filepath)
        else:
            report = {"Overall Result": "ERROR: No file selected"}
    return render_template_string(HTML_TEMPLATE, report=report)

if __name__ == "__main__":
    app.run(debug=True)