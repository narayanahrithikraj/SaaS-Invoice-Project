import os
import re
import tempfile
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import google.generativeai as genai
import logging

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app) # Allow requests from your frontend

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- 1. TESSERACT CONFIGURATION ---
# On Render, Tesseract is installed in the system PATH by 'render.yaml',
# so we do not need to set the pytesseract.tesseract_cmd path.
# We leave this blank for production.
logging.info("Tesseract should be in system PATH on production.")


# --- 2. GEMINI (LLM) CONFIGURATION ---
llm_model = None
try:
    # IMPORTANT: Get API key from environment variable on Render
    # This is set in the Render dashboard
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    generation_config = {
      "temperature": 0.1,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 4096,
      "response_mime_type": "application/json",
    }
    
    llm_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        generation_config=generation_config # <-- This was the typo, now fixed
    )
    logging.info("Gemini AI configured successfully.")
except Exception as e:
    logging.error(f"Error configuring Gemini AI: {e}")

# --- 3. HELPER FUNCTIONS ---

def parse_date(date_str, default_date):
    """
    Tries to parse a date string from various common formats.
    """
    if not date_str or date_str.lower() == 'n/a':
        return default_date

    formats_to_try = [
        '%Y-%m-%d', # 2025-11-02
        '%m/%d/%Y', # 11/02/2025
        '%d/%m/%Y', # 02/11/2025
        '%m-%d-%Y', # 11-02-2025
        '%d-%m-%Y', # 02-11-2025
        '%d.%m.%Y', # 02.11.2025
        '%m.%d.%Y', # 11.02.2025
        '%b %d, %Y', # Nov 02, 2025
        '%d %b %Y', # 02 Nov 2025
    ]
    
    for date_format in formats_to_try:
        try:
            return datetime.strptime(date_str, date_format).strftime('%Y-%m-%d')
        except ValueError:
            continue
            
    logging.warning(f"Could not parse date: {date_str}. Defaulting.")
    return default_date

def process_file(file_path, file_extension):
    """
    Converts a PDF or image file to text using Tesseract OCR.
    """
    all_text = ""
    try:
        if file_extension == '.pdf':
            with tempfile.TemporaryDirectory() as temp_path:
                try:
                    # On Render, Poppler is installed in the PATH by 'render.yaml',
                    # so we set poppler_path=None.
                    images = convert_from_path(
                        file_path, 
                        output_folder=temp_path, 
                        poppler_path=None
                    )
                    for i, img in enumerate(images):
                        logging.info(f"Processing page {i+1}...")
                        all_text += pytesseract.image_to_string(img) + "\n\n"
                except Exception as e:
                    logging.error(f"Error during PDF processing (Poppler): {e}")
                    raise Exception(f"PDF processing failed. Is poppler installed? Error: {e}")

        elif file_extension in ['.png', '.jpg', '.jpeg']:
            all_text = pytesseract.image_to_string(Image.open(file_path))
        else:
            raise ValueError("Unsupported file type")
            
        if not all_text:
            logging.warning("OCR returned no text.")
            raise Exception("OCR failed to extract any text from the document.")

        return all_text
        
    except Exception as e:
        logging.error(f"Error in process_file: {e}")
        raise

def extract_data_from_text(text):
    """
    Uses the Gemini LLM to extract all structured data from OCR text.
    """
    if not llm_model:
        raise Exception("Gemini AI Model is not configured or failed to initialize.")

    today_date = datetime.now().strftime('%Y-%m-%d')

    # This is the full, advanced prompt
    prompt = f"""
    You are an expert financial analyst. Analyze the following OCR text from an invoice and extract the key fields.
    Return your answer in a strict JSON format. Do not include any text outside of the JSON block.
    
    The JSON schema must be:
    {{
      "vendorName": "Vendor's company name",
      "invoiceNumber": "The invoice ID or number",
      "invoiceDate": "YYYY-MM-DD",
      "dueDate": "YYYY-MM-DD",
      "subtotal": 0.00,
      "tax": 0.00,
      "totalAmount": 0.00,
      "currency": "e.g., 'INR', 'USD'",
      "lineItems": [
        {{ "description": "Item description", "quantity": 1, "unitPrice": 0.00, "total": 0.00 }}
      ],
      "confidenceScore": 0.0,
      "rationale": "A one-sentence explanation for your extraction."
    }}

    Rules:
    - If a field is not found, return "N/A" for strings, 0.00 for numbers, and [] for lineItems.
    - If invoiceDate is not found, use today's date: {today_date}.
    - If dueDate is not found, use the invoiceDate.
    - "totalAmount" must be the final total. "subtotal" is the total before tax.
    - "lineItems" is an array of objects. Try to find at least one. If none are clear, return an empty array.
    - "confidenceScore" is your estimated confidence from 0.0 (low) to 1.0 (high) that you correctly extracted the main fields.
    - "rationale" is a *short* (one sentence) justification.

    Here is the OCR text (first 4000 chars):
    ---
    {text[:4000]}
    ---
    """
    
    try:
        logging.info("Sending text to Gemini LLM for extraction...")
        response = llm_model.generate_content(prompt)
        
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        data = json.loads(json_text)
        
        invoice_date = parse_date(data.get("invoiceDate"), default_date=today_date)
        due_date = parse_date(data.get("dueDate"), default_date=invoice_date)
        
        extracted_data = {
            "vendorName": data.get("vendorName", "Unknown Vendor").strip(),
            "invoiceNumber": data.get("invoiceNumber", "N/A").strip(),
            "invoiceDate": invoice_date,
            "dueDate": due_date,
            "subtotal": float(data.get("subtotal", 0.0) or 0.0),
            "tax": float(data.get("tax", 0.0) or 0.0),
            "totalAmount": float(data.get("totalAmount", 0.0) or 0.0),
            "currency": data.get("currency", "INR"),
            "lineItems": data.get("lineItems", []),
            "confidenceScore": float(data.get("confidenceScore", 0.5) or 0.5),
            "rationale": data.get("rationale", "N/A"),
            "status": "Pending" # Default status for human review
        }
        
        logging.info(f"LLM Extraction successful: {extracted_data['vendorName']}, {extracted_data['totalAmount']}")
        return extracted_data

    except Exception as e:
        logging.error(f"Error during LLM extraction: {e}")
        error_message = f"AI Error: {e}"
        return {
            "vendorName": "AI Error",
            "invoiceNumber": "N/A",
            "invoiceDate": today_date,
            "dueDate": today_date,
            "subtotal": 0.0,
            "tax": 0.0,
            "totalAmount": 0.0,
            "currency": "INR",
            "lineItems": [],
            "confidenceScore": 0.0,
            "rationale": str(e),
            "status": f"AI Error: {e}"
        }

# --- 4. FLASK API ENDPOINT ---
@app.route('/process-invoice', methods=['POST'])
def process_invoice():
    logging.info("Received request at /process-invoice...")
    if 'invoiceFile' not in request.files:
        return jsonify({"status": "Error", "message": "No file part in the request"}), 400

    file = request.files['invoiceFile']
    if file.filename == '':
        return jsonify({"status": "Error", "message": "No selected file"}), 400

    if file:
        filename, file_extension = os.path.splitext(file.filename)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, f"upload{file_extension}")
            file.save(temp_file_path)
            logging.info(f"File saved temporarily at {temp_file_path}")

            try:
                ocr_text = process_file(temp_file_path, file_extension)
                extracted_data = extract_data_from_text(ocr_text)
                return jsonify({ "status": "Success", "extractedData": extracted_data }), 200
            except Exception as e:
                logging.error(f"Error in process_invoice endpoint: {e}")
                return jsonify({"status": "Error", "message": str(e)}), 500

    return jsonify({"status": "Error", "message": "Unknown error"}), 500

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    # Use 0.0.0.0 to be accessible on the network
    # Use Render's dynamically assigned PORT
    port = int(os.environ.get("PORT", 5000))
    # Run with debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)

