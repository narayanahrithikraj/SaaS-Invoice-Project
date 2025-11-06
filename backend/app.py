import os
import re
import tempfile
import json
import mimetypes  # <-- Added to detect file type
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
# from pdf2image import convert_from_path  # <-- REMOVED
# from PIL import Image                   # <-- REMOVED
# import pytesseract                      # <-- REMOVED
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()  # Make sure this is at the top
app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- 1. TESSERACT CONFIGURATION ---
# (This section is no longer needed)
logging.info("Tesseract and Poppler are NO longer used in this project.")

# --- 2. GEMINI (LLM) CONFIGURATION ---
llm_model = None
try:
   
    genai.configure(api_key="AIzaSyCPHChd7zPV6QwWXOv1Z90WsHzVrXaCXl8")
    generation_config = {
      "temperature": 0.1,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 4096,
      "response_mime_type": "application/json",
    }
    
    # --- IMPORTANT ---
    # Changed to a model that supports file uploads (multimodal)
    # Your old model name was invalid.
    llm_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config
    )
    logging.info("Gemini AI configured successfully (multimodal).")
except Exception as e:
    logging.error(f"Error configuring Gemini AI: {e}")

# --- 3. HELPER FUNCTIONS ---

def parse_date(date_str, default_date):
    """
    Tries to parse a date string from various common formats.
    (This function remains unchanged)
    """
    if not date_str or date_str.lower() == 'n/a':
        return default_date

    formats_to_try = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
        '%d.%m.%Y', '%m.%d.%Y', '%b %d, %Y', '%d %b %Y',
    ]
    
    for date_format in formats_to_try:
        try:
            return datetime.strptime(date_str, date_format).strftime('%Y-%m-%d')
        except ValueError:
            continue
            
    logging.warning(f"Could not parse date: {date_str}. Defaulting.")
    return default_date

# --- REMOVED 'process_file' FUNCTION ---
# Tesseract/Poppler logic is no longer needed.

def extract_data_from_file(file_path):  # <-- This function is NEW
    """
    Uploads the invoice file (PDF/image) to Gemini and extracts structured data.
    """
    if not llm_model:
        raise Exception("Gemini AI Model is not configured or failed to initialize.")

    today_date = datetime.now().strftime('%Y-%m-%d')
    uploaded_file = None
    
    try:
        # 1. Upload the file to Gemini
        logging.info(f"Uploading file to Gemini: {file_path}")
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream' # Fallback
            
        uploaded_file = genai.upload_file(path=file_path, mime_type=mime_type)
        logging.info(f"File uploaded successfully: {uploaded_file.name}")

        # 2. Create the multimodal prompt
        prompt = [
            f"""
            You are an expert financial analyst. Analyze the following invoice file (which could be a PDF or image) 
            and extract the key fields. Perform OCR on the file as needed.
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
            """,
            uploaded_file  # <-- Pass the file object directly to Gemini
        ]

        # 3. Send to Gemini
        logging.info("Sending file to Gemini LLM for extraction...")
        response = llm_model.generate_content(prompt)
        
        # 4. Parse the response (same as before)
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
        }

        print("extracted_data",extracted_data)

        logging.info(f"LLM Extraction successful: {extracted_data['vendorName']}, {extracted_data['totalAmount']}")
        return extracted_data

    except Exception as e:
        logging.error(f"Error during LLM extraction: {e}")
        error_message = f"AI Error: {e}"
        # (Error handling logic remains the same)
        return {
            "vendorName": "AI Error", "invoiceNumber": "N/A", "invoiceDate": today_date,
            "dueDate": today_date, "subtotal": 0.0, "tax": 0.0, "totalAmount": 0.0,
            "currency": "INR", "lineItems": [], "confidenceScore": 0.0, "rationale": str(e),
            "status": f"AI Error: {e}"
        }
    finally:
        # 5. Clean up the uploaded file from Google's servers
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                logging.info(f"Cleaned up uploaded file: {uploaded_file.name}")
            except Exception as e:
                logging.warning(f"Could not delete uploaded file: {e}")


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
        # We still need to save the file temporarily to get a path
        filename = file.filename
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, filename)
            file.save(temp_file_path)
            logging.info(f"File saved temporarily at {temp_file_path}")

            try:
                # --- THIS IS THE ONLY CHANGE ---
                # Call the new file-based extraction function
                extracted_data = extract_data_from_file(temp_file_path)
                # --- END OF CHANGE ---
                
                return jsonify({ "status": "Success", "extractedData": extracted_data }), 200
            except Exception as e:
                logging.error(f"Error in process_invoice endpoint: {e}")
                return jsonify({"status": "Error", "message": str(e)}), 500

    return jsonify({"status": "Error", "message": "Unknown error"}), 500

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # Use debug=True for local dev
