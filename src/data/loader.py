import os
import json
import time
from PyPDF2 import PdfReader
import google.generativeai as genai

class ExtractBenchLoader:
    def __init__(self, base_path: str, schema_name: str):
        self.schema_dir = os.path.join(base_path, schema_name)
        
        # Initialize Google's Native API for Vision OCR
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            # Pointing directly to the newest Flash model on Google's servers
            self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.vision_model = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            # 1. Try standard text extraction first
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            # If standard extraction worked, return the text
            if text.strip():
                return text

            # 2. If it's empty (scanned/image PDF), fallback to Native Gemini Vision
            print(f"    🔍 Text extraction empty for {os.path.basename(pdf_path)}. Trying Google AI Studio OCR...")
            
            if not self.vision_model:
                print("    ⚠️ GEMINI_API_KEY not found in terminal. Cannot run OCR fallback.")
                return ""

            # Upload the PDF directly to Google's File API for processing
            pdf_file = genai.upload_file(path=pdf_path)
            
            # Brief pause to ensure the file is processed on their end
            time.sleep(2)
            
            response = self.vision_model.generate_content([
                "Extract all the readable text from this document exactly as it appears. Do not summarize.", 
                pdf_file
            ])
            
            # Clean up the file from Google's servers to prevent cluttering your account
            genai.delete_file(pdf_file.name)
            
            return response.text

        except Exception as e:
            print(f"    ❌ Error reading {pdf_path}: {e}")
            return ""

    def load_all_document_pairs(self) -> list:
        pdf_gold_dir = os.path.join(self.schema_dir, "pdf+gold")
        
        schema_file = [f for f in os.listdir(self.schema_dir) if f.endswith("-schema.json")][0]
        schema_path = os.path.join(self.schema_dir, schema_file)
        
        with open(schema_path, 'r') as f:
            schema = f.read()

        docs = []
        if not os.path.exists(pdf_gold_dir):
            print(f"Directory not found: {pdf_gold_dir}")
            return docs

        for pdf_file in os.listdir(pdf_gold_dir):
            if pdf_file.endswith(".pdf"):
                base_name = pdf_file.replace(".pdf", "")
                json_file = f"{base_name}.gold.json"
                
                pdf_path = os.path.join(pdf_gold_dir, pdf_file)
                json_path = os.path.join(pdf_gold_dir, json_file)
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        gold_json = f.read() 
                        
                    print(f"Loading {pdf_file}...")
                    text = self.extract_text_from_pdf(pdf_path)
                    
                    # Only append if we actually got text back
                    if text.strip():
                        docs.append({
                            "id": base_name,
                            "text": text,
                            "gold_json": gold_json,
                            "schema": schema
                        })
        return docs