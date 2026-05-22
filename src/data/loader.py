import os
import json
from PyPDF2 import PdfReader

class ExtractBenchLoader:
    def __init__(self, base_path: str, schema_name: str):
        self.schema_dir = os.path.join(base_path, schema_name)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

    def load_all_document_pairs(self) -> list:
        pdf_gold_dir = os.path.join(self.schema_dir, "pdf+gold")
        
        # Automatically find the schema file (e.g., resume-schema.json)
        schema_file = [f for f in os.listdir(self.schema_dir) if f.endswith("-schema.json")][0]
        schema_path = os.path.join(self.schema_dir, schema_file)
        
        with open(schema_path, 'r') as f:
            schema = f.read()

        docs = []
        if not os.path.exists(pdf_gold_dir):
            print(f"Directory not found: {pdf_gold_dir}")
            return docs

        # Iterate through all PDFs in the directory
        for pdf_file in os.listdir(pdf_gold_dir):
            if pdf_file.endswith(".pdf"):
                base_name = pdf_file.replace(".pdf", "")
                json_file = f"{base_name}.gold.json"
                
                pdf_path = os.path.join(pdf_gold_dir, pdf_file)
                json_path = os.path.join(pdf_gold_dir, json_file)
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        gold_json = f.read() # Read as string to pass to Critic
                        
                    print(f"Loading {pdf_file}...")
                    text = self.extract_text_from_pdf(pdf_path)
                    docs.append({
                        "id": base_name,
                        "text": text,
                        "gold_json": gold_json,
                        "schema": schema
                    })
        return docs