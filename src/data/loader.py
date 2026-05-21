import os
import json
from PyPDF2 import PdfReader
from typing import Dict, Any

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

    def load_document_pair(self, doc_id: str) -> Dict[str, Any]:
        pdf_path = os.path.join(self.schema_dir, "pdfs", f"{doc_id}.pdf")
        json_path = os.path.join(self.schema_dir, "annotations", f"{doc_id}.json")
        schema_path = os.path.join(self.schema_dir, "schema.json")
        
        with open(json_path, 'r') as f:
            gold_json = json.load(f)
            
        with open(schema_path, 'r') as f:
            schema = f.read()

        text = self.extract_text_from_pdf(pdf_path)
        return {"id": doc_id, "text": text, "gold_json": gold_json, "schema": schema}