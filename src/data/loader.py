import os
import json
from PyPDF2 import PdfReader
from typing import Dict, Any

class ExtractBenchLoader:
    def __init__(self, base_path: str, schema_name: str):
        self.schema_dir = os.path.join(base_path, schema_name)
        self.pdf_gold_dir = os.path.join(self.schema_dir, "pdf+gold")
        self.schema_path = os.path.join(self.schema_dir, "resume-schema.json")
        
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
        pdf_path = os.path.join(self.pdf_gold_dir, f"{doc_id}.pdf")
        json_path = os.path.join(self.pdf_gold_dir, f"{doc_id}.gold.json")
        schema_path = self.schema_path
        with open(json_path, 'r', encoding='utf-8') as f:
            gold_json = json.load(f)
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = f.read()
        text = self.extract_text_from_pdf(pdf_path)
        return {"id": doc_id, "text": text, "gold_json": gold_json, "schema": schema}

    def list_doc_ids(self) -> list:
        # List all .pdf files in pdf+gold and return their base names (without .pdf)
        doc_ids = []
        for fname in os.listdir(self.pdf_gold_dir):
            if fname.endswith('.pdf'):
                doc_ids.append(fname[:-4])
        return doc_ids

    def load_all_document_pairs(self) -> list:
        # Loads all document pairs in the dataset
        doc_ids = self.list_doc_ids()
        return [self.load_document_pair(doc_id) for doc_id in doc_ids]