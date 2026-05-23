"""
Dataset loader for ExtractBench PDF/JSON pairs.

PDF text extraction pipeline (tried in order):
  1. pdfplumber  — best for complex layouts and multi-column text
  2. pymupdf     — fast, handles many edge cases pdfplumber misses
  3. PyPDF2      — legacy fallback
  4. OpenRouter vision model (via config vision_model) — for scanned/image PDFs
  5. Google Gemini native API (GEMINI_API_KEY) — final OCR fallback

Only documents where text could be extracted are returned, so the optimizer
always receives usable input regardless of PDF type.
"""

import base64
import os
import time
from typing import List, Optional


class ExtractBenchLoader:
    def __init__(
        self,
        base_path: str,
        schema_name: str,
        vision_model: Optional[str] = None,
        openrouter_key: Optional[str] = None,
    ):
        self.schema_dir = os.path.join(base_path, schema_name)
        self.vision_model = vision_model
        self.openrouter_key = openrouter_key

        # Optional Gemini client for OCR (requires GEMINI_API_KEY)
        self._gemini_client = None
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=gemini_key)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # PDF text extraction
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF using a multi-stage pipeline.
        Returns empty string only if every strategy fails.
        """
        filename = os.path.basename(pdf_path)

        # 1. pdfplumber — best for complex layouts
        text = self._try_pdfplumber(pdf_path)
        if text:
            return text

        # 2. pymupdf (fitz) — fast and handles many edge cases
        text = self._try_pymupdf(pdf_path)
        if text:
            return text

        # 3. PyPDF2 — legacy fallback
        text = self._try_pypdf2(pdf_path)
        if text:
            return text

        # 4. All text-layer extractors failed → OCR required
        print(f"    🔍 Text extraction empty for {filename}. Trying vision OCR...")

        # 4a. Google Gemini 2.5 Flash (primary OCR — highest quality)
        if self._gemini_client:
            text = self._try_gemini_ocr(pdf_path)
            if text:
                return text

        # 4b. OpenRouter vision model (fallback if Gemini not configured)
        if self.vision_model and self.openrouter_key:
            text = self._try_openrouter_vision(pdf_path)
            if text:
                return text

        print(f"    ⚠️  All extraction methods failed for {filename}. Skipping.")
        return ""

    def _try_pdfplumber(self, pdf_path: str) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n".join(pages).strip()
        except Exception:
            return ""

    def _try_pymupdf(self, pdf_path: str) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(pages).strip()
        except Exception:
            return ""

    def _try_pypdf2(self, pdf_path: str) -> str:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            pages = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            return "\n".join(pages).strip()
        except Exception:
            return ""

    def _try_openrouter_vision(self, pdf_path: str) -> str:
        """Render PDF pages as images and send to OpenRouter vision model."""
        try:
            import fitz
            from openai import OpenAI

            client = OpenAI(
                api_key=self.openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            )

            doc = fitz.open(pdf_path)
            page_texts: List[str] = []

            for page_num, page in enumerate(doc):
                # Render at 2× zoom for legibility
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                response = client.chat.completions.create(
                    model=self.vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL text from this document page exactly as it appears. "
                                    "Return only the raw text with no formatting, labels, or commentary."
                                ),
                            },
                        ],
                    }],
                    max_tokens=2000,
                )
                page_texts.append(response.choices[0].message.content or "")
                time.sleep(3)  # Polite inter-page pause

            doc.close()
            return "\n\n".join(page_texts).strip()

        except Exception as exc:
            print(f"    ⚠️  OpenRouter vision OCR failed: {exc}")
            return ""

    def _try_gemini_ocr(self, pdf_path: str) -> str:
        """Upload PDF to Google File API and extract text with Gemini."""
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            pdf_file = self._gemini_client.files.upload(
                file=pdf_bytes,
                config={"mime_type": "application/pdf"},
            )
            time.sleep(2)

            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Extract all readable text from this document exactly as it appears. "
                    "Return only the raw text, no commentary.",
                    pdf_file,
                ],
            )

            try:
                self._gemini_client.files.delete(name=pdf_file.name)
            except Exception:
                pass

            return (response.text or "").strip()

        except Exception as exc:
            print(f"    ⚠️  Gemini OCR failed: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_all_document_pairs(self) -> List[dict]:
        """
        Load all (PDF text, gold JSON, schema) triples from the dataset directory.

        Returns only documents where text extraction succeeded.
        """
        pdf_gold_dir = os.path.join(self.schema_dir, "pdf+gold")

        # Locate the schema file
        try:
            schema_files = [
                f for f in os.listdir(self.schema_dir) if f.endswith("-schema.json")
            ]
            if not schema_files:
                print(f"No schema file found in {self.schema_dir}")
                return []
            schema_path = os.path.join(self.schema_dir, schema_files[0])
        except FileNotFoundError:
            print(f"Schema directory not found: {self.schema_dir}")
            return []

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = f.read()

        docs = []
        if not os.path.exists(pdf_gold_dir):
            print(f"pdf+gold directory not found: {pdf_gold_dir}")
            return docs

        for pdf_file in sorted(os.listdir(pdf_gold_dir)):
            if not pdf_file.endswith(".pdf"):
                continue

            base_name = pdf_file.replace(".pdf", "")
            json_file = f"{base_name}.gold.json"
            pdf_path = os.path.join(pdf_gold_dir, pdf_file)
            json_path = os.path.join(pdf_gold_dir, json_file)

            if not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                gold_json = f.read()

            print(f"Loading {pdf_file}...")
            text = self.extract_text_from_pdf(pdf_path)

            if text.strip():
                docs.append({
                    "id": base_name,
                    "text": text,
                    "gold_json": gold_json,
                    "schema": schema,
                })
            else:
                print(f"    ⚠️  Skipping {pdf_file} — could not extract text.")

        return docs
