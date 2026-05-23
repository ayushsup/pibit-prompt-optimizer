"""
ExtractBench dataset loader with multi-strategy PDF extraction.

Extraction strategy (in priority order):
  1. pymupdf (fitz)  — best for digital/hybrid PDFs, fast, handles most layouts
  2. pdfplumber      — good fallback for columnar/table-heavy documents
  3. PyPDF2          — last-resort pure-Python fallback
  4. Vision LLM      — for fully scanned/image-based PDFs where all text methods fail
                       Uses a vision-capable model via OpenRouter to OCR each page.

This means scanned resume PDFs (image-only) are handled automatically via the
vision fallback, rather than silently returning empty strings and being skipped.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Per-strategy extractors
# ---------------------------------------------------------------------------

def _try_pymupdf(pdf_path: str) -> str:
    import fitz  # pymupdf
    doc = fitz.open(pdf_path)
    pages = [doc[i].get_text("text") for i in range(len(doc))]
    doc.close()
    return "\n".join(p for p in pages if p).strip()


def _try_pdfplumber(pdf_path: str) -> str:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def _try_pypdf2(pdf_path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _try_vision_llm(pdf_path: str, api_key: str, model: str) -> str:
    """
    OCR fallback for image-based / scanned PDFs.

    Renders each page to a PNG at 150 DPI using pymupdf (no external
    dependencies required), then sends to a vision-capable LLM via OpenRouter.
    """
    import fitz

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    doc = fitz.open(pdf_path)
    page_texts: List[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # 150 DPI gives good OCR quality without huge payloads
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
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
                                    "This is a page from a resume. "
                                    "Extract ALL text exactly as it appears. "
                                    "Preserve structure (sections, bullet points). "
                                    "Return only the raw text — no commentary."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            page_texts.append(response.choices[0].message.content or "")
            time.sleep(3)  # Respect rate limits between page calls
        except Exception as exc:
            print(f"    ⚠️  Vision LLM failed on page {page_num + 1}: {exc}")

    doc.close()
    return "\n".join(page_texts).strip()


# ---------------------------------------------------------------------------
# Unified extraction entry point
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    pdf_path: str,
    api_key: Optional[str] = None,
    vision_model: str = "google/gemini-2.0-flash-exp:free",
) -> str:
    """
    Extract all text from a PDF using a progressive fallback chain.

    Falls back to vision LLM when all text-based methods return empty content
    (typically image-only / scanned PDFs).
    """
    filename = os.path.basename(pdf_path)

    for strategy_name, strategy_fn in [
        ("pymupdf",    lambda: _try_pymupdf(pdf_path)),
        ("pdfplumber", lambda: _try_pdfplumber(pdf_path)),
        ("PyPDF2",     lambda: _try_pypdf2(pdf_path)),
    ]:
        try:
            text = strategy_fn()
            if text and len(text.strip()) > 50:  # non-trivial content threshold
                return text
        except Exception as exc:
            print(f"    [{strategy_name}] failed for {filename}: {exc}")

    # All text strategies returned empty — try vision OCR
    if api_key:
        print(f"    🔍 Text extraction empty for {filename}. Trying vision OCR…")
        try:
            text = _try_vision_llm(pdf_path, api_key, vision_model)
            if text and len(text.strip()) > 50:
                print(f"    ✅ Vision OCR succeeded for {filename}.")
                return text
        except Exception as exc:
            print(f"    ⚠️  Vision OCR failed for {filename}: {exc}")
    else:
        print(
            f"    ⚠️  {filename}: all text methods returned empty. "
            "Set OPENROUTER_API_KEY and 'vision_model' in config to enable OCR fallback."
        )

    return ""


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class ExtractBenchLoader:
    """
    Loads (PDF text, gold JSON, schema) triples from an ExtractBench bundle.

    Expected directory layout:
      <base_path>/<schema_name>/
        ├── <name>-schema.json        ← JSON Schema with per-field evaluation_config
        └── pdf+gold/
            ├── document.pdf
            └── document.gold.json
    """

    def __init__(
        self,
        base_path: str,
        schema_name: str,
        vision_model: str = "google/gemini-2.0-flash-exp:free",
    ):
        # os.path.join handles Windows (\) and Unix (/) separators transparently
        self.schema_dir = os.path.join(base_path, *schema_name.split("/"))
        self.pdf_gold_dir = os.path.join(self.schema_dir, "pdf+gold")
        self.vision_model = vision_model
        self._api_key = os.environ.get("OPENROUTER_API_KEY")

    def _load_schema(self) -> str:
        if not os.path.isdir(self.schema_dir):
            raise FileNotFoundError(
                f"Schema directory not found: {self.schema_dir}\n"
                f"Check base_path and schema_name in your config."
            )
        candidates = [f for f in os.listdir(self.schema_dir) if f.endswith("-schema.json")]
        if not candidates:
            raise FileNotFoundError(
                f"No *-schema.json file found in: {self.schema_dir}"
            )
        schema_path = os.path.join(self.schema_dir, candidates[0])
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_all_document_pairs(self) -> List[Dict]:
        """
        Returns a list of dicts:
          id        : document base name
          text      : extracted PDF text (may be from vision OCR)
          gold_json : raw gold annotation JSON string
          schema    : raw JSON Schema string
        """
        if not os.path.isdir(self.pdf_gold_dir):
            raise FileNotFoundError(
                f"pdf+gold directory not found: {self.pdf_gold_dir}\n"
                "Run: git clone https://github.com/ContextualAI/extract-bench.git data/extract-bench"
            )

        schema = self._load_schema()
        docs: List[Dict] = []

        pdf_files = sorted(f for f in os.listdir(self.pdf_gold_dir) if f.endswith(".pdf"))

        for pdf_file in pdf_files:
            base_name = pdf_file[:-4]
            gold_path = os.path.join(self.pdf_gold_dir, f"{base_name}.gold.json")
            pdf_path  = os.path.join(self.pdf_gold_dir, pdf_file)

            if not os.path.exists(gold_path):
                print(f"  ⚠️  No gold file for {pdf_file} — skipping.")
                continue

            with open(gold_path, "r", encoding="utf-8") as f:
                gold_json = f.read()

            print(f"  Loading {pdf_file}…")
            text = extract_text_from_pdf(
                pdf_path,
                api_key=self._api_key,
                vision_model=self.vision_model,
            )

            if not text:
                print(f"  ❌ {pdf_file}: could not extract any text. Skipping.")
                continue

            docs.append({
                "id": base_name,
                "text": text,
                "gold_json": gold_json,
                "schema": schema,
            })

        print(f"\n  ✅ Loaded {len(docs)}/{len(pdf_files)} document pairs from: {self.schema_dir}")
        return docs