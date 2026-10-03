import sys
from typing import List

import pdfplumber

from backend.core.exceptions import CustomException
from backend.core.logger import logging


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from every page of a bank statement PDF and
    concatenates it into a single string.
    """
    try:
        logging.info(f"Extracting text from PDF: {pdf_path}")
        pages_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages_text.append(text)

                # Tables (common in bank statements for transaction lists)
                # often extract more reliably as structured rows than as
                # free text, so pull them separately and append as text.
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(cell or "" for cell in row)
                        pages_text.append(row_text)

        full_text = "\n".join(pages_text)
        logging.info(f"Extracted {len(full_text)} characters from {len(pdf.pages)} pages")
        return full_text

    except Exception as e:
        raise CustomException(e, sys)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Splits text into overlapping chunks suitable for retrieval. Overlap
    keeps a transaction or statement line from being cut in half right at
    a chunk boundary and losing the context on either side of it.
    """
    try:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap

        logging.info(f"Split text into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        raise CustomException(e, sys)
