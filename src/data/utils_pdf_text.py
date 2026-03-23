"""
Utilities for converting uploaded PDFs to plain text and splitting into clauses.
"""

from typing import List
from pypdf import PdfReader
import re


def pdf_to_text(path) -> str:
    """
    Read a PDF file and return concatenated text.
    TODO: handle encoding / errors.
    """
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def split_into_clauses(text: str) -> List[str]:
    """
    Split ToS text into clauses/sentences.

    TODO:
    - Improve sentence splitting (e.g., use nltk or spacy).
    - For now: simple regex-based split on periods + newlines.
    """
    rough = re.split(r"[.\n]+", text)
    clauses = [c.strip() for c in rough if c.strip()]
    return clauses
