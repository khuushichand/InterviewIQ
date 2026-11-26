"""
Resume Text Extraction Utility for InterviewIQ.

Supports:
- PDF extraction via pypdf (in-memory, no OCR, no disk persistence)
- DOCX extraction via python-docx (paragraphs + tables, in-memory)
- Strict file format validation (.pdf, .docx)
- Maximum file size enforcement (5 MB)
- Text sanitization and whitespace normalization
- Guardrails against corrupted or unreadable documents
"""

import io
import os
import re
from typing import Tuple

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


class ResumeExtractionError(Exception):
    """Raised when resume extraction fails due to formatting, corruption, or reading issues."""
    pass


class UnsupportedFileFormatError(ResumeExtractionError):
    """Raised when an unsupported file type is provided."""
    pass


class FileTooLargeError(ResumeExtractionError):
    """Raised when the uploaded file exceeds the 5 MB limit."""
    pass


class EmptyResumeError(ResumeExtractionError):
    """Raised when the document contains no extractable or readable text."""
    pass


def normalize_extracted_text(text: str) -> str:
    """
    Normalizes extracted resume text:
    - Removes null bytes and unprintable control characters.
    - Preserves newlines and structural paragraph spacing.
    - Normalizes multi-spaces to single space while keeping section layout readable.
    """
    if not text:
        return ""
    # Remove null bytes and non-printable control chars (except \n, \t, \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Clean up lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Collapse 3+ consecutive newlines to 2 (paragraph break)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf without writing to disk."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        raise ResumeExtractionError("PDF extraction library (pypdf) is not available.")

    try:
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)
        if len(reader.pages) == 0:
            raise EmptyResumeError("PDF contains no pages.")
        
        extracted_pages = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
            except Exception as e:
                # Page level warning; continue extracting remaining pages
                continue

        full_text = "\n\n".join(extracted_pages)
        return normalize_extracted_text(full_text)
    except EmptyResumeError:
        raise
    except (PdfReadError, Exception) as e:
        raise ResumeExtractionError(f"Corrupted or unreadable PDF document: {type(e).__name__}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text from DOCX bytes using python-docx without writing to disk."""
    try:
        import docx
    except ImportError:
        raise ResumeExtractionError("DOCX extraction library (python-docx) is not available.")

    try:
        stream = io.BytesIO(file_bytes)
        doc = docx.Document(stream)

        text_parts = []
        # Extract paragraph text
        for p in doc.paragraphs:
            if p.text:
                text_parts.append(p.text)

        # Extract table text (many resumes format skills/experience in tables)
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    # De-duplicate adjacent identical cells resulting from merged cells
                    deduped = []
                    for c in row_cells:
                        if not deduped or c != deduped[-1]:
                            deduped.append(c)
                    text_parts.append(" | ".join(deduped))

        full_text = "\n".join(text_parts)
        return normalize_extracted_text(full_text)
    except Exception as e:
        raise ResumeExtractionError(f"Corrupted or unreadable DOCX document: {type(e).__name__}")


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    """
    Main extraction interface.
    Validates file size, extension, extracts text without filesystem leakage,
    and returns sanitized text.
    """
    if not file_bytes:
        raise EmptyResumeError("Uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext or 'unknown'}'. Please upload a {allowed} file."
        )

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        text = extract_text_from_docx(file_bytes)
    else:
        raise UnsupportedFileFormatError(f"Unsupported extension {ext}")

    if not text or len(text.strip()) < 20:
        raise EmptyResumeError(
            "The resume contains no readable text. Scanned images or password-protected files are not supported."
        )

    return text
