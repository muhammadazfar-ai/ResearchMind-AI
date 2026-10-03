import os
import re

from src.config import MAX_CHARS_PER_DOCUMENT, MAX_UPLOAD_FILES

SUPPORTED = (".pdf", ".txt", ".md")


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(paths) -> tuple:
    """Reads uploaded PDF/TXT/MD files into source dicts.

    Returns (documents, warnings). Each document is {title, url, content, kind}.
    Problems with one file never stop the others.
    """
    documents, warnings = [], []
    paths = [getattr(p, "name", p) for p in (paths or [])]

    if len(paths) > MAX_UPLOAD_FILES:
        warnings.append(f"Only the first {MAX_UPLOAD_FILES} files are used.")
        paths = paths[:MAX_UPLOAD_FILES]

    for path in paths:
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED:
            warnings.append(f"`{name}` skipped: unsupported file type.")
            continue
        try:
            text = _read_pdf(path) if ext == ".pdf" else _read_text(path)
        except Exception as e:
            warnings.append(f"`{name}` could not be read: {e}")
            continue

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 50:
            warnings.append(f"`{name}` has no readable text (a scanned PDF needs OCR first).")
            continue
        if len(text) > MAX_CHARS_PER_DOCUMENT:
            text = text[:MAX_CHARS_PER_DOCUMENT]
            warnings.append(f"`{name}` is long; only the first {MAX_CHARS_PER_DOCUMENT:,} characters are used.")

        documents.append({"title": name, "url": "", "content": text, "kind": "document"})

    return documents, warnings
