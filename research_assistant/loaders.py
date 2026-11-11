"""Turns a file into plain text, one reader per file type.

Supported: .md, .txt, .pdf, .docx, .xlsx (and .csv)
Each reader returns a single string. Blank lines separate paragraphs,
which is what the chunker in retriever.py splits on.
"""
import csv
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # scanned pages with no text layer come back empty
            pages.append(f"[page {number}]\n{text}")
    return "\n\n".join(pages)


def read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:  # tables are not part of document.paragraphs
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), data_only=True)  # values, not formulas
    sheets = []
    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            sheets.append(f"[sheet: {sheet.title}]\n" + "\n".join(rows))
    return "\n\n".join(sheets)


def read_csv(path: Path) -> str:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = [" | ".join(cell.strip() for cell in row if cell.strip())
                for row in csv.reader(handle)]
    return "\n".join(r for r in rows if r)


READERS = {
    ".md": read_text,
    ".txt": read_text,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".xlsx": read_xlsx,
    ".csv": read_csv,
}

SUPPORTED = ", ".join(sorted(READERS))


def read_any(path: Path) -> str:
    """Reads one file. Returns an empty string for an unsupported type."""
    reader = READERS.get(path.suffix.lower())
    return reader(path) if reader else ""
