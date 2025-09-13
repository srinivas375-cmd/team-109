# backend/utils.py
import io
from pdfminer.high_level import extract_text
from docx import Document
import re

async def extract_text_from_file(file):
    contents = await file.read()
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        # write to temp and extract
        with open("tmp_upload.pdf", "wb") as f:
            f.write(contents)
        return extract_text("tmp_upload.pdf")
    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(contents))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return contents.decode("utf-8")

def split_into_clauses(text):
    # Very simple: split on double newlines or "Clause" markers.
    text = re.sub(r'\r\n', '\n', text)
    if "Clause" in text or "CLAUSE" in text:
        # split by lines starting with "Clause" or a number
        clauses = re.split(r'\n(?=(Clause|CLAUSE|\d+\.) )', text)
        # fallback simpler split:
    clauses = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip())>20]
    # limit to first 200 clauses for speed
    return clauses[:200]
