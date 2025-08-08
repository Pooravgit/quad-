# ingest.py
import os
import fitz  # PyMuPDF
import pdfplumber
from docx import Document
import extract_msg
# from pdf2image import convert_from_path
import pytesseract
import hashlib


def normalize_whitespace(text):
    return " ".join(text.split())


def detect_repeated_lines_across_pages(pages_texts, threshold=3):
    from collections import Counter
    line_counter = Counter()
    for page in pages_texts:
        lines = [l.strip() for l in page.splitlines() if l.strip()]
        line_counter.update(set(lines))
    return {line for line, count in line_counter.items() if count >= threshold}


def clean_page_text(text, common_lines_to_remove):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in common_lines_to_remove:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def read_pdf_fulltext_with_ocr_fallback(file_path, ocr_threshold_chars=50):
    full_text_pages = []
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            if len(text.strip()) < ocr_threshold_chars:
                try:
                    pil_pages = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1, dpi=200)
                    if pil_pages:
                        ocr_text = pytesseract.image_to_string(pil_pages[0])
                        page_text = ocr_text
                    else:
                        page_text = text
                except Exception:
                    page_text = text
            else:
                page_text = text
            full_text_pages.append(f"-- PAGE {page_num + 1} --\n{page_text}")
    common = detect_repeated_lines_across_pages(full_text_pages)
    cleaned_pages = [clean_page_text(p, common) for p in full_text_pages]
    combined = "\n\n".join(cleaned_pages)
    return normalize_whitespace(combined)


def extract_tables_pdfplumber(file_path):
    tables_data = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for ti, table in enumerate(tables):
                if table and len(table) >= 1:
                    header = table[0]
                    rows = table[1:]
                    table_str = " | ".join([h if h else "" for h in header]) + "\n"
                    for row in rows:
                        table_str += " | ".join([cell if cell else "" for cell in row]) + "\n"
                else:
                    table_str = ""
                if table_str.strip():
                    tables_data.append({
                        "page": i + 1,
                        "table_index": ti,
                        "text": normalize_whitespace(table_str.strip())
                    })
    return tables_data


def read_docx(file_path):
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return normalize_whitespace("\n".join(paragraphs))


def read_email(file_path):
    if file_path.lower().endswith(".eml"):
        import email
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f)
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body = payload.decode("utf-8", errors="ignore")
            else:
                body = str(payload)
            return normalize_whitespace(body)
    elif file_path.lower().endswith(".msg"):
        msg = extract_msg.Message(file_path)
        body = msg.body or ""
        return normalize_whitespace(body)
    return ""


def generate_clause_id(source, page, text_snippet):
    h = hashlib.sha256(f"{source}|{page}|{text_snippet[:100]}".encode("utf-8")).hexdigest()
    return h


def load_documents(folder_path):
    docs = []
    for filename in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            full_text = read_pdf_fulltext_with_ocr_fallback(path)
            tables = extract_tables_pdfplumber(path)
            for t in tables:
                t["clause_id"] = generate_clause_id(filename, t.get("page"), t["text"])
            docs.append({
                "source": filename,
                "type": "pdf",
                "full_text": full_text,
                "tables": tables
            })
        elif filename.lower().endswith(".docx"):
            full_text = read_docx(path)
            docs.append({
                "source": filename,
                "type": "docx",
                "full_text": full_text,
                "tables": []
            })
        elif filename.lower().endswith((".eml", ".msg")):
            full_text = read_email(path)
            docs.append({
                "source": filename,
                "type": "email",
                "full_text": full_text,
                "tables": []
            })
    return docs
