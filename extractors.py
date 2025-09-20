from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join(paragraph.text or "" for paragraph in doc.paragraphs)