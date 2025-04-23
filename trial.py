import os
import email
import re
import nltk
import pdfplumber
import spacy
import easyocr
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from email import policy
from docx import Document
from PIL import Image, ImageOps

# Handle optional imports and environment restrictions
def safe_import(module_name, install_name=None):
    try:
        __import__(module_name)
        print(f"[INFO] {module_name} is available.")
    except ImportError:
        print(f"[ERROR] {module_name} is not installed. Please install using: pip install {install_name or module_name}")

# Checking essential modules
safe_import("nltk")
safe_import("spacy")
safe_import("easyocr")
safe_import("pdf2image")
safe_import("fitz", "PyMuPDF")

# Download NLTK datasets if not present
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"[ERROR] NLTK resource download failed: {e}")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[ERROR] spaCy model not found. Run: python -m spacy download en_core_web_sm")

reader = easyocr.Reader(['en'])

def extract_email_body(eml_path, output_folder="attachments"):
    """Extracts email body and saves attachments."""
    email_body = ""
    attachments = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with open(eml_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
            for part in msg.walk():
                content_type = part.get_content_type()
                filename = part.get_filename() or f"attachment_{len(attachments) + 1}"

                if content_type == "text/plain" and not filename:
                    email_body += part.get_payload(decode=True).decode(errors='ignore')
                elif part.get_payload(decode=True):
                    attachment_path = os.path.join(output_folder, filename)
                    with open(attachment_path, "wb") as attachment_file:
                        attachment_file.write(part.get_payload(decode=True))
                    attachments.append(attachment_path)
                    print(f"[INFO] Attachment saved: {attachment_path}")
    except Exception as e:
        print(f"[ERROR] Failed to extract email body or attachments: {e}")

    return email_body.strip(), attachments

def preprocess_image(image_path):
    """Preprocess image for OCR."""
    try:
        image = Image.open(image_path).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((image.width * 2, image.height * 2))
        return image
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return None

def extract_image_text(image_path):
    """Extract text from image with OCR."""
    print(f"[INFO] Extracting text from image: {image_path}")
    image = preprocess_image(image_path)
    if image:
        import numpy as np
        return "\n".join(reader.readtext(np.array(image), detail=0))
    return ""

def extract_pdf_text(pdf_path):
    """Extract text from PDF with OCR fallback."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text if text else extract_text_ocr(pdf_path)
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return ""

def extract_text_ocr(pdf_path):
    """OCR for scanned PDFs."""
    text = ""
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            text += "\n".join(reader.readtext(img, detail=0)) + "\n"
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
    return text.strip()

def process_attachment(attachment_path):
    """Process attachments based on extension."""
    ext = os.path.splitext(attachment_path)[1].lower()
    print(f"[INFO] Processing attachment: {attachment_path}")
    if ext == ".pdf":
        return extract_pdf_text(attachment_path)
    elif ext == ".docx":
        return "\n".join([p.text for p in Document(attachment_path).paragraphs])
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_image_text(attachment_path)
    else:
        print(f"[WARNING] Unsupported file type: {ext}")
        return ""

def cleanup_attachments(folder="attachments"):
    """Clean up attachment folder."""
    if os.path.exists(folder):
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)
        print("[INFO] Cleaned attachment folder.")

def main(eml_path):
    """Main function for email processing."""
    print("[INFO] Starting email processing...")
    if not os.path.exists(eml_path):
        print(f"[ERROR] File not found: {eml_path}")
        return

    email_body, attachments = extract_email_body(eml_path)
    all_text = email_body + "\n" + "\n".join(process_attachment(a) for a in attachments)

    print("\n=== Extracted Content Preview ===\n", all_text[:1000])

    cleanup_attachments()
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    eml_file_path = os.path.abspath("m://resume_match//File (6).eml")
    print(f"[INFO] Using file path: {eml_file_path}")
    main(eml_file_path)
