import email
import re
import os
import nltk
import pdfplumber
import spacy
import easyocr
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from email import policy
from docx import Document
from ollama import Client
from PIL import Image, ImageOps

# Download necessary NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Load model for English

# ✅ Connect to Ollama's local server
ollama_client = Client(host='http://localhost:11434')

def summarize_with_ollama(text):
    """Summarizes given text using the local LLM (Ollama with LLaMA2)."""
    if len(text.split()) < 50:
        return text
    print("\n[INFO] Summarizing text using local LLM (LLaMA2)...")
    response = ollama_client.chat(
        model='llama2',
        messages=[{'role': 'user', 'content': f'Summarize this text:\n{text}'}]
    )
    return response['message']['content']

def extract_email_body(eml_path, output_folder="attachments"):
    """Extracts email body and saves all attachments of any format."""
    email_body = ""
    attachments = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
        for part in msg.walk():
            content_type = part.get_content_type()
            filename = part.get_filename() or f"attachment_{len(attachments)+1}"

            if content_type == "text/plain" and not filename:
                email_body += part.get_payload(decode=True).decode(errors='ignore')
            elif part.get_payload(decode=True):
                attachment_path = os.path.join(output_folder, filename)
                with open(attachment_path, "wb") as attachment_file:
                    attachment_file.write(part.get_payload(decode=True))
                attachments.append(attachment_path)
                print(f"[INFO] Extracted attachment: {attachment_path}")

    return email_body.strip(), attachments

def preprocess_image(image_path):
    """Preprocess the image for better OCR results."""
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert colors if needed
        image = image.resize((image.width * 2, image.height * 2))  # Upscale
        return image
    except Exception as e:
        print(f"[ERROR] Failed to preprocess image: {e}")
        return None

def extract_image_text(image_path):
    """Extract text from image using OCR with preprocessing."""
    print(f"[INFO] Extracting text from image: {image_path}")
    image = preprocess_image(image_path)
    if image:
        result = reader.readtext(np.array(image), detail=0)
        return "\n".join(result)
    else:
        print("[WARNING] Could not process image correctly.")
        return ""

def extract_pdf_text(pdf_path):
    """Extract text from PDF with OCR fallback."""
    text = ""
    if os.path.exists(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        if not text:
            text = extract_text_ocr(pdf_path)
    return text.strip()

def extract_text_ocr(pdf_path):
    """Extract text from scanned PDFs using OCR."""
    text = ""
    images = convert_from_path(pdf_path)
    for image in images:
        result = reader.readtext(image, detail=0)
        text += "\n".join(result) + "\n"
    return text.strip()

def extract_docx_text(docx_path):
    """Extract text from DOCX file."""
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_txt_text(txt_path):
    """Extract text from TXT file."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def process_attachment(attachment_path):
    """Processes attachments of various formats."""
    ext = os.path.splitext(attachment_path)[1].lower()
    print(f"[INFO] Processing {attachment_path}...")
    if ext == ".pdf":
        return extract_pdf_text(attachment_path)
    elif ext == ".docx":
        return extract_docx_text(attachment_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        return extract_image_text(attachment_path)
    elif ext == ".txt":
        return extract_txt_text(attachment_path)
    else:
        print(f"[WARNING] Unsupported file type: {ext}, skipping.")
        return ""

def cleanup_attachments(folder="attachments"):
    """Remove extracted attachments after processing."""
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            os.remove(file_path)
        os.rmdir(folder)
        print("[INFO] Attachments removed after processing.")

def main(eml_path):
    """Main function to process emails and attachments with Ollama summarization."""
    print("[INFO] Extracting email body and attachments...")
    email_body, attachments = extract_email_body(eml_path)

    all_text = email_body + "\n"
    for attachment in attachments:
        all_text += process_attachment(attachment) + "\n"

    summarized_content = summarize_with_ollama(all_text)

    print("\n=== Extracted Email & Attachments Text ===\n", all_text[:1000])
    print("\n=== Summarized Content (Local LLM) ===\n", summarized_content)

    cleanup_attachments()

if __name__ == "__main__":
    eml_file_path = "m:\\resume_match\\File (6).eml"  # Update path as needed
    main(eml_file_path)
