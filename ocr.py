import email
import re
import os
import nltk
import pdfplumber
import spacy
import easyocr
import fitz  # PyMuPDF
import requests
import json
from pdf2image import convert_from_path
from email import policy
from docx import Document

# Download necessary NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Load model for English


def query_ollama(prompt):
    """Queries the local LLM running on Ollama for entity extraction."""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "llama2", "prompt": prompt},
        stream=True
    )
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_line = json.loads(line.decode('utf-8'))
                full_response += json_line.get("response", "")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
    return full_response.strip()


def extract_email_body(eml_path):
    """Extracts the email body and saves the first PDF or DOCX attachment."""
    email_body = ""
    attachment_path = None

    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
        for part in msg.walk():
            content_type = part.get_content_type()
            filename = part.get_filename()

            if content_type == "text/plain" and not filename:
                email_body += part.get_payload(decode=True).decode(errors='ignore')

            if filename and (filename.lower().endswith('.pdf') or filename.lower().endswith('.docx')):
                attachment_path = filename
                with open(attachment_path, "wb") as attachment_file:
                    attachment_file.write(part.get_payload(decode=True))
                break  # Stop after saving the first attachment

    return email_body.strip(), attachment_path


def extract_entities_ollama(text):
    """Uses Ollama's local LLM to extract client name and address from text."""
    prompt = f"Extract only the client name and address from the following text:\n{text}\nReturn in JSON format with 'name','education'and 'skills_set'."
    llm_response = query_ollama(prompt)
    return llm_response


def extract_pdf_text(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    if os.path.exists(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()


def extract_text_ocr(pdf_path):
    """Extracts text from scanned PDFs using EasyOCR."""
    text = ""
    if os.path.exists(pdf_path):
        images = convert_from_path(pdf_path)  # Convert PDF to images
        for image in images:
            result = reader.readtext(image, detail=0)  # Extract text with EasyOCR
            text += "\n".join(result) + "\n"
    return text.strip()


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """Extracts images from a PDF file and performs OCR on them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)
    image_paths = []
    extracted_text = ""

    for page_number in range(len(doc)):
        for img_index, img in enumerate(doc[page_number].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image_filename = os.path.join(output_folder, f"page{page_number + 1}_img{img_index + 1}.{img_ext}")

            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_filename)

            # OCR on the extracted image
            ocr_text = reader.readtext(image_filename, detail=0)
            extracted_text += "\n".join(ocr_text) + "\n"

    return image_paths, extracted_text.strip()


def extract_images_from_docx(docx_path, output_folder="extracted_docx_images"):
    """Extracts images from a DOCX file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    doc = Document(docx_path)
    image_paths = []

    for rel in doc.part._rels:
        rel_obj = doc.part._rels[rel]
        if "image" in rel_obj.target_ref:
            img_data = rel_obj.target_part.blob
            img_ext = rel_obj.target_ref.split(".")[-1]
            img_filename = os.path.join(output_folder, f"image_{len(image_paths) + 1}.{img_ext}")
            with open(img_filename, "wb") as f:
                f.write(img_data)
            image_paths.append(img_filename)
    return image_paths


def main():
    eml_path = "m:\\resume_match\\new file.eml"  # Provide the correct path here
    email_body, attachment_path = extract_email_body(eml_path)

    extracted_text = ""
    extracted_images = []
    extracted_images_text = ""

    if attachment_path:
        if attachment_path.lower().endswith(".pdf"):
            extracted_text = extract_pdf_text(attachment_path) or extract_text_ocr(attachment_path)
            extracted_images, extracted_images_text = extract_images_from_pdf(attachment_path)
        elif attachment_path.lower().endswith(".docx"):
            extracted_images = extract_images_from_docx(attachment_path)

    # === Output results ===
    print("\n=== Extracted Email Body ===\n", email_body)
    print("\n=== LLM Extracted Entities from Email Body ===\n", extract_entities_ollama(email_body))
    print("\n=== Extracted Attachment Text ===\n", extracted_text[:1000])
    print("\n=== LLM Extracted Entities from Attachment ===\n", extract_entities_ollama(extracted_text))
    print("\n=== Extracted Images ===\n", extracted_images)
    print("\n=== OCR Extracted Text from Images ===\n", extracted_images_text)
    print("\n=== LLM Extracted Entities from Image Text ===\n", extract_entities_ollama(extracted_images_text))

    # Optional: Remove attachment after processing
    # if attachment_path and os.path.exists(attachment_path):
    #     os.remove(attachment_path)


if __name__ == "__main__":
    main()