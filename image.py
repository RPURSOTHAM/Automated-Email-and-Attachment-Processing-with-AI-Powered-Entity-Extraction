import email
import re
import os
import nltk
import pdfplumber
import easyocr
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from email import policy
from docx import Document
from transformers import pipeline

# Download necessary NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load Transformer-based models
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Load model for English

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

def extract_keywords(text):
    """Extracts key information using rule-based keyword matching."""
    keywords = {
        "Invoice": ["invoice", "bill", "payment due", "amount", "total"],
        "Client": ["client name", "customer", "buyer", "purchaser"],
        "Date": ["date", "issued on", "created on", "due date"],
    }
    extracted_info = {key: [] for key in keywords}
    
    for category, words in keywords.items():
        for word in words:
            matches = re.findall(rf"\b{word}\b\s*[:\-]?\s*([\w\s\d-]+)", text, re.IGNORECASE)
            if matches:
                extracted_info[category].extend(matches)
    
    return extracted_info

def extract_entities_transformers(text):
    """Extracts named entities using a Transformer-based NER model (more accurate than spaCy)."""
    ner_results = ner_pipeline(text)
    entities = {}
    for entity in ner_results:
        entity_text = entity['word'].replace("##", "")
        entity_label = entity['entity'].split("_")[-1]  # Extract label like PER, LOC, etc.
        if entity_label not in entities:
            entities[entity_label] = []
        entities[entity_label].append(entity_text)
    return entities

def extract_entities_nltk(text):
    """Extract named entities using NLTK (lighter alternative to spaCy)."""
    from nltk import word_tokenize, pos_tag, ne_chunk
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags, binary=True)

    named_entities = []
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NE':
            named_entities.append(' '.join([leaf[0] for leaf in subtree.leaves()]))

    return named_entities

def summarize_text(text, min_length=30, max_length=150):
    """Summarizes long text using Transformers."""
    if len(text.split()) < 50:
        return text
    summary = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
    return summary[0]["summary_text"]

def answer_question(text, question):
    """Uses a Transformer model to answer specific questions from text."""
    response = qa_pipeline(question=question, context=text)
    return response["answer"]

def extract_regex_patterns(text):
    """Extracts structured data like emails, phone numbers, and invoice numbers using regex."""
    return {
        "Emails": re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text),
        "Phone Numbers": re.findall(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}", text),
        "Invoice Numbers": re.findall(r"\bINV-\d{5,10}\b", text),
    }

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
    """Extracts images from a PDF file, saves them, and extracts text from them using OCR."""
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
            img_ext = base_image["ext"]  # Usually "png" or "jpeg"
            image_filename = os.path.join(output_folder, f"page{page_number + 1}_img{img_index + 1}.{img_ext}")
            
            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_filename)

            # Perform OCR on the extracted image
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

    print("\n=== Extracted Email Body ===\n", email_body)
    print("\n=== Rule-Based Extraction ===\n", extract_keywords(email_body))
    print("\n=== Named Entities (Transformers) ===\n", extract_entities_transformers(email_body))
    print("\n=== Named Entities (NLTK) ===\n", extract_entities_nltk(email_body))
    print("\n=== Regex Extracted Data ===\n", extract_regex_patterns(email_body))
    print("\n=== Summarized Email ===\n", summarize_text(email_body))
    print("\n=== Answer to Question ===\n", answer_question(email_body, "What is the invoice amount?"))
    print("\n=== Extracted Attachment Text ===\n", extracted_text[:1000])
    print("\n=== Named Entities (Attachment) ===\n", extract_entities_transformers(extracted_text))
    print("\n=== Summarized Attachment ===\n", summarize_text(extracted_text))
    print("\n=== Extracted Images ===\n", extracted_images)
    print("\n=== OCR Extracted Text from Images ===\n", extracted_images_text)

if __name__ == "__main__":
    main()
