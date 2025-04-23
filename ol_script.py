import email
import re
import os
from email import policy
import nltk
from bs4 import BeautifulSoup
import pdfplumber
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def extract_email_body(eml_path):
    """
    Opens and parses the .eml file.
    Extracts the plain text email body and saves any PDF attachment.
    
    Returns:
        email_body (str): the extracted text from the email.
        pdf_attachment_path (str or None): path to the saved PDF attachment.
    """
    pdf_attachment_path = None
    email_body = ""
    
    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    
    for part in msg.walk():
        content_type = part.get_content_type()
        filename = part.get_filename()
        
        # If it's the plain text body (not an attachment)
        if content_type == "text/plain" and not filename:
            email_body += part.get_content()
        
        # If it's a PDF attachment, save it locally
        if filename and filename.lower().endswith('.pdf'):
            pdf_attachment_path = filename  # You can modify the saving path if needed
            with open(pdf_attachment_path, "wb") as pdf_file:
                pdf_file.write(part.get_payload(decode=True))
    
    return email_body, pdf_attachment_path


def extract_fields_from_text(text):
    """
    Uses regular expressions to extract key fields from the email body.
    Expected format in the email:
       Name: <name>
       Education:<education>
       Skillset: <skill1>, <skill2>, ...
    
    Returns:
        fields (dict): Dictionary containing extracted fields.
    """
    fields = {}
    
    # Extract Name
    name_match = re.search(r'\Name:\\s*(.+)', text)
    if name_match:
        fields['name'] = name_match.group(1).strip()
    
    # Extract Education
    education_match = re.search(r'\Education\:\s*(.+)', text)
    if education_match:
        fields['education'] = education_match.group(1).strip()
    
    # Extract Skillset and split into list
    skills_match = re.search(r'\Skillset:\\s*(.+)', text)
    if skills_match:
        skills_str = skills_match.group(1).strip()
        fields['skills'] = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
    
    return fields


def extract_pdf_text(pdf_path):
    text = ""
    if os.path.exists(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    return text.strip()


def search_fields_in_pdf(email_fields, pdf_text):
    """
    Searches for each field value extracted from the email within the PDF text.
    For non-list fields, it uses a regex word-boundary match.
    For list fields (like skills), it searches each item individually.
    
    Returns:
        results (dict): Dictionary with search results for each field.
    """
    results = {}
    for key, value in email_fields.items():
        if isinstance(value, list):
            # For each item in the list, search using regex (case-insensitive)
            found_items = []
            for item in value:
                pattern = rf'\b{re.escape(item)}\b'
                if re.search(pattern, pdf_text, re.IGNORECASE):
                    found_items.append(item)
            results[key] = found_items if found_items else "Not Found"
        else:
            pattern = rf'\b{re.escape(value)}\b'
            if re.search(pattern, pdf_text, re.IGNORECASE):
                results[key] = value
            else:
                results[key] = "Not Found"
    return results


def extract_keywords_nltk(text, top_n=10):
    """
    Extracts top keywords using NLTK.
    
    Steps:
      1. Clean any HTML from the text.
      2. Tokenize using NLTK.
      3. Remove punctuation and stopwords.
      4. Count word frequencies and return the most common tokens.
      
    Returns:
        keywords (list): List of top keywords.
    """
    # Clean HTML if present
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    
    # Tokenize text
    tokens = word_tokenize(clean_text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Frequency analysis
    freq = Counter(filtered_tokens)
    common_tokens = freq.most_common(top_n)
    keywords = [word for word, count in common_tokens]
    return keywords


def extract_keywords_tfidf(text, top_n=10):
    """
    Extracts keywords using the TF-IDF vectorizer.
    
    Returns:
        top_n_keywords (list): List of keywords with the highest TF-IDF scores.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    # Get indices of top scores in descending order
    sorted_indices = tfidf_matrix.toarray().flatten().argsort()[::-1]
    top_n_keywords = [feature_array[i] for i in sorted_indices[:top_n]]
    return top_n_keywords


def main():
    # Specify the path to your .eml file
    eml_path = r"m:\resume_match\File (6).eml"  # Update with your actual path/filename
    email_body, pdf_attachment_path = extract_email_body(eml_path)

    # Print the email body for debugging purposes
    print("=== Email Body ===")
    print(email_body)
    print("\n")

    # Extract structured fields from the email text
    email_fields = extract_fields_from_text(email_body)
    print("=== Extracted Email Fields ===")
    print(email_fields)
    print("\n")

    # Extract keywords from the email body (optional)
    nltk_keywords = extract_keywords_nltk(email_body, top_n=10)
    tfidf_keywords = extract_keywords_tfidf(email_body, top_n=10)
    print("=== Keywords using NLTK ===")
    print(nltk_keywords)
    print("\n=== Keywords using TF-IDF ===")
    print(tfidf_keywords)
    print("\n")

    # Extract text from the attached PDF
    if pdf_attachment_path and os.path.exists(pdf_attachment_path):
        pdf_text = extract_pdf_text(pdf_attachment_path)
    else:
        pdf_text = ""

    print("=== Extracted PDF Text (Preview) ===")
    print(pdf_text[:1000])  # Print the first 1000 characters as a preview
    print("\n")

    # Search for the value of each field in the PDF text using regex
    search_results = search_fields_in_pdf(email_fields, pdf_text)

    print("--- Field Search Results in PDF ---")
    print(f"Name in PDF: {search_results.get('name')}")
    print(f"Education in PDF: {search_results.get('education')}")
    print(f"Skills in PDF: {search_results.get('skills')}")
    
    # Optionally, remove the saved PDF file after processing
    if pdf_attachment_path and os.path.exists(pdf_attachment_path):
        os.remove(pdf_attachment_path)


if __name__ == "__main__":
    main()