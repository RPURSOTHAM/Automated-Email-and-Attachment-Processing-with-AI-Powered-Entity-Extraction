# AI-Powered Email and Document Processing System

## Overview

This project is an AI-driven system designed to automate the processing of emails and documents, particularly useful for finance and client servicing teams. It extracts key information such as client names, addresses, and invoice details using NLP techniques and OCR technologies.


## Features

- NER with BERT: Extracts client names, addresses, and invoice information from email bodies and attachments.
- OCR with EasyOCR & PyMuPDF: Handles scanned PDFs and image-based documents for robust text extraction.
- Regex-based Parsing: Structured extraction of specific patterns and entities.
- Text Summarization & QnA: Uses Transformer models (BART, RoBERTa) to summarize long documents and answer user queries.
- Multi-format Support: Works with .pdf, .eml, and image files.

## Tech Stack

- Python
- BERT, RoBERTa, BART (via Hugging Face Transformers)
- EasyOCR
- PyMuPDF (fitz)
- RAG
- Regex
- Flask (optional, for deployment)
  
