Invoice Information Extraction using LayoutLMv3
A simple yet powerful system that extracts key information from invoice images and PDFs using a fine-tuned LayoutLMv3 model.

Features
📄 Processes both images and PDFs - Upload any invoice in common formats

🧠 Uses LayoutLMv3 - State-of-the-art document understanding model

🔍 Automatic text detection - Integrated OCR with Tesseract

📊 Structured JSON output - Clean, organized extraction results

🚀 Easy to use - Simple interface for quick testing

Quick Start
1. Installation & Setup
   # Run in Google Colab or local environment
!pip install transformers datasets torch torchvision pillow pdf2image pytesseract
!apt-get install -y poppler-utils tesseract-ocr
# Copy and run the entire code from the latest version
# The system will:
# 1. Load and preprocess the FUNSD dataset
# 2. Fine-tune LayoutLMv3 model
# 3. Save the trained model
# 4. Provide an interface for invoice extraction

3. Extract Information from Your Invoice
After running the code, simply:

Upload your invoice (image or PDF) when prompted

Wait for processing

View the extracted information in JSON format

Results are automatically saved to extracted_invoice_info.json

How It Works
Model Training
Dataset: FUNSD (Form Understanding in Noisy Scanned Documents)

Base Model: Microsoft's LayoutLMv3-base

Training: 2 epochs with learning rate 5e-5

Labels: Header, Question, Answer entities

Extraction Process
OCR Processing: Uses Tesseract to detect text and bounding boxes

Bounding Box Normalization: Converts coordinates to LayoutLMv3 format (0-1000 range)

Entity Recognition: Model predicts entities in the document

Result Aggregation: Groups entities into structured JSON output

Output Format
The system returns a JSON object with extracted entities:
{
  "header": "INVOICE",
  "question": "Invoice Number",
  "answer": "INV-2024-001",
  "date": "2024-01-15",
  "total": "$1,500.00"
}

Limitations
Processes only the first page of PDF documents

Extraction quality depends on invoice clarity and layout

May not extract all fields from complex or unusual invoice formats

Future Improvements
Support for multi-page PDF processing

Additional entity types specific to invoices

Improved handling of tables and line items

Confidence scores for extracted fields

License
This project uses pre-trained models from Microsoft Research under their respective licenses.
