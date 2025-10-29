
#pip install transformers torch torchvision pillow pandas numpy pdf2image pytesseract datasets
#apt-get install -y poppler-utils tesseract-ocr
# pip install seqeval

# Step 2: Import Libraries
import torch
import torch.nn as nn
from transformers import (LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
                         TrainingArguments, Trainer)
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from google.colab import files
import cv2
import pytesseract
from pdf2image import convert_from_path
import os
from seqeval.metrics import classification_report, f1_score
from typing import Dict, List, Any, Tuple
import re
from collections import defaultdict

#  Load Pre-trained Model and Processor for Key-Value Extraction
def load_model_and_processor():
    """Load pre-trained LayoutLMv3 model and processor for key-value extraction"""
    print("Loading LayoutLMv3 model and processor...")

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )

    # Define comprehensive label names for key-value extraction
    label_names = [
        "O",
        "B-KEY", "I-KEY",
        "B-VALUE", "I-VALUE",
        "B-HEADER", "I-HEADER",
        "B-TABLE", "I-TABLE"
    ]

    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for i, label in enumerate(label_names)}

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id
    )

    print("Model and processor loaded successfully!")
    return processor, model, label2id, id2label

processor, model, label2id, id2label = load_model_and_processor()

#  Advanced Key-Value Pair Extractor Class
class AdvancedInvoiceExtractor:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Common invoice key patterns
        self.key_patterns = {
            'invoice': ['invoice', 'inv', 'bill', 'receipt'],
            'date': ['date', 'issued', 'created', 'billing date'],
            'due_date': ['due date', 'payment due', 'due'],
            'total': ['total', 'amount due', 'balance', 'grand total'],
            'subtotal': ['subtotal', 'sub total'],
            'tax': ['tax', 'vat', 'gst', 'sales tax'],
            'vendor': ['vendor', 'seller', 'from', 'company', 'supplier'],
            'customer': ['customer', 'client', 'bill to', 'ship to', 'sold to'],
            'address': ['address', 'street', 'city', 'state', 'zip', 'country'],
            'phone': ['phone', 'tel', 'telephone'],
            'email': ['email', 'e-mail'],
            'website': ['website', 'web', 'site'],
            'item': ['item', 'description', 'product', 'service'],
            'quantity': ['quantity', 'qty'],
            'price': ['price', 'rate', 'unit price'],
            'amount': ['amount', 'line total'],
            'terms': ['terms', 'payment terms'],
            'notes': ['notes', 'remarks', 'comments']
        }

    def normalize_bbox(self, bbox, image_size):
        """Normalize bounding box coordinates to 0-1000 range"""
        width, height = image_size
        x0, y0, x1, y1 = bbox

        normalized_x0 = max(0, min(1000, int(1000 * (x0 / width))))
        normalized_y0 = max(0, min(1000, int(1000 * (y0 / height))))
        normalized_x1 = max(0, min(1000, int(1000 * (x1 / width))))
        normalized_y1 = max(0, min(1000, int(1000 * (y1 / height))))

        return [normalized_x0, normalized_y0, normalized_x1, normalized_y1]

    def extract_text_and_boxes(self, image_path):
        """Extract text and bounding boxes with improved OCR"""
        try:
            if image_path.lower().endswith('.pdf'):
                images = convert_from_path(image_path, dpi=300)
                image = images[0]
            else:
                image = Image.open(image_path).convert("RGB")

            width, height = image.size

            # Improved OCR configuration
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

            words = []
            boxes = []
            confidences = []

            current_word = ""
            current_box = [0, 0, 0, 0]
            word_count = 0

            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = ocr_data['conf'][i]

                if text and confidence > 30:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]

                    bbox = [x, y, x + w, y + h]
                    normalized_bbox = self.normalize_bbox(bbox, (width, height))

                    words.append(text)
                    boxes.append(normalized_bbox)
                    confidences.append(confidence)
                    word_count += 1

            print(f"OCR extracted {len(words)} words")
            return image, words, boxes, confidences

        except Exception as e:
            print(f"Error in extract_text_and_boxes: {e}")
            if image_path.lower().endswith('.pdf'):
                images = convert_from_path(image_path)
                image = images[0]
            else:
                image = Image.open(image_path)
            return image, [], [], []

    def predict_entities(self, image_path):
        """Predict key-value entities using LayoutLMv3"""
        try:
            image, words, boxes, confidences = self.extract_text_and_boxes(image_path)

            if not words:
                print("No text detected in the image")
                return words, [], None, image.size

            print(f"Processing {len(words)} words with LayoutLMv3...")

            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            offset_mapping = encoding.pop('offset_mapping').cpu().numpy()
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)

            predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
            token_labels = [self.model.config.id2label[pred] for pred in predictions]

            return words, token_labels, offset_mapping, image.size

        except Exception as e:
            print(f"Error in predict_entities: {e}")
            return [], [], None, (0, 0)

    def extract_spatial_key_value_pairs(self, words, boxes, image_size):
        """Extract key-value pairs based on spatial relationships"""
        key_value_pairs = {}

        # Group words into lines based on y-coordinates
        lines = defaultdict(list)
        for i, (word, box) in enumerate(zip(words, boxes)):
            y_center = (box[1] + box[3]) / 2
            line_key = round(y_center / 10)  # Group by y-position
            lines[line_key].append((word, box, i))

        # Process each line
        for line_key in sorted(lines.keys()):
            line_items = sorted(lines[line_key], key=lambda x: x[1][0])  # Sort by x-position
            line_text = " ".join([item[0] for item in line_items])

            # Try to split line into key-value pairs
            kv_pairs = self.extract_kv_from_line(line_text, line_items)
            key_value_pairs.update(kv_pairs)

        return key_value_pairs

    def extract_kv_from_line(self, line_text, line_items):
        """Extract key-value pairs from a single line"""
        kv_pairs = {}

        # Common separators
        separators = [':', '=', '-', '~', '—', '–', '->']

        for sep in separators:
            if sep in line_text:
                parts = line_text.split(sep, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()

                    if self.is_likely_key(key) and value:
                        kv_pairs[key] = value
                        break

        # If no separator found, try pattern matching
        if not kv_pairs:
            patterns = [
                (r'^([A-Za-z\s]+)\s+([A-Za-z0-9\$\.,]+)$', 1, 2),  # "Key Value"
                (r'^([A-Za-z]+\s+[A-Za-z]+)\s+([0-9\$\.,]+)$', 1, 2),  # "Multiple Words 123"
            ]

            for pattern, key_group, value_group in patterns:
                match = re.match(pattern, line_text)
                if match:
                    key = match.group(key_group).strip()
                    value = match.group(value_group).strip()
                    if self.is_likely_key(key):
                        kv_pairs[key] = value
                        break

        return kv_pairs

    def is_likely_key(self, text):
        """Check if text is likely to be a key (not a value)"""
        text_lower = text.lower().strip()

        # Check against known key patterns
        for key_type, patterns in self.key_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return True

        # Keys are usually words, not numbers or amounts
        if re.match(r'^[\$\€\£]?\d+[,.]?\d*$', text):  # Numbers/amounts
            return False

        # Keys usually don't contain only numbers
        if text.replace('.', '').replace(',', '').isdigit():
            return False

        # Keys are usually shorter (but not always)
        if len(text.split()) > 5:  # Too many words might be description
            return False

        return True

    def extract_using_model_predictions(self, words, token_labels):
        """Extract key-value pairs using model predictions"""
        key_value_pairs = {}

        current_key = ""
        current_value = ""
        current_entity = None

        for word, label in zip(words, token_labels[:len(words)]):
            if label == "B-KEY":
                # Save previous pair
                if current_key and current_value:
                    key_value_pairs[current_key.strip()] = current_value.strip()

                current_key = word
                current_value = ""
                current_entity = "KEY"

            elif label == "I-KEY" and current_entity == "KEY":
                current_key += " " + word

            elif label == "B-VALUE":
                # Save previous pair
                if current_key and current_value:
                    key_value_pairs[current_key.strip()] = current_value.strip()

                current_value = word
                current_entity = "VALUE"

            elif label == "I-VALUE" and current_entity == "VALUE":
                current_value += " " + word

            else:
                # Save current pair if we have both key and value
                if current_key and current_value:
                    key_value_pairs[current_key.strip()] = current_value.strip()
                    current_key = ""
                    current_value = ""
                    current_entity = None

        # Save final pair
        if current_key and current_value:
            key_value_pairs[current_key.strip()] = current_value.strip()

        return key_value_pairs

    def post_process_key_value_pairs(self, kv_pairs):
        """Clean and organize extracted key-value pairs"""
        processed_pairs = {}

        for key, value in kv_pairs.items():
            # Clean the key
            clean_key = key.strip(' :=-').title()

            # Clean the value
            clean_value = value.strip(' :=-')

            # Remove empty pairs
            if clean_key and clean_value and clean_value not in ['', 'N/A', 'None']:
                processed_pairs[clean_key] = clean_value

        return processed_pairs

    def extract_all_key_value_pairs(self, image_path):
        """Extract ALL key-value pairs from invoice using multiple methods"""
        print("Extracting key-value pairs...")

        # Method 1: Using model predictions
        words, token_labels, offset_mapping, image_size = self.predict_entities(image_path)

        if not words:
            return {"error": "No text detected in the document"}

        # Extract using model
        model_kv_pairs = self.extract_using_model_predictions(words, token_labels)

        # Method 2: Extract text and boxes for spatial analysis
        image, words_spatial, boxes, confidences = self.extract_text_and_boxes(image_path)

        if words_spatial:
            # Method 2: Spatial analysis
            spatial_kv_pairs = self.extract_spatial_key_value_pairs(words_spatial, boxes, image_size)

            # Method 3: Pattern-based extraction from full text
            full_text = " ".join(words_spatial)
            pattern_kv_pairs = self.pattern_based_extraction(full_text)

            # Combine all methods
            all_kv_pairs = {}
            all_kv_pairs.update(pattern_kv_pairs)
            all_kv_pairs.update(spatial_kv_pairs)
            all_kv_pairs.update(model_kv_pairs)

        else:
            all_kv_pairs = model_kv_pairs

        # Post-process and organize
        processed_pairs = self.post_process_key_value_pairs(all_kv_pairs)

        print(f"Extracted {len(processed_pairs)} key-value pairs")
        return processed_pairs

    def pattern_based_extraction(self, text):
        """Extract key-value pairs using pattern matching"""
        kv_pairs = {}

        # Common patterns for invoice data
        patterns = {
            'invoice_number': [
                r'invoice\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)',
                r'inv\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)',
                r'bill\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)'
            ],
            'date': [
                r'date\s*:?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})',
                r'invoice\s+date\s*:?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})'
            ],
            'due_date': [
                r'due\s+date\s*:?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})',
                r'payment\s+due\s*:?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})'
            ],
            'total_amount': [
                r'total\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'amount\s+due\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'balance\s+due\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'grand\s+total\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})'
            ],
            'subtotal': [
                r'subtotal\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'sub\s+total\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})'
            ],
            'tax': [
                r'tax\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'vat\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'gst\s*:?\s*[\$€£]?\s*(\d{1,3}(?:,\d{3})*\.?\d{0,2})'
            ]
        }

        for field, regex_list in patterns.items():
            for pattern in regex_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if field == 'total_amount' or field == 'subtotal' or field == 'tax':
                        value = f"${value}"
                    kv_pairs[field.replace('_', ' ').title()] = value
                    break

        return kv_pairs

# Initialize the advanced extractor
extractor = AdvancedInvoiceExtractor(model, processor)

# Step 5: Test Functions and User Interface
def create_detailed_sample_invoice():
    """Create a detailed sample invoice with multiple key-value pairs"""
    img = Image.new('RGB', (1200, 1000), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial", size=18)
        font_bold = ImageFont.truetype("Arial", size=20)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    # Invoice content with various key-value formats
    content = [
        ("INVOICE", 100, 50, font_bold),
        (" ", 100, 80, font),
        ("Invoice No: INV-2024-001", 100, 120, font),
        ("Date: 2024-01-15", 100, 150, font),
        ("Due Date: 2024-02-15", 100, 180, font),
        (" ", 100, 210, font),
        ("From: ABC Company Inc.", 100, 240, font),
        ("Address: 123 Business Street", 100, 270, font),
        ("City: New York, NY 10001", 100, 300, font),
        ("Phone: (555) 123-4567", 100, 330, font),
        ("Email: info@abccompany.com", 100, 360, font),
        (" ", 100, 390, font),
        ("Bill To: XYZ Corporation", 100, 420, font),
        ("Client ID: CL-789", 100, 450, font),
        ("Contact: John Smith", 100, 480, font),
        (" ", 100, 510, font),
        ("Description: Web Development Services", 100, 540, font),
        ("Hours: 40", 100, 570, font),
        ("Rate: $75.00", 100, 600, font),
        (" ", 100, 630, font),
        ("Subtotal: $3,000.00", 100, 660, font),
        ("Tax Rate: 10%", 100, 690, font),
        ("Tax Amount: $300.00", 100, 720, font),
        ("Total Amount: $3,300.00", 100, 750, font),
        (" ", 100, 780, font),
        ("Payment Terms: Net 30", 100, 810, font),
        ("Status: Paid", 100, 840, font),
        ("Notes: Thank you for your business!", 100, 870, font)
    ]

    for text, x, y, font_style in content:
        draw.text((x, y), text, fill='black', font=font_style)

    img.save("detailed_invoice.png")
    print("Detailed sample invoice created: detailed_invoice.png")
    return "detailed_invoice.png"

def process_invoice_file(file_path):
    """Process uploaded invoice file and extract ALL key-value pairs"""
    print(f"Processing file: {file_path}")
    print("Extracting all key-value pairs...")

    try:
        # Extract ALL key-value pairs
        extracted_data = extractor.extract_all_key_value_pairs(file_path)

        # Convert to JSON format
        json_output = json.dumps(extracted_data, indent=2)

        return extracted_data, json_output

    except Exception as e:
        print(f"Error processing file: {e}")
        return {"error": str(e)}, json.dumps({"error": str(e)}, indent=2)

def display_comprehensive_results(extracted_data):
    """Display comprehensive results of all extracted key-value pairs"""
    print("\n" + "="*70)
    print("COMPREHENSIVE KEY-VALUE EXTRACTION RESULTS")
    print("="*70)

    if "error" in extracted_data:
        print(f"❌ Error: {extracted_data['error']}")
        return

    if not extracted_data:
        print("❌ No key-value pairs extracted")
        return

    print(f"✅ Successfully extracted {len(extracted_data)} key-value pairs:\n")

    for key, value in extracted_data.items():
        print(f"🔑 {key:25}: {value}")

    print(f"\n📊 Total pairs extracted: {len(extracted_data)}")

def main():
    """Main function to run the comprehensive invoice extraction system"""
    print("=== COMPREHENSIVE INVOICE KEY-VALUE EXTRACTION SYSTEM ===")
    print("This system extracts ALL key-value pairs from invoices")
    print("="*60)
    print("Choose an option:")
    print("1. Test with detailed sample invoice")
    print("2. Upload your own invoice file")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        sample_path = create_detailed_sample_invoice()
        result, json_output = process_invoice_file(sample_path)

    elif choice == "2":
        print("Please upload an invoice image or PDF file")
        uploaded = files.upload()

        if uploaded:
            file_name = list(uploaded.keys())[0]
            result, json_output = process_invoice_file(file_name)
        else:
            print("No file uploaded! Using sample instead.")
            sample_path = create_detailed_sample_invoice()
            result, json_output = process_invoice_file(sample_path)
    else:
        print("Invalid choice! Using sample invoice.")
        sample_path = create_detailed_sample_invoice()
        result, json_output = process_invoice_file(sample_path)

    # Display JSON output
    print("\n" + "="*70)
    print("JSON OUTPUT")
    print("="*70)
    print(json_output)

    # Display comprehensive results
    display_comprehensive_results(result)

    # Save results
    output_filename = "all_extracted_key_values.json"
    with open(output_filename, 'w') as f:
        f.write(json_output)

    print(f"\n💾 Results saved to: {output_filename}")

    return result

# Run the system
print("Initializing Comprehensive Invoice Key-Value Extraction System...")
print("Device:", extractor.device)

# Test the comprehensive system
final_result = main()

if final_result and "error" not in final_result:
    print("\n🎉 Comprehensive key-value extraction completed successfully!")
    print(f"📈 Extracted {len(final_result)} key-value pairs in total!")
else:
    print("\n⚠️  Extraction completed with some issues.")



