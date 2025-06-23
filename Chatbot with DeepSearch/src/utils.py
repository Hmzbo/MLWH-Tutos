import os
import requests
import base64
from pathlib import Path
from mistralai import Mistral
from PyPDF2 import PdfReader, PdfWriter


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Get the project root and then the data directory in one compact line
temp_directory = Path(__file__).parent.parent / "temp"
# Ensure the 'temp/' directory exists
os.makedirs(temp_directory, exist_ok=True)


def download_pdf(pdf_url, download_path):
  
    response = requests.get(pdf_url)
    with open(download_path, "wb") as f:
        f.write(response.content)


# Extract only the first N pages
def extract_pages(input_path, output_path, page_count=3):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    for i in range(min(page_count, len(reader.pages))):
        writer.add_page(reader.pages[i])
    with open(output_path, "wb") as f:
        writer.write(f)


def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None
    
def pdf_ocr(url, page_count=5):

    # Construct the full path for PDF file to download
    download_path = temp_directory / "original.pdf"

    # Download the PDF from URL
    original_pdf_path = download_pdf(url, download_path)

    # Construct the full path for the output PDF file
    shortened_pdf_path = temp_directory / "shortened.pdf"

    extract_pages(original_pdf_path, shortened_pdf_path, page_count=page_count)

    # Getting the base64 string
    base64_pdf = encode_pdf(shortened_pdf_path)

    client = Mistral(api_key=MISTRAL_API_KEY)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}" 
        },
        include_image_base64=True
    )
    return ocr_response