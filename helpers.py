import logging
from fpdf import FPDF
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.extend(page.extract_text().splitlines())
    return [line.strip() for line in text if line.strip()]


def txt_to_pdf(txt_file, pdf_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)

    # Open the text file and add its content to the PDF
    with open(txt_file, 'r') as file:
        for line in file:
            pdf.multi_cell(120, 10, txt=line.strip())

    # Save the PDF
    pdf.output(pdf_file)
    logger.info(f'PDF saved as: {pdf_file}')