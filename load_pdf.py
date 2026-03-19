import PyPDF2

def process_pdf(file_path):
    print(f"Loading {file_path}...")
    pdf_text = ""
    
    # Read the PDF using PyPDF2
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
            
    print(f"Successfully extracted {len(pdf_text)} characters from the PDF.")
    return pdf_text

if __name__ == "__main__":
    # We will test this using the PDF that is already sitting in the folder
    process_pdf("GenAI_14Day_Portfolio_Plan.pdf")