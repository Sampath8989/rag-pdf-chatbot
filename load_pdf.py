import sys
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    print(f"Loading {file_path}...")
    pdf_text = ""
    
    # 1. Read the PDF using PyPDF2
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
            
    # 2. Chunk the text
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(pdf_text)
    
    # 3. Print chunks to terminal to verify
    print(f"Successfully split the PDF into {len(chunks)} chunks!")
    print("\n--- Here is a sample of Chunk #1 ---")
    print(chunks[0])
    print("------------------------------------\n")
    
    return chunks

if __name__ == "__main__":
    # Check if the user actually typed a file name in the terminal
    if len(sys.argv) < 2:
        print("Error: You forgot to provide a PDF file!")
        print("Usage: python3 load_pdf.py <name_of_pdf_file.pdf>")
        sys.exit(1)
        
    # Get the file name from the terminal command (argv[1] is the first word after the script name)
    target_pdf = sys.argv[1]
    
    # Run the function with that file
    process_pdf(target_pdf)