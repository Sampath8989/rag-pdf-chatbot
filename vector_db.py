import sys
import os
import shutil
from load_pdf import process_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# We will save the database in a folder called 'chroma_db'
CHROMA_PATH = "chroma_db"

def create_vector_db(pdf_path):
    # EDGE CASE 1: Did the PDF load correctly?
    try:
        chunks = process_pdf(pdf_path)
    except Exception as e:
        print(f" Error reading PDF: {e}")
        sys.exit(1)

    if not chunks:
        print("No text chunks found. Is the PDF empty?")
        sys.exit(1)

    # Initialize Embeddings (Downloads a free, local model from HuggingFace)
    print("\n⏳ Loading embedding model (this may take a minute the first time)...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # EDGE CASE 2: Prevent Duplicate Data
    # If we run this script twice, it will add the same chunks twice. 
    # We delete the old database folder before making a new one to keep it clean.
    if os.path.exists(CHROMA_PATH):
        print(" Clearing old database to prevent duplicates...")
        shutil.rmtree(CHROMA_PATH)

    # Create and store in ChromaDB
    print(f" Storing {len(chunks)} chunks in ChromaDB...")
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    print(" Database created successfully!")
    return db

def test_query(db, query):
    print(f"\n🔍 Testing Query: '{query}'")
    # k=3 means we want the top 3 most relevant chunks
    results = db.similarity_search(query, k=3)
    
    print(f" Top {len(results)} Results:\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content)
        print("-" * 20)

if __name__ == "__main__":
    # EDGE CASE 3: Did the user forget the file name?
    if len(sys.argv) < 2:
        print(" Error: You forgot to provide a PDF file!")
        print(" Usage: python3 vector_db.py <name_of_pdf_file.pdf>")
        sys.exit(1)

    target_pdf = sys.argv[1]

    # EDGE CASE 4: Does the file actually exist in the folder?
    if not os.path.exists(target_pdf):
        print(f" Error: The file '{target_pdf}' does not exist in this folder.")
        sys.exit(1)

    # 1. Run the pipeline
    database = create_vector_db(target_pdf)

    # 2. Test it to make sure retrieval works!
    test_query(database, "What are the tasks for Day 2?")