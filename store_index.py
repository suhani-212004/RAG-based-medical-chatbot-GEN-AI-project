from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("Step 1: Loading PDF files...")
extracted_data = load_pdf_file(data='data/')
print("PDF files loaded successfully.")
print("Total pages loaded:", len(extracted_data))

print("Step 2: Filtering documents...")
filter_data = filter_to_minimal_docs(extracted_data)
print("Filtering completed.")

print("Step 3: Splitting text into chunks...")
text_chunks = text_split(filter_data)
print("Text splitting completed.")
print("Total chunks created:", len(text_chunks))

print("Step 4: Loading Hugging Face embedding model...")
embeddings = download_hugging_face_embeddings()
print("Embedding model loaded successfully.")

pinecone_api_key = PINECONE_API_KEY

print("Step 5: Connecting to Pinecone...")
pc = Pinecone(api_key=pinecone_api_key)
print("Connected to Pinecone.")

index_name = "medical-chatbot"

print("Step 6: Checking Pinecone index...")

if not pc.has_index(index_name):
    print("Index does not exist. Creating index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
    print("Index created successfully.")
else:
    print("Index already exists.")

index = pc.Index(index_name)

print("Step 7: Uploading embeddings to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Step 8: Upload completed successfully!")
print("Vector database is ready.")