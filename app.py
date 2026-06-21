from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Pinecone API Key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone Index
index_name = "medical-chatbot"

# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

# Ollama LLM (Local)
chatModel = ChatOllama(
    model="llama3",
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# RAG Chain
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Home Page
@app.route("/")
def index():
    return render_template("chat.html")

# Chat Endpoint
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]

    print("User:", msg)

    response = rag_chain.invoke(
        {"input": msg}
    )

    answer = response["answer"]

    print("Bot:", answer)

    return str(answer)

# Run Flask App
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )