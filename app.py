import streamlit as st
import tempfile
import os
from io import BytesIO

from pypdf import PdfReader
from docx import Document

import chromadb
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Chroma DB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="docs")

# -----------------------------
# FILE PARSING
# -----------------------------
def extract_text(file_name, file_bytes):
    text = ""

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""

    elif file_name.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        for p in doc.paragraphs:
            text += p.text + "\n"

    elif file_name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")

    return text


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# -----------------------------
# EMBEDDINGS
# -----------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# -----------------------------
# INGEST DOCUMENT
# -----------------------------
def ingest_document(file_name, file_bytes):
    text = extract_text(file_name, file_bytes)

    if not text.strip():
        st.error("No text found in document (maybe scanned PDF?)")
        return

    chunks = chunk_text(text)

    ids = []
    embeddings = []
    documents = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{file_name}_{i}")
        embeddings.append(get_embedding(chunk))
        documents.append(chunk)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents
    )

    st.success(f"Document indexed with {len(chunks)} chunks ✅")


# -----------------------------
# ASK QUESTION
# -----------------------------
def ask_question(question):
    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    retrieved_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.
If answer is not present, say:
"I could not find that in the document."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You answer using document context only."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="RAG App", layout="wide")

st.title("📄 RAG Document QA App")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    if st.button("Process Document"):
        ingest_document(uploaded_file.name, file_bytes)

st.divider()

question = st.text_input("Ask a question about your document:")

if st.button("Get Answer"):
    if question:
        answer = ask_question(question)
        st.write("### Answer:")
        st.write(answer)
