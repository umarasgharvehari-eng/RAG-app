import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
import math

st.set_page_config(page_title="RAG Document QA App", layout="wide")

# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# HELPERS
# -----------------------------
def extract_text(file_name, file_bytes):
    file_name = file_name.lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    elif file_name.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif file_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        return ""


def chunk_text(text, chunk_size=1000, overlap=200):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def index_document(file_name, file_bytes):
    text = extract_text(file_name, file_bytes)

    if not text.strip():
        return None, "Document se text extract nahi hua. Agar scanned PDF hai to OCR chahiye hoga."

    chunks = chunk_text(text)

    if not chunks:
        return None, "Document me usable text chunks nahi bane."

    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        embedded_chunks.append({
            "chunk_id": i,
            "text": chunk,
            "embedding": emb
        })

    return {
        "file_name": file_name,
        "chunks": embedded_chunks
    }, None


def retrieve_relevant_chunks(question, indexed_doc, top_k=4):
    query_embedding = get_embedding(question)

    scored = []
    for item in indexed_doc["chunks"]:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_answer(question, relevant_chunks):
    context = "\n\n---\n\n".join(
        [f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in relevant_chunks]
    )

    prompt = f"""
You are a helpful document question-answering assistant.

Answer ONLY from the provided context.
If the answer is not clearly present in the context, say:
"I could not find that in the uploaded document."

Also mention the chunk numbers you used when relevant.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You answer questions only from retrieved document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# SESSION STATE
# -----------------------------
if "indexed_doc" not in st.session_state:
    st.session_state.indexed_doc = None


# -----------------------------
# UI
# -----------------------------
st.title("📄 RAG Document QA App")
st.write("Upload PDF, DOCX, ya TXT file aur us par questions poocho.")

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Document process ho raha hai..."):
            file_bytes = uploaded_file.read()
            indexed_doc, error = index_document(uploaded_file.name, file_bytes)

            if error:
                st.error(error)
            else:
                st.session_state.indexed_doc = indexed_doc
                st.success(
                    f"{indexed_doc['file_name']} successfully indexed. "
                    f"Total chunks: {len(indexed_doc['chunks'])}"
                )

if st.session_state.indexed_doc is not None:
    st.subheader("Ask a question")
    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Pehle question likho.")
        else:
            with st.spinner("Answer generate ho raha hai..."):
                relevant_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.indexed_doc,
                    top_k=4
                )

                answer = generate_answer(question, relevant_chunks)

                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Retrieved Chunks"):
                    for chunk in relevant_chunks:
                        st.markdown(
                            f"**Chunk {chunk['chunk_id']}** "
                            f"(score: {chunk['score']:.4f})"
                        )
                        st.write(chunk["text"])
                        st.markdown("---")
