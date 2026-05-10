import os
import uuid
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env file")

groq_client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# Store a collection per session
sessions = {}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload a PDF, chunk it, embed, and store in ChromaDB."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    session_id = str(uuid.uuid4())

    try:
        pdf_bytes = file.read()
        text = extract_text_from_pdf(pdf_bytes)

        if not text:
            return jsonify({"error": "Could not extract text from PDF"}), 400

        chunks = chunk_text(text)

        if not chunks:
            return jsonify({"error": "Document appears to be empty"}), 400

        # Create a fresh collection for this session
        collection = chroma_client.create_collection(name=session_id)

        embeddings = embedder.encode(chunks).tolist()

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

        sessions[session_id] = collection

        return jsonify({
            "session_id": session_id,
            "chunks": len(chunks),
            "filename": file.filename,
            "message": f"Processed {len(chunks)} chunks from {file.filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """Query the uploaded document."""
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("question", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or expired session. Please re-upload your document."}), 400

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        collection = sessions[session_id]

        # Embed the question and retrieve top 5 relevant chunks
        q_embedding = embedder.encode([question]).tolist()
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=min(5, collection.count()),
        )

        context_chunks = results["documents"][0]
        context = "\n\n---\n\n".join(context_chunks)

        # Ask Groq/LLaMA with the retrieved context
        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on the provided document context. "
            "If the answer is not in the context, say so clearly. Do not make things up. "
            "Be concise and direct."
        )

        user_message = f"""Context from the document:

{context}

---

Question: {question}

Answer based only on the context above:"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({
            "answer": answer,
            "sources": len(context_chunks),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)
