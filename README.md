# DocMind — RAG Document Q&A

Upload a PDF, ask questions in plain English, get grounded answers. No hallucination, the model only answers from your document.

Built with Python, Flask, ChromaDB, Sentence Transformers, and LLaMA 3.3 70B via Groq.

---

## How it works

1. **Ingest** — your PDF is extracted, split into overlapping chunks, and each chunk is embedded into a vector using Sentence Transformers (`all-MiniLM-L6-v2`)
2. **Retrieve** — when you ask a question, it's embedded the same way and the 5 most semantically similar chunks are retrieved from ChromaDB
3. **Generate** — those chunks are passed as context to LLaMA 3.3 70B (via Groq), which answers strictly from the document

This is a Retrieval-Augmented Generation (RAG) pipeline, grounding the LLM in your actual documents rather than its training data.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| LLM | LLaMA 3.3 70B via Groq API |
| PDF parsing | PyMuPDF |
| Frontend | Vanilla HTML/CSS/JS |

---

## Getting started

### Prerequisites

- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/YOURUSERNAME/docmind-rag.git
cd docmind-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root of the project:

```
GROQ_API_KEY=your_groq_api_key_here
```

Never commit this file — it's already in `.gitignore`.

### Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. Drop a text-based PDF onto the upload zone (or click to browse)
2. Wait for indexing — you'll see how many chunks were processed
3. Ask any question about the document
4. Use the suggested questions or type your own
5. Click **New doc** to start over with a different file

> **Note:** Only text-based PDFs are supported. Scanned or image-based PDFs where you cannot highlight text will not work without an OCR layer.

---

## Project structure

```
docmind-rag/
├── app.py              # Flask backend — upload, chunking, embedding, querying
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Frontend UI
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## Author

Built by Prachi Bhatt as part of a portfolio of applied AI projects.
