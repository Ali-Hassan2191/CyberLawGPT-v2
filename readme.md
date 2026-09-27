# ⚖️ CyberLawGPT

**CyberLawGPT** is a free-to-deploy Retrieval-Augmented Generation (RAG) application for asking questions about Pakistani cyber law.

It uses:

- **Python**
- **Streamlit** for the interactive UI
- **FAISS** for vector similarity search
- **Sentence Transformers** for free local embeddings
- **Groq API**
- **`openai/gpt-oss-120b`** for answer generation
- **PyPDF** for PDF extraction

## 1. What the app does

On startup, CyberLawGPT:

1. Downloads the cyber-law PDF from the supplied Google Drive file.
2. Falls back to the public Pakistan Code copy if the Drive download is unavailable.
3. Extracts text page-by-page.
4. Splits the text into overlapping chunks.
5. Creates embeddings locally using:
   `sentence-transformers/all-MiniLM-L6-v2`
6. Stores normalized vectors in a FAISS `IndexFlatIP` index.
7. When the user asks a question, retrieves the most relevant chunks.
8. Sends only the retrieved legal context plus the question to Groq.
9. Generates a source-grounded answer with `openai/gpt-oss-120b`.
10. Shows the retrieved source chunks and PDF page numbers in the UI.

The FAISS index is cached locally using a hash of the downloaded PDF, so it is rebuilt only when the source PDF changes.

## 2. UI features

CyberLawGPT includes:

- Technicality Level:
  - Simple
  - Intermediate
  - Technical
- Response Size:
  - Short
  - Medium
  - Detailed
- Legal Focus
- Response Language:
  - English
  - English + Urdu
  - Urdu
- Reasoning Effort:
  - low
  - medium
  - high
- Retrieved Sources slider
- Sample **Try Asking** questions
- Chat interface
- Source/retrieval viewer
- Legal Information Notice
- Cyber Safety notice
- Clear Chat button

## 3. Files

Only three project files are required:

```text
CyberLawGPT/
├── app.py
├── requirements.txt
└── readme.md
```

The application automatically creates a `.cyberlaw_cache/` directory at runtime.

## 4. Groq API key

Create a Groq API key and set it as:

```text
GROQ_API_KEY
```

### Google Colab

Run:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Then install:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

If using Colab, expose the Streamlit port with your preferred tunnel method.

### Streamlit Cloud

1. Upload `app.py`, `requirements.txt`, and `readme.md` to GitHub.
2. Create a Streamlit Cloud app from the repository.
3. Open the app's **Secrets** settings.
4. Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

5. Deploy.

Do **not** put the API key directly inside `app.py`.

## 5. Important Google Drive requirement

The app uses this file ID:

```text
1Alve7SH7pEtCyK3o-9B_uem7ATGQ8Dda
```

For Streamlit Cloud/Colab to download it automatically, the Google Drive file must be accessible to the deployment.

If the Drive file cannot be downloaded, the app attempts to use the public Pakistan Code PDF as a fallback.

## 6. Legal grounding

The application is designed to answer from the supplied cyber-law document rather than relying on the model's general memory.

The prompt explicitly tells the model:

- Do not invent legal sections.
- Do not invent penalties.
- Do not fabricate citations.
- Say when the retrieved document does not contain enough information.
- Cite relevant sections/pages when available.
- Keep the answer informational rather than presenting it as professional legal advice.

This is important because an LLM should not be treated as the legal authority itself.

## 7. RAG architecture

```text
                 Cyber-law PDF
                       │
                       ▼
                 PDF extraction
                       │
                       ▼
              Text normalization
                       │
                       ▼
            Chunking + page metadata
                       │
                       ▼
       Sentence Transformer embeddings
                       │
                       ▼
                 FAISS index
                       │
             User's question
                       │
                       ▼
              Query embedding
                       │
                       ▼
            Similarity retrieval
                       │
                       ▼
          Relevant legal context
                       │
                       ▼
            Groq GPT-OSS 120B
                       │
                       ▼
             Grounded legal answer
                       │
                       ▼
            Sources / PDF pages
```

## 8. Security notes

- Keep `GROQ_API_KEY` in environment variables or Streamlit Secrets.
- Do not commit secrets to GitHub.
- Do not ask users to submit passwords, private keys, CNIC numbers, financial information, or other sensitive data.
- The app is intended for educational/legal-information use.
- For an actual legal dispute or urgent legal matter, consult a qualified Pakistani lawyer or relevant official authority.

## 9. Changing the source PDF

The Drive file ID is near the top of `app.py`:

```python
DRIVE_FILE_ID = "1Alve7SH7pEtCyK3o-9B_uem7ATGQ8Dda"
```

To use another PDF, replace the ID.

The app hashes the downloaded PDF. A changed PDF automatically receives a new FAISS cache/index.

## 10. Notes about the model

The application uses:

```text
openai/gpt-oss-120b
```

through Groq.

The app deliberately uses a low temperature and supplies retrieved legal context to reduce unsupported answers.

## 11. Deployment goal

The project is intentionally lightweight enough for:

- Google Colab development
- Streamlit Cloud deployment
- Free/local embedding generation

The Groq API is the external model service, while embeddings and FAISS retrieval run in the application environment.
