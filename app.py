import os
import re
import hashlib
from io import BytesIO

import faiss
import numpy as np
import requests
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CyberLawGPT
# RAG assistant for Pakistan's cyber-law document
# ============================================================

st.set_page_config(
    page_title="CyberLawGPT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "CyberLawGPT"
MODEL_NAME = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The supplied Google Drive file ID.
DRIVE_FILE_ID = "1Alve7SH7pEtCyK3o-9B_uem7ATGQ8Dda"

# Google Drive's downloadable endpoint. If the file is private,
# the owner must make it accessible to the deployment.
PDF_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# Official Pakistan Code fallback/current public copy.
OFFICIAL_PDF_URL = (
    "https://www.pakistancode.gov.pk/"
    "pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
)

CACHE_DIR = ".cyberlaw_cache"
PDF_PATH = os.path.join(CACHE_DIR, "cyber_laws.pdf")
INDEX_PATH = os.path.join(CACHE_DIR, "cyber_laws.faiss")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.npy")
META_PATH = os.path.join(CACHE_DIR, "metadata.npy")


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-top: 0.25rem;
    }
    .notice {
        padding: 14px 18px;
        border-radius: 12px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        margin: 12px 0 18px 0;
    }
    .safety {
        padding: 14px 18px;
        border-radius: 12px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        margin: 10px 0 20px 0;
    }
    .source-card {
        border-left: 4px solid #6366f1;
        padding: 8px 12px;
        margin: 8px 0;
        background: #f8fafc;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Download + preprocessing
# -----------------------------
def download_pdf():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 10_000:
        return PDF_PATH, "Cached PDF"

    urls = [PDF_URL, OFFICIAL_PDF_URL]
    errors = []

    for url in urls:
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "CyberLawGPT/1.0"},
                allow_redirects=True,
            )
            response.raise_for_status()
            content = response.content

            # Basic PDF validation.
            if not content.startswith(b"%PDF"):
                errors.append(f"Downloaded content was not a PDF from {url}")
                continue

            with open(PDF_PATH, "wb") as f:
                f.write(content)

            return PDF_PATH, url
        except Exception as e:
            errors.append(str(e))

    raise RuntimeError(
        "Could not download the cyber-law PDF. "
        "Make sure the Google Drive file is accessible, or check internet access. "
        + " | ".join(errors)
    )


def normalize_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=1100, overlap=180):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def extract_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    all_chunks = []
    metadata = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = normalize_text(raw)

        if not text:
            continue

        for chunk_number, chunk in enumerate(chunk_text(text), start=1):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "page": page_number,
                    "chunk": chunk_number,
                    "source": "Pakistan cyber-law PDF",
                }
            )

    if not all_chunks:
        raise RuntimeError(
            "No readable text was extracted from the PDF. "
            "If the PDF is scanned/image-only, OCR is required."
        )

    return all_chunks, metadata


# -----------------------------
# RAG resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_rag():
    os.makedirs(CACHE_DIR, exist_ok=True)

    pdf_path, pdf_source = download_pdf()

    # Build a fingerprint so changing the PDF rebuilds the index.
    with open(pdf_path, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()[:16]

    index_file = os.path.join(CACHE_DIR, f"index_{fingerprint}.faiss")
    chunks_file = os.path.join(CACHE_DIR, f"chunks_{fingerprint}.npy")
    meta_file = os.path.join(CACHE_DIR, f"meta_{fingerprint}.npy")

    embedding_model = SentenceTransformer(EMBED_MODEL)

    if (
        os.path.exists(index_file)
        and os.path.exists(chunks_file)
        and os.path.exists(meta_file)
    ):
        index = faiss.read_index(index_file)
        chunks = np.load(chunks_file, allow_pickle=True).tolist()
        metadata = np.load(meta_file, allow_pickle=True).tolist()
        return embedding_model, index, chunks, metadata, pdf_source

    chunks, metadata = extract_chunks(pdf_path)

    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_file)
    np.save(chunks_file, np.array(chunks, dtype=object))
    np.save(meta_file, np.array(metadata, dtype=object))

    return embedding_model, index, chunks, metadata, pdf_source


def retrieve(query, embedding_model, index, chunks, metadata, top_k=6):
    query_vector = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "text": chunks[idx],
                "score": float(score),
                "page": metadata[idx]["page"],
                "chunk": metadata[idx]["chunk"],
                "source": metadata[idx]["source"],
            }
        )

    return results


def build_context(results):
    blocks = []

    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {i} | Page {item['page']} | Chunk {item['chunk']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(blocks)


# -----------------------------
# Groq answer generation
# -----------------------------
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Colab environment variables "
            "or Streamlit Cloud Secrets."
        )

    return Groq(api_key=api_key)


def generate_answer(
    question,
    context,
    technicality,
    response_size,
    legal_focus,
    language,
    reasoning_effort,
):
    client = get_groq_client()

    technicality_text = {
        "Simple": "Use plain, beginner-friendly language and explain legal terms briefly.",
        "Intermediate": "Use clear legal terminology but explain difficult terms.",
        "Technical": "Use precise legal and technical terminology and discuss relevant legal concepts in depth.",
    }[technicality]

    size_text = {
        "Short": "Give a concise answer with only the important points.",
        "Medium": "Give a balanced answer with the important details and practical explanation.",
        "Detailed": "Give a thorough answer, organized with headings, bullets, and relevant legal distinctions.",
    }[response_size]

    focus_text = legal_focus.strip() or "General cyber law in Pakistan"

    system_prompt = f"""
You are CyberLawGPT, a retrieval-augmented legal information assistant
focused on Pakistani cyber law.

IMPORTANT LEGAL GROUNDING RULES:
1. Answer primarily from the supplied Pakistan cyber-law document in the context.
2. Do not invent sections, penalties, procedures, definitions, authorities,
   exceptions, or legal conclusions.
3. If the retrieved context does not contain enough information, say that the
   supplied document does not provide enough information to answer confidently.
4. Clearly distinguish what the law says from explanation or practical guidance.
5. When possible, cite the relevant section number and page from the provided source.
6. If a question is unrelated to Pakistani cyber law, politely say it is outside
   the scope of this legal knowledge base.
7. Do not claim to be a lawyer and do not present the response as a substitute
   for professional legal advice.
8. For potentially serious legal matters, recommend consulting a qualified
   Pakistani lawyer or relevant official authority.
9. Never fabricate a citation.
10. The source document is the authority for this RAG answer. Do not use
    unstated legal assumptions as if they were in the document.

USER PREFERENCES:
Technicality: {technicality}
Response size: {response_size}
Legal focus: {focus_text}
Language: {language}

STYLE:
{technicality_text}
{size_text}

ANSWER FORMAT:
- Start with a direct answer.
- Then give the relevant legal basis.
- Add a practical explanation when useful.
- End with "Source(s) used" and list the source pages/sections actually used.
"""

    user_prompt = f"""
Question:
{question}

Retrieved legal context:
{context}

Answer the question using only the retrieved context and the rules above.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens={
            "Short": 700,
            "Medium": 1300,
            "Detailed": 2200,
        }[response_size],
        reasoning_effort=reasoning_effort,
    )

    return completion.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="main-title">⚖️ CyberLawGPT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Pakistan Cyber Law • RAG-powered legal information assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <b>⚠️ Legal Information Notice</b><br>
        CyberLawGPT provides AI-generated legal information grounded in the
        selected Pakistani cyber-law document. It is for educational and
        informational purposes and is not a substitute for advice from a
        qualified lawyer or an official legal authority.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="safety">
        <b>🛡️ Cyber Safety</b><br>
        Do not share passwords, private keys, CNIC numbers, financial details,
        private messages, or other sensitive personal information in your questions.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Answer Settings")

    technicality = st.select_slider(
        "Technicality Level",
        options=["Simple", "Intermediate", "Technical"],
        value="Intermediate",
    )

    response_size = st.select_slider(
        "Response Size",
        options=["Short", "Medium", "Detailed"],
        value="Medium",
    )

    legal_focus = st.text_input(
        "Legal Focus",
        placeholder="e.g. cyber stalking, identity theft, online fraud",
        help="Optional topic to focus retrieval and explanation around.",
    )

    language = st.selectbox(
        "Response Language",
        ["English", "English + Urdu", "Urdu"],
        index=0,
    )

    reasoning_effort = st.selectbox(
        "Reasoning Effort",
        ["low", "medium", "high"],
        index=1,
        help="Higher reasoning can improve difficult legal questions but may use more tokens.",
    )

    top_k = st.slider(
        "Retrieved Sources",
        min_value=3,
        max_value=10,
        value=6,
        help="Number of PDF chunks supplied to the model.",
    )

    st.divider()

    st.caption("Model")
    st.code(MODEL_NAME)

    st.caption("Embeddings")
    st.code(EMBED_MODEL)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize resources.
with st.spinner("Loading cyber-law PDF and preparing FAISS knowledge base..."):
    try:
        embedding_model, index, chunks, metadata, pdf_source = load_rag()
    except Exception as e:
        st.error(str(e))
        st.stop()

st.success(
    f"Knowledge base ready • {len(chunks):,} chunks indexed • "
    f"Source: {pdf_source if pdf_source != 'Cached PDF' else 'cached PDF'}"
)

st.subheader("💬 Ask a Cyber-Law Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

sample_questions = [
    "What does Pakistani law say about cyber stalking?",
    "What is unauthorized use of identity information under PECA?",
    "What does the law say about electronic fraud?",
    "What is the legal position on online harassment?",
    "What are the rules concerning malicious code?",
    "What does PECA say about spoofing?",
]

st.caption("Try Asking")
sample_cols = st.columns(3)

for i, sample in enumerate(sample_questions):
    if sample_cols[i % 3].button(sample, key=f"sample_{i}", use_container_width=True):
        st.session_state.pending_question = sample

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask about a Pakistani cyber-law issue..."
)

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the cyber-law knowledge base..."):
            retrieval_query = question
            if legal_focus.strip():
                retrieval_query = f"{legal_focus}: {question}"

            results = retrieve(
                retrieval_query,
                embedding_model,
                index,
                chunks,
                metadata,
                top_k=top_k,
            )

            context = build_context(results)

        with st.spinner("Generating grounded legal information..."):
            try:
                answer = generate_answer(
                    question=question,
                    context=context,
                    technicality=technicality,
                    response_size=response_size,
                    legal_focus=legal_focus,
                    language=language,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                answer = f"Unable to generate an answer: {e}"

        st.markdown(answer)

        st.divider()
        with st.expander("📚 Sources Used in Retrieval"):
            for i, item in enumerate(results, start=1):
                st.markdown(
                    f"**Source {i} — Page {item['page']} — "
                    f"Similarity {item['score']:.3f}**"
                )
                st.write(item["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption(
    "CyberLawGPT • RAG + FAISS + Sentence Transformers + Groq GPT-OSS 120B"
)
