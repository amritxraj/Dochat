import os
import uuid
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from time import sleep

st.set_page_config(page_title="DoChat", page_icon="📖", layout="centered")

# ---------------------------------------------------------------------------
# STYLING — "library at dusk": ink-blue depths, brass/gold accents, a serif
# nameplate over clean sans body text. One signature flourish (the gold
# "spine" rule under the header) — everything else stays quiet.
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;600&display=swap');

:root {
    --bg-deep: #0F1420;
    --bg-panel: #171D2E;
    --bg-panel-light: #1E2740;
    --accent-gold: #C9A876;
    --accent-slate: #6B8CAE;
    --text-primary: #EDE8DC;
    --text-muted: #8B93A8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0B0F18 0%, var(--bg-deep) 100%);
    color: var(--text-primary);
}

.dochat-header {
    text-align: center;
    padding: 2.2rem 0 0.6rem 0;
}
.dochat-header h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.4rem;
    letter-spacing: 0.01em;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.dochat-header p {
    font-family: 'Inter', sans-serif;
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0;
}

.dochat-spine {
    width: 120px;
    height: 3px;
    margin: 0.9rem auto 2.2rem auto;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
    border-radius: 2px;
}

[data-testid="stFileUploader"] {
    background: var(--bg-panel);
    border: 1px dashed rgba(201, 168, 118, 0.35);
    border-radius: 14px;
    padding: 1.4rem;
}
[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stChatMessage"] {
    background: var(--bg-panel);
    border-radius: 14px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.04);
}

[data-testid="stChatInput"] {
    border-radius: 14px;
}
[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 2px rgba(201, 168, 118, 0.45);
    border-radius: 14px;
}

.stButton button {
    background: var(--accent-gold);
    color: #0F1420;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}
.stButton button:hover {
    background: #D9BC8D;
    color: #0F1420;
}

.stAlert {
    border-radius: 10px;
}

.block-container {
    padding-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# APP LOGIC (unchanged behavior — just wired up to the new visuals)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")

llm = get_llm()

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def extract_text(content):
    """Gemini 3.x returns content as a list of structured blocks instead of
    a plain string. This pulls just the text back out."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def document_process(path):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(docs)

        embeddings = get_embeddings()
        vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

        st.session_state.vector_db = vector_db
        st.session_state.document_uploaded = True
        return True
    except Exception as e:
        st.error(f"Couldn't process this document: {e}")
        st.session_state.document_uploaded = False
        return False


def build_prompt_with_history(context, current_query, history, max_turns=5):
    recent = history[-(max_turns * 2):]
    convo_text = ""
    for msg in recent:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        convo_text += f"{speaker}: {msg['content']}\n"

    prompt = (
        "You are answering questions about a document. Use the document "
        "context below, and the conversation so far, to answer the latest "
        "question. If the question refers to something mentioned earlier "
        "in the conversation, use that earlier context to understand it.\n\n"
        f"Document context:\n{context}\n\n"
        f"Conversation so far:\n{convo_text}\n"
        f"User: {current_query}\n"
        "Assistant:"
    )
    return prompt


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="dochat-header">
    <h1>📖 DoChat</h1>
    <p>Upload a document. Ask it anything.</p>
</div>
<div class="dochat-spine"></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------------------------
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select a PDF file", type="pdf")
    if file:
        file_path = f"uploaded_{st.session_state.session_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Reading your document..."):
            success = document_process(file_path)

        if success:
            st.success("Ready — ask it anything below.")
            sleep(1)
            st.rerun()


# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------
if st.session_state.document_uploaded and st.session_state.vector_db:

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "📖"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    query = st.chat_input("Ask anything about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user", avatar="🧑").markdown(query)

        documents = st.session_state.vector_db.similarity_search(query, k=2)
        context = ""
        for doc in documents:
            context += doc.page_content + "\n\n"

        prompt = build_prompt_with_history(
            context=context,
            current_query=query,
            history=st.session_state.messages[:-1],
        )

        with st.chat_message("assistant", avatar="📖"):
            with st.spinner("Turning pages..."):
                try:
                    result = llm.invoke(prompt)
                    answer = extract_text(result.content)
                except ResourceExhausted:
                    answer = (
                        "⚠️ You've reached the AI chat limit for now. "
                        "Please wait a bit and try again."
                    )
                except Exception as e:
                    error_text = str(e)
                    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
                        answer = (
                            "⚠️ You've reached the AI chat limit for now. "
                            "Please wait a bit and try again."
                        )
                    else:
                        answer = f"⚠️ Something went wrong while getting a response: {error_text}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})