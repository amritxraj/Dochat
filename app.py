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
# SESSION STATE INITIALIZATION — must happen before anything reads these
# ---------------------------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ---------------------------------------------------------------------------
# CACHED RESOURCES — built once, reused across reruns for speed
# ---------------------------------------------------------------------------
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


llm = get_llm()


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS (unchanged — RAG/LLM logic untouched)
# ---------------------------------------------------------------------------
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
# INITIAL HEADER — only shown before a document is uploaded
# ---------------------------------------------------------------------------
if not st.session_state.document_uploaded:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="color:#C9A876; margin-bottom:0.2rem;">📖 DoChat</h1>
        <p style="color:#8B93A8; margin-top:0;">Upload a document. Ask it anything.</p>
    </div>
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

        st.session_state.document_name = file.name  # store for display later

        with st.spinner("Reading your document..."):
            success = document_process(file_path)

        if success:
            st.success("Ready — ask it anything below.")
            sleep(1)
            st.rerun()


# ---------------------------------------------------------------------------
# DOCUMENT INFO SECTION — replaces the old header once a document is ready
# ---------------------------------------------------------------------------
if st.session_state.document_uploaded and st.session_state.vector_db:
    st.markdown(f"""
    <div style="
        background: #171D2E;
        border: 1px solid rgba(201, 168, 118, 0.35);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0 1.6rem 0;
    ">
        <p style="color:#C9A876; font-weight:700; font-size:1.1rem; margin-bottom:0.4rem;">
            📄 Document uploaded
        </p>
        <p style="
            color:#EDE8DC;
            font-family: monospace;
            font-size:1.05rem;
            background: rgba(255,255,255,0.05);
            display:inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 8px;
            margin-top:0;
            margin-bottom:0.7rem;
        ">
            {st.session_state.document_name}
        </p>
        <p style="color:#8B93A8; margin-top:0; margin-bottom:0;">
            Your document has been uploaded successfully. You can now ask questions about it.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT (unchanged behavior — same message loop, same input handling)
# ---------------------------------------------------------------------------
if st.session_state.document_uploaded and st.session_state.vector_db:

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "📖"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user", avatar="🧑").markdown(query)

        documents = st.session_state.vector_db.similarity_search(query, k=8)
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
        