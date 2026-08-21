import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from time import sleep


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "messages" not in st.session_state:
    # Each item: {"role": "user"/"assistant", "content": "..."}
    # This list is what gives the app conversation memory + visible history.
    st.session_state.messages = []


def extract_text(content):
    """
    Gemini 3.x models return `.content` as a list of structured blocks
    (each with type/text/extras) instead of a plain string like older
    models did. This pulls out just the readable text so we never show
    raw dicts/signatures to the user.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def document_process(path):
    """Loads a PDF, splits it into chunks, and builds a searchable vector store."""
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

        st.session_state.vector_db = vector_db
        st.session_state.document_uploaded = True
        return True
    except Exception as e:
        # If processing fails (bad PDF, embedding error, etc.), show a clean
        # message instead of crashing, and don't mark the doc as "uploaded".
        st.error(f"Couldn't process this document: {e}")
        st.session_state.document_uploaded = False
        return False


def build_prompt_with_history(context, current_query, history, max_turns=5):
    """
    Builds a single prompt that includes recent conversation turns so the
    model can resolve references like "the second point" back to earlier
    answers. We keep only the last `max_turns` exchanges to avoid the
    prompt growing unbounded as the chat gets long.
    """
    recent = history[-(max_turns * 2):]  # each turn = 1 user + 1 assistant msg

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


st.subheader("📚 Chat with your document - Ask Anything!!")


### document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select a PDF file", type="pdf")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing document..."):
            success = document_process("./uploaded_document.pdf")

        if success:
            st.success("✅ Document processed successfully!")
            sleep(1)
            st.rerun()


### chat UI
if st.session_state.document_uploaded and st.session_state.vector_db:

    # Replay all previous messages so history stays visible across reruns.
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask Anything...")

    if query:
        # Show the user's message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        # Retrieve relevant document chunks
        documents = st.session_state.vector_db.similarity_search(query, k=2)
        context = ""
        for doc in documents:
            context += doc.page_content + "\n\n"

        prompt = build_prompt_with_history(
            context=context,
            current_query=query,
            history=st.session_state.messages[:-1],  # exclude the message we just added
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = llm.invoke(prompt)
                    answer = extract_text(result.content)
                except ResourceExhausted:
                    # Specific Google quota/rate-limit exception
                    answer = (
                        "⚠️ You've reached the AI chat limit for now. "
                        "Please wait a bit and try again."
                    )
                except Exception as e:
                    # Fallback: catch any other API error (network issues,
                    # auth problems, etc.) without crashing the app.
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
