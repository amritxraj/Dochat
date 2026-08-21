import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def docment_process(path):
    # document loading
    loader = PyPDFLoader(path)
    docs = loader.load()

    ## splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    ## embedding and vector stores
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_db = InMemoryVectorStore.from_documents(documents = docs, embedding = embeddings)


st.subheader("Chat with your document - Ask Anything")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False


### document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select a PDF file", type="pdf")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())
        st.markdown("Document uploaded successfully!")

### chat UI

