import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma  # ✅ Using ChromaDB for persistence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import time
import threading

# === ✅ 1. Streamlit Configuration ===
st.set_page_config(page_title="Document Genie", layout="wide", page_icon="📝")

st.markdown("""
    <style>
    h1, h2 { color: #FF6F00 !important; font-weight: bold; }
    .stButton button { background-color: #FF6F00 !important; color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
## 📖 Document Genie: Get Insights from Your PDFs

This AI-powered chatbot leverages Google's Generative AI **Gemini-PRO** to analyze PDF documents and provide instant answers using the **RAG (Retrieval-Augmented Generation)** framework.

### **How It Works**
1. 📂 **Upload PDFs** (Multiple files supported).
2. 🔎 **Ask Questions** (Get precise answers from the document).
""")

# === ✅ 2. API Key Handling (Streamlit Cloud Compatible) ===
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key or not isinstance(api_key, str):
    st.error("🚨 API Key missing! Add GOOGLE_API_KEY to `.streamlit/secrets.toml`.")
    st.stop()

genai.configure(api_key=api_key)

# === ✅ 3. Helper Functions ===
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # ✅ Prevent NoneType issues
    return text

def get_text_chunks(text):
    """Split text into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a persistent vector store using ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="./vector_db")  # ✅ Persistent storage
    vector_store.persist()
    st.session_state["vector_store"] = vector_store  # ✅ Store in session state
    st.success("✅ Document processed successfully!")

def get_conversational_chain():
    """Load the QA Chain for document-based answering."""
    prompt_template = """
    Answer the question based only on the provided context.
    If the answer is not in the context, respond with "Answer not available in the document."

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# === ✅ 4. API Retry Function ===
def retry_api_call(api_call, max_retries=3, wait=5):
    """Retry API calls to handle temporary failures."""
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as e:
            st.warning(f"⚠️ API Error: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(wait)
    st.error("🚨 API is not responding. Please try again later.")
    return None

def user_input(user_question):
    """Handle user queries using the stored vector database."""
    if "vector_store" not in st.session_state:
        st.error("❌ No document processed. Please upload PDFs first.")
        return

    vector_store = st.session_state["vector_store"]
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = retry_api_call(lambda: chain({"input_documents": docs, "question": user_question}, return_only_outputs=True))

    if response:
        st.write("✍️ **Reply:** ", response["output_text"])

# === ✅ 5. Streamlit UI ===
def main():
    st.header("🤖 AI Document Chatbot")

    user_question = st.text_input("🔎 **Ask a Question from the PDF Files**", key="user_question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload & Process Documents")
        pdf_docs = st.file_uploader("📥 Upload PDFs", accept_multiple_files=True, key="pdf_uploader")

        if st.button("🚀 Process PDFs", key="process_button"):
            with st.spinner("⏳ Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

if __name__ == "__main__":
    main()
