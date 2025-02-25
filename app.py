import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # ✅ Fixed Deprecated Import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import time
import threading

# === ✅ 1. Fix: Streamlit Config & Custom Styling ===
st.set_page_config(page_title="Document Genie", layout="wide", page_icon="📝")

st.markdown("""
    <style>
    h1, h2 { color: #FF6F00 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. 
It processes uploaded PDF documents, creates a searchable vector store, and generates accurate answers.

## How It Works
1. **Upload Your Documents** (Multiple PDFs allowed).
2. **Ask a Question** (Query about the document).
""")

# === ✅ 2. Fix: API Key Retrieval & Validation ===
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not api_key or not isinstance(api_key, str):
    st.error("API Key is missing or invalid. Please set GOOGLE_API_KEY in `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

# === ✅ 3. Fix: Keep Streamlit Session Alive ===
def keep_alive():
    while True:
        time.sleep(240)  # Ping every 4 minutes
        st.write("Keeping app active...")

threading.Thread(target=keep_alive, daemon=True).start()

# === ✅ 4. Helper Functions ===
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # ✅ Prevents NoneType issues
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    genai.configure(api_key=api_key)  # ✅ Explicit API Key Setting
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("/tmp/faiss_index")
        st.success("Vector store successfully created!")
    except Exception as e:
        st.error(f"Embedding error: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question based only on the provided context. 
    If the answer is not in the context, respond with "Answer is not available in the context."
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# === ✅ 5. Fix: Auto-Retry API Calls ===
def retry_api_call(api_call, max_retries=3, wait=5):
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as e:
            st.warning(f"API Error: {e}. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(wait)
    st.error("API is not responding. Please try again later.")
    return None

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

        if os.path.exists("/tmp/faiss_index"):
            new_db = FAISS.load_local("/tmp/faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            st.error("FAISS index not found. Please re-upload PDFs.")
            return

        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        response = retry_api_call(lambda: chain({"input_documents": docs, "question": user_question}, return_only_outputs=True))

        if response:
            st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"Error: {e}")

# === ✅ 6. Streamlit UI ===
def main():
    st.header("AI Document Chatbot 📝")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")

        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
