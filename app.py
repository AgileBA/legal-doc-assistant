import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.jina import JinaEmbeddings # Your corrected import path
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pytesseract
from pdf2image import convert_from_bytes

# --- Core Functions: Optimized for Speed and Accuracy (Your versions) ---

def get_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            extracted_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            if extracted_text.strip():
                text += extracted_text
            else:
                pdf.seek(0)
                images = convert_from_bytes(pdf.read())
                ocr_text = "".join(pytesseract.image_to_string(image) for image in images)
                text += ocr_text
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    if text:
        st.info("Successfully extracted text from PDF(s).")
    return text

def get_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = ' '.join(t.strip() for t in soup.stripped_strings)
        if not text:
            st.warning("Could not find any meaningful text content on the page.")
            return None
        st.info("Successfully fetched and parsed content from the URL.")
        return text
    except Exception as e:
        st.error(f"Could not process URL: {e}")
        return None

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, jina_api_key):
    if not text_chunks:
        return None
    try:
        # Minimal change: add session=None because your installed JinaEmbeddings signature expects it
        embeddings = JinaEmbeddings(
            jina_api_key=jina_api_key,
            model_name="jina-embeddings-v2-base-en",
            session=None
        )
        return FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store using Jina API: {e}")
        return None

def get_rag_chain(api_key):
    prompt_template = """
    You are a highly skilled AI assistant, an expert in document analysis and information synthesis. Your primary directive is to answer the user's question with a detailed, comprehensive, and well-structured response, derived *exclusively* from the provided text context.

    **Instructions:**

    1.  **Analyze the Entire Context:** Thoroughly read and understand all the information contained in the "CONTEXT" section below.
    2.  **Synthesize, Do Not Just Copy:** Do not simply copy-paste sentences from the context. Synthesize the relevant information into a coherent, flowing, and comprehensive answer.
    3.  **Structure Your Answer:** Organize the answer in a logical manner. Use markdown for clarity:
        - Use **bold text** to highlight key terms, names, or figures.
        - Use bullet points (`-`) for lists of items or key points.
    4.  **Strictly Adhere to the Context:** Your answer **MUST** be based only on the provided context. Do not use any external knowledge. If the context does not contain the information required to answer the question, you **MUST** respond with the exact phrase: "The provided context does not contain sufficient information to answer this question."

    ---
    **CONTEXT:**
    {context}
    ---
    **QUESTION:**
    {question}
    ---
    **COMPREHENSIVE AND STRUCTURED ANSWER:**
    """
    # Minimal change: use the parameter names expected by the installed ChatGroq
    llm = ChatGroq(api_key=api_key, model="llama3-8b-8192", temperature=0.2)
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | PromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Main Streamlit Application UI ---

def main():
    load_dotenv()
    st.set_page_config(page_title="Content AI Assistant", layout="wide", initial_sidebar_state="expanded")

    groq_api_key = os.getenv("GROQ_API_KEY")
    jina_api_key = os.getenv("JINA_API_KEY")

    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    # NEW: Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("📚 Content AI Assistant")
        st.markdown("---")
        
        if groq_api_key and jina_api_key:
            st.success("All API Keys Loaded Successfully!", icon="✅")
        else:
            if not groq_api_key:
                st.error("Groq API Key not found in .env file.", icon="❌")
            if not jina_api_key:
                st.error("Jina API Key not found in .env file.", icon="❌")

        st.subheader("Provide Your Content")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
        st.markdown("<h5 style='text-align: center;'>OR</h5>", unsafe_allow_html=True)
        url_input = st.text_input("Enter a URL")

        if st.button("Process Content"):
            if not (groq_api_key and jina_api_key):
                st.warning("Cannot process content without both API keys in the .env file.")
            else:
                raw_text = None
                if pdf_docs:
                    raw_text = get_text_from_pdfs(pdf_docs)
                elif url_input:
                    raw_text = get_text_from_url(url_input)
                else:
                    st.warning("Please upload a PDF or enter a URL.")

                if raw_text:
                    with st.spinner("Creating vector store with Jina API..."):
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(text_chunks, jina_api_key)
                        # NEW: Reset chat history for the new content
                        st.session_state.chat_history = []
                        if st.session_state.vector_store:
                            st.success("Ready! You can now ask questions.")
                elif url_input or pdf_docs:
                    st.error("Processing failed to extract any text.")
                    st.session_state.vector_store = None

    st.header("Chat with Your Content")

    # NEW: Display chat history from session state
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # NEW: Use st.chat_input for a persistent input box
    if user_question := st.chat_input("Ask a question about the content..."):
        if not st.session_state.vector_store:
            st.warning("Please process a document or URL first.")
        elif not groq_api_key:
            st.warning("Cannot answer questions without a valid Groq API key.")
        else:
            # Add user's question to history and display it
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Generate and display the AI's response
            with st.spinner("Groq is thinking..."):
                with st.chat_message("assistant"):
                    docs = st.session_state.vector_store.similarity_search(user_question, k=5)
                    context = "\n".join([doc.page_content for doc in docs])
                    chain = get_rag_chain(groq_api_key)
                    response = chain.invoke({"context": context, "question": user_question})
                    st.markdown(response)
                    # Add AI's response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Show a welcome message if no content has been processed yet
    if not st.session_state.vector_store:
        st.info("Welcome! Please provide content in the sidebar to start chatting.")

if __name__ == "__main__":
    main()
