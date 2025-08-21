# 🧠 AI Content Analysis Assistant

A powerful and lightning-fast web application built with **Streamlit** and **LangChain**, designed to analyze and answer questions about PDF documents and web content. It supports both digital and scanned PDFs, and integrates cutting-edge APIs for high performance.

---

## 🚀 Features

- **📄 Multi-Source Input**: Upload PDF files (digital or scanned) or input any webpage URL.
- **🧠 Advanced Text Extraction**: Automatically handles OCR for scanned/image-based PDFs.
- **⚡ High-Speed Processing**: Utilizes Groq and Jina AI APIs for blazing-fast computation.
- **💬 High-Quality Q&A**: Employs a powerful LLM with refined prompting for detailed, structured answers.
- **🔐 Secure API Key Management**: API keys are safely stored using `.env` (never exposed in code/UI).
- **✨ User-Friendly Interface**: Simple, clean, and intuitive UI built using Streamlit.

---

## 🛠️ Tech Stack

- **LLM Chat Model**: [Groq](https://console.groq.com/) running **Llama 3**
- **Embeddings**: [Jina AI](https://jina.ai/embeddings/)
- **OCR**: Tesseract via `pytesseract` for scanned PDFs
- **UI**: Streamlit
- **Framework**: LangChain

---

## 📁 Project Structure

content-ai-assistant/
├── .gitignore
├── app.py
├── requirements.txt
└── .env



---

## 🧪 Setup & Installation

### 1️⃣ Get Free API Keys

#### 🔹 Groq API Key (for LLM Chat):
- Go to [console.groq.com/keys](https://console.groq.com/keys)
- Sign up with Google or GitHub
- Generate and copy your API key

#### 🔹 Jina AI API Key (for Embeddings):
- Go to [jina.ai/embeddings](https://jina.ai/embeddings/)
- Sign in and copy the API key from the dashboard

---

### 2️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd content-ai-assistant
------------------------------------------------/
Set Up the Environment

# Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

---------------------------------------------------------/


Then install dependencies

pip install -r requirements.txt

-------------------------------------------------------------/

 Configure API Keys
Create a .env file in the root directory:

GROQ_API_KEY="paste_your_groq_api_key_here"
JINA_API_KEY="paste_your_jina_api_key_here"

-------------------------------------------------------------/

How to Run
Launch the app with:

streamlit run app.py

