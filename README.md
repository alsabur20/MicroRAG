# 📚 MicroRAG

An elegant, high-performance Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, and **Groq**.

This application allows users to upload custom documents (PDF, TXT, or Markdown) and have seamless, real-time contextual conversations backed by semantic vector search using FAISS and HuggingFace embeddings.

## ✨ Key Features

- **Multi-Format Support:** Seamless text extraction and chunking for `.pdf`, `.txt`, and `.md` files.
- **Resource Caching:** Pre-loads HuggingFace embedding models (`all-MiniLM-L6-v2`) into memory instantly for fluid interaction and faster reruns.
- **Real-Time Streaming:** Responses type out character-by-character dynamically, mimicking the native ChatGPT experience.
- **Resilient Architecture:** Graceful exception handling equipped with a built-in interactive **Retry** state layout for API timeouts.
- **Secure Key Management:** Implements Streamlit's official encrypted production workflow (`st.secrets`), ensuring your API keys never touch version control.

## 🛠️ Prerequisites

- Python 3.10 or 3.11 recommended.
- A free API key from [Groq](https://console.groq.com/keys).

## 🚀 Local Setup Instructions

1. **Clone the repository** (or download the project folder):
    ```bash
    git clone [https://github.com/alsabur20.MicroRAG](https://github.com/alsabur20/MicroRAG)
    cd MicroRAG
    ```
2. **Create a virtual environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Securely configure your API key**:
   Create a hidden folder named .streamlit in the root of your project, and inside it, create a secrets.toml file.
   ```bash
    mkdir .streamlit
    touch .streamlit/secrets.toml
    ```
   Add your Groq API key to the secrets.toml file:
   ```toml
    GROQ_API_KEY = "your_actual_groq_api_key_here"
    ```
   ***(Note: Ensure .streamlit/ is included in your .gitignore file before pushing to GitHub).***

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 💡 Usage Guide
1. Open the local URL provided in your terminal (usually http://localhost:8501).

2. If you didn't set up the secrets.toml file, paste your Groq API key into the sidebar.

3. Upload a document (.pdf, .txt, or .md) via the sidebar.

4. Wait a few seconds for the document to be processed and vectorized.

5. Start chatting! Ask specific questions about the contents of your uploaded document.
