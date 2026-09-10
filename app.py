import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Configuration ---
st.set_page_config(
    page_title="MicroRAG", page_icon="⚡", layout="centered"
)


# --- Resource Caching for Seamless UI ---
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Secure API Key Handling
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API Key securely loaded.")
    else:
        api_key = st.text_input("Enter Groq API Key", type="password")

    selected_model = st.selectbox(
        "Groq Model",
        options=[
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "qwen/qwen3.8-27b",
            "groq/compound",
            "groq/compound-mini",
        ],
        index=0,
        help="Select an active model from your Groq console.",
    )

    st.divider()

    st.header("📄 Upload Document")
    # Expanded file types to support PDF, TXT, and Markdown
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

    if st.button("Process Document") and uploaded_file and api_key:
        with st.spinner("Analyzing document and building vector database..."):
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                # Conditional loading based on file extension
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                elif file_extension in [".txt", ".md"]:
                    loader = TextLoader(temp_file_path, encoding="utf-8")
                    documents = loader.load()
                else:
                    st.error("Unsupported file format.")
                    st.stop()

                # Split text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)

                # Vectorize and cache
                embeddings = get_embeddings_model()
                vector_store = FAISS.from_documents(chunks, embeddings)

                # Store in session state and clear past history if a new file is uploaded
                st.session_state.vector_store = vector_store
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = (
                    []
                )  # Clear chat history for the new document

                st.success(f"Successfully processed: {uploaded_file.name}")

            except Exception as e:
                st.error(f"An error occurred while parsing: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# --- Main Chat Interface ---
st.title("⚡ MicroRAG")
if st.session_state.current_file:
    st.caption(f"Active Document: **{st.session_state.current_file}**")
else:
    st.markdown(
        "Upload a PDF, TXT, or Markdown file in the sidebar to initialize the RAG pipeline."
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Retry Logic ---
user_input = st.chat_input("Ask a question about the document...")

if st.session_state.retry_prompt:
    user_input = st.session_state.retry_prompt
    st.session_state.retry_prompt = None

if user_input:
    if (
        not st.session_state.messages
        or st.session_state.messages[-1]["content"] != user_input
    ):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please provide a Groq API Key.")
            st.stop()
        if not st.session_state.vector_store:
            st.error("Please upload and process a document first.")
            st.stop()

        try:
            os.environ["GROQ_API_KEY"] = api_key
            llm = ChatGroq(
                model=selected_model,
                groq_api_key=api_key,
                temperature=0,
            )

            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4
                },  # Bumped context chunks to 4 for better document coverage
            )

            # Genericized system prompt for any context
            system_prompt = (
                "You are an expert data analysis assistant. Your sole task is to answer "
                "the user's question accurately using only the provided context from their uploaded document.\n\n"
                "Guidelines:\n"
                "1. Base your answer strictly on the context provided below.\n"
                "2. If the answer cannot be found or inferred from the context, state clearly: "
                "'I cannot find the answer to that in the uploaded document.'\n"
                "3. Do not make up facts, external statistics, or historical information outside the text.\n\n"
                "Context:\n{context}"
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            def stream_rag_response():
                for chunk in rag_chain.stream({"input": user_input}):
                    if "answer" in chunk:
                        yield chunk["answer"]

            full_response = st.write_stream(stream_rag_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Failed to communicate with the Groq API. Reason: {e}")
            if st.button("🔄 Retry"):
                st.session_state.retry_prompt = user_input
                st.rerun()
