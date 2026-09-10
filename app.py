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
    page_title="MicroRAG",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom Styling for Modern UX ---
st.markdown(
    """
    <style>
    .stAppDeployButton { display: none !important; }
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
        background: linear-gradient(135deg, #f55036, #ff7849);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.04);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .welcome-step {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }
    .welcome-step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background-color: #f55036;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Curated Models ---
MODEL_OPTIONS = {
    "openai/gpt-oss-20b": "OpenAI GPT-OSS 20B (Fast & Free Tier)",
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B (Heavy Reasoning)",
    "qwen/qwen3.6-27b": "Qwen 3.6 27B (Multilingual & Balanced)",
}


# --- Resource Caching for Embeddings ---
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- Document Processing Function ---
def process_document(file_bytes, file_name, file_extension):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(temp_file_path, encoding="utf-8")
            documents = loader.load()
        else:
            return None, 0, "Unsupported file format."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        embeddings = get_embeddings_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store, len(chunks), None
    except Exception as e:
        return None, 0, str(e)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Secure API Key Handling
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API Key loaded from secrets")
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Obtain a free key at https://console.groq.com/keys",
        )

    # 3 Curated Models
    selected_model = st.selectbox(
        "Groq Model",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda key: MODEL_OPTIONS[key],
        index=0,
        help="Switching models does not lose your processed document!",
    )

    st.divider()

    st.header("📄 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        help="Documents are automatically parsed and embedded.",
    )

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        # Auto-process if not already indexed
        if (
            st.session_state.current_file_id != file_id
            or st.session_state.vector_store is None
        ):
            with st.spinner(f"Processing `{uploaded_file.name}`..."):
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                file_bytes = uploaded_file.getvalue()
                vstore, count, err = process_document(
                    file_bytes, uploaded_file.name, file_extension
                )

                if err:
                    st.error(f"Failed to process document: {err}")
                else:
                    st.session_state.vector_store = vstore
                    st.session_state.current_file_id = file_id
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.chunk_count = count
                    st.session_state.messages = []
                    st.toast(
                        f"Indexed {count} chunks from {uploaded_file.name}!",
                        icon="⚡",
                    )
                    st.rerun()

        # Persistent Active Document Card
        st.success(
            f"🟢 **Ready**\n\n"
            f"**File:** `{st.session_state.current_file_name}`  \n"
            f"**Indexed Chunks:** {st.session_state.chunk_count}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-index", use_container_width=True):
                st.session_state.vector_store = None
                st.session_state.current_file_id = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    else:
        # File was removed by user
        if st.session_state.current_file_id is not None:
            st.session_state.vector_store = None
            st.session_state.current_file_id = None
            st.session_state.current_file_name = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

# --- Main App Header ---
st.markdown('<div class="main-title">⚡ MicroRAG</div>', unsafe_allow_html=True)

if st.session_state.current_file_name and st.session_state.vector_store:
    st.caption(
        f"Active Document: **{st.session_state.current_file_name}** ({st.session_state.chunk_count} chunks) | "
        f"Active Model: **{MODEL_OPTIONS.get(selected_model, selected_model)}**"
    )
else:
    st.caption("High-performance contextual RAG powered by FAISS & Groq")

# --- Empty State / Onboarding Card ---
if not st.session_state.vector_store:
    st.markdown(
        """
        <div class="welcome-card">
            <h4 style="margin-top:0; margin-bottom: 1rem;">👋 Getting Started</h4>
            <div class="welcome-step">
                <div class="welcome-step-num">1</div>
                <div>Enter your <b>Groq API Key</b> in the sidebar (or configure in secrets).</div>
            </div>
            <div class="welcome-step">
                <div class="welcome-step-num">2</div>
                <div>Upload your <b>PDF, TXT, or Markdown</b> file. It indexes automatically!</div>
            </div>
            <div class="welcome-step">
                <div class="welcome-step-num">3</div>
                <div>Ask any question below. You can seamlessly switch models anytime without reprocessing.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Quick Starter Prompts if no chat history yet
    if len(st.session_state.messages) == 0:
        st.markdown("**💡 Quick Question Ideas:**")
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            if st.button("📝 Summarize Document", use_container_width=True):
                st.session_state.retry_prompt = (
                    "Please provide a comprehensive summary of the main points of this document."
                )
                st.rerun()
        with qcol2:
            if st.button("🔑 Key Takeaways", use_container_width=True):
                st.session_state.retry_prompt = (
                    "What are the top 3-5 key takeaways from this document?"
                )
                st.rerun()
        with qcol3:
            if st.button("📊 Data & Numbers", use_container_width=True):
                st.session_state.retry_prompt = (
                    "List any key statistics, metrics, or important data points mentioned in this document."
                )
                st.rerun()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("model"):
            st.caption(
                f"⚡ *Answered by {MODEL_OPTIONS.get(message['model'], message['model'])}*"
            )
        if message.get("sources"):
            with st.expander(
                f"🔍 Retrieved Sources ({len(message['sources'])} chunks)"
            ):
                for idx, src in enumerate(message["sources"], 1):
                    page_val = src.get("metadata", {}).get("page")
                    page_badge = (
                        f" • Page {page_val + 1}" if page_val is not None else ""
                    )
                    st.markdown(f"**Chunk {idx}{page_badge}:**")
                    st.caption(
                        src["content"][:300]
                        + ("..." if len(src["content"]) > 300 else "")
                    )

# --- Chat Input & Execution ---
user_input = st.chat_input(
    "Ask a question about your document..."
    if st.session_state.vector_store
    else "Upload a document first to start chatting..."
)

if st.session_state.retry_prompt:
    user_input = st.session_state.retry_prompt
    st.session_state.retry_prompt = None

if user_input:
    if not api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        st.stop()
    if not st.session_state.vector_store:
        st.error("⚠️ Please upload a document in the sidebar first.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(
                model=selected_model,
                groq_api_key=api_key,
                temperature=0,
            )

            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )

            system_prompt = (
                "You are an expert analysis assistant. Answer the user's question "
                "accurately using only the provided context from their uploaded document.\n\n"
                "Guidelines:\n"
                "1. Base your answer strictly on the context provided below.\n"
                "2. If the answer cannot be found or inferred from the context, state clearly: "
                "'I cannot find the answer to that in the uploaded document.'\n"
                "3. Do not invent external facts or historical data outside the context.\n\n"
                "Context:\n{context}"
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(
                llm, prompt_template
            )
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            captured_sources = []

            def stream_rag_response():
                for chunk in rag_chain.stream({"input": user_input}):
                    if "context" in chunk:
                        captured_sources.clear()
                        captured_sources.extend(
                            [
                                {
                                    "content": doc.page_content,
                                    "metadata": doc.metadata,
                                }
                                for doc in chunk["context"]
                            ]
                        )
                    if "answer" in chunk:
                        yield chunk["answer"]

            full_response = st.write_stream(stream_rag_response)

            st.caption(
                f"⚡ *Answered by {MODEL_OPTIONS.get(selected_model, selected_model)}*"
            )

            if captured_sources:
                with st.expander(
                    f"🔍 Retrieved Sources ({len(captured_sources)} chunks)"
                ):
                    for idx, src in enumerate(captured_sources, 1):
                        page_val = src.get("metadata", {}).get("page")
                        page_badge = (
                            f" • Page {page_val + 1}"
                            if page_val is not None
                            else ""
                        )
                        st.markdown(f"**Chunk {idx}{page_badge}:**")
                        st.caption(
                            src["content"][:300]
                            + ("..." if len(src["content"]) > 300 else "")
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "model": selected_model,
                    "sources": captured_sources,
                }
            )

        except Exception as e:
            st.error(f"Error communicating with Groq API: {e}")
            if st.button("🔄 Retry"):
                st.session_state.retry_prompt = user_input
                st.rerun()
