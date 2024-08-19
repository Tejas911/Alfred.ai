import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    YoutubeLoader,
    UnstructuredURLLoader,
)
from langchain.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fpdf import FPDF
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def get_document_text(docs):
    text = ""
    for doc in docs:
        try:
            if doc.name.endswith(".pdf"):
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif doc.name.endswith(".txt"):
                text += doc.read().decode("utf-8")
            else:
                st.warning(f"Unsupported file type: {doc.name}")
        except Exception as e:
            st.error(f"Error reading file {doc.name}: {e}")
            if doc.name.endswith(".pdf"):
                try:
                    doc.seek(0)
                    text += doc.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Error reading text file {doc.name}: {e}")

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    progress_bar = st.progress(0)
    total_chunks = len(text_chunks)

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    for i, _ in enumerate(text_chunks):
        time.sleep(0.1)
        progress_bar.progress((i + 1) / total_chunks)

    vector_store.save_local("faiss_index")
    st.success("Vector store saved successfully.")
    return vector_store


def get_conversational_chain(model_name):
    prompt_template = """
    Answer the questions based on the provided context only.
    - Provide the most accurate and clear response based strictly on the context.
    - If the answer is not available in the context, state, "Answer is not available in the context."
    - Offer any relevant additional information or background on the topic to enhance the user's understanding.
    - Always mention the page number(s) or section where you found the information.
    - Based on the current question and the context, generate three more related questions.
    <context>
    {context}
    <context>
    Questions: {input}
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)
    return chain


def user_input(user_question, model_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()
    chain = get_conversational_chain(model_name)
    retrieval_chain = create_retrieval_chain(retriever, chain)

    progress_bar = st.progress(0)

    for i in range(5):
        time.sleep(0.5)
        progress_bar.progress((i + 1) * 20)

    response = retrieval_chain.invoke({"input": user_question})
    return response["answer"], response["context"]


def generate_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for msg in chat_history:
        msg_type = msg["type"].capitalize()
        avatar = msg["avatar"]
        content = msg["content"]
        content = content.encode("latin1", "replace").decode("latin1")
        pdf.multi_cell(0, 10, f"{msg_type} ({avatar}): {content}")
    return pdf


def load_documents(docs, urls):
    text = get_document_text(docs)

    if urls:
        for url in urls:
            try:
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = WebBaseLoader(url)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    text += doc.page_content
            except Exception as e:
                st.error(f"Error loading URL {url}: {e}")

    return text


def main():
    st.set_page_config(
        page_title="Alfred.ai",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "# This is a header. This is an *extremely* cool app!"},
    )

    logo_url = ""
    col1, col2 = st.columns([1, 17])
    with col1:
        st.image(logo_url, width=55)
    with col2:
        st.header("ChatPDF")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_confirmation" not in st.session_state:
        st.session_state.show_confirmation = False
    if "reset_confirmed" not in st.session_state:
        st.session_state.reset_confirmed = False
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "embeddings_generated" not in st.session_state:
        st.session_state.embeddings_generated = False

    model_options = {
        "Gemma2-9b-IT": "gemma2-9b-it",
        "Llama3-70b-8192": "llama3-70b-8192",
        "Mixtral-8x7b-32768": "mixtral-8x7b-32768",
    }
    selected_model = st.sidebar.selectbox(
        "Select LLM Model", options=list(model_options.keys())
    )
    selected_model_name = model_options[selected_model]

    for msg in st.session_state.history:
        st.chat_message(msg["type"], avatar=msg["avatar"]).write(msg["content"])

    if user_question := st.chat_input("Ask a Question from the PDF Files"):
        st.session_state.last_question = user_question
        st.chat_message("human", avatar="").write(user_question)
        answer, context = user_input(user_question, selected_model_name)
        st.chat_message("ai", avatar="").write(answer)
        st.session_state.history.append(
            {
                "type": "human",
                "content": user_question,
                "avatar": "",
            }
        )
        st.session_state.history.append(
            {
                "type": "ai",
                "content": answer,
                "avatar": "",
            }
        )

    if st.session_state.last_question:
        if st.button("Regenerate"):
            st.chat_message("human", avatar="").write(st.session_state.last_question)
            answer, context = user_input(
                st.session_state.last_question, selected_model_name
            )
            st.chat_message("ai", avatar="").write(answer)
            st.session_state.history.append(
                {
                    "type": "human",
                    "content": st.session_state.last_question,
                    "avatar": "",
                }
            )
            st.session_state.history.append(
                {
                    "type": "ai",
                    "content": answer,
                    "avatar": "",
                }
            )

    if not st.session_state.embeddings_generated:
        docs = st.file_uploader(
            "Upload your PDF or text Files",
            accept_multiple_files=True,
            type=["pdf", "txt"],
        )

        # URL input section
        st.write("### Add URLs for Processing")

        # Number of URLs input
        num_urls = st.number_input(
            "How many URLs do you want to process?", min_value=0, step=1
        )

        urls = []

        for i in range(num_urls):
            url = st.text_input(f"Enter URL {i + 1}")
            if url:
                urls.append(url)

        if st.button("Submit & Process"):
            if docs or urls:
                with st.spinner("Processing..."):
                    raw_text_with_pages = load_documents(docs, urls)

                    text_chunks = get_text_chunks(raw_text_with_pages)
                    get_vector_store(text_chunks)
                    st.session_state.embeddings_generated = True
                    st.success("Embeddings Stored Successfully")

                st.write(
                    "### 🤔 Here are some suggested questions based on your Resource"
                )
                st.write("📝 **Can you summarise the key points discussed?**")
                st.write("🧐 **What are the most important takeaways?**")
                st.write("🎯 **What is the conclusion?**")

    with st.sidebar:
        st.title("Menu")

        if st.session_state.history:
            pdf = generate_pdf(st.session_state.history)
            pdf_output = pdf.output(dest="S").encode("latin1")
            st.download_button(
                label="Download Chat History",
                data=pdf_output,
                file_name="chat_history.pdf",
                mime="application/pdf",
            )

        if st.button("Reset Chat"):
            st.session_state.show_confirmation = True

        if st.session_state.show_confirmation:
            st.error("Are you sure you want to delete the chat history?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm"):
                    st.session_state.history = []
                    st.session_state.show_confirmation = False
                    st.session_state.reset_confirmed = True
                    st.experimental_rerun()
            with col2:
                if st.button("Close"):
                    st.session_state.show_confirmation = False
                    st.experimental_rerun()

        if st.session_state.reset_confirmed:
            st.success("Chat history has been reset")
            st.session_state.reset_confirmed = False
        st.markdown(
            "<p style=' margin-top: 200px;'>Powered by Groq </p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
