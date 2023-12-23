import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os
import time
import shutil
import fitz
from langchain.docstore.document import Document

# --- GENERAL SETTINGS ---
PAGE_TITLE = "DENSO GPT Expert"
PAGE_ICON = "🤖"
AI_MODEL_OPTIONS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-32k",
]

loaders = []
docs = []

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
storage_dir = current_dir.parent.parent / "storage"
persist_directory_dir = str(current_dir.parent.parent.joinpath("chroma_db"))
#model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder") - BKAI
model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
database = Chroma(persist_directory=persist_directory_dir, embedding_function=model)

ss = st.session_state
ss.setdefault('debug', {})

def split_documents(docs, chunk_size=800, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Splitting the documents into chunks
    chunks = text_splitter.create_documents([docs])

    return chunks

def insert_pdf_to_db(file_path):
    pages = fitz.open(file_path)
    chunks = []

    for page in pages:
        docs = split_documents(page.get_text().replace('\n', ' '))
        for doc in docs:
            chunk = Document(page_content=doc.page_content, metadata={"source": pages.name,"page": page.number})
            chunks.append(chunk)
    db2 = Chroma.from_documents(chunks, model, persist_directory=persist_directory_dir)
    st.write(chunks)

if __name__ == '__main__':
    st.title(PAGE_TITLE)
    st.sidebar.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.sidebar.write("Welcome to the DENSO GPT Expert")

    # the guild of the chatbot in the sidebar
    with st.sidebar.expander("ℹ️ About"):
        st.write("Mục \"UPLOAD\" dùng để upload file PDF lên hệ thống")
        st.write("Bước 1: Click nút \"Browse files\"  để upload data lên hệ thống")
        st.write("Bước 2: Click nút \"Save to storage\" để lưu data vào hệ thống")
        st.write("Mục \"STORAGE\" dùng để xem các file PDF đã upload lên hệ thống")

    st.write('## Upload your PDF')
    t1, t2 = st.tabs(['UPLOAD', 'STORAGE'])
    with t1:
        uploaded_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
        result = st.button("Save to storage")
        # if the user clicks the button
        if len(uploaded_files) and result:
            with st.spinner("Processing..."):
                if not storage_dir.exists():
                    storage_dir.mkdir(parents=True, exist_ok=True)
                for uploaded_file in uploaded_files:
                    # save the uploaded file to the storage folder
                    with open(os.path.join(storage_dir, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("Saved File:{} to storage".format(uploaded_file.name))

                files_in_storage = list(storage_dir.iterdir())
                if len(files_in_storage) > 0:
                    for file in files_in_storage:
                        file_path = str(storage_dir / file.name)
                        insert_pdf_to_db(file_path)
                else:
                    st.write("No files in storage.")

    with t2:
        persist_directory_dir_db = current_dir.parent.parent / "chroma_db"
        files_in_storage = list(storage_dir.iterdir())
        persist_files = list(persist_directory_dir_db.iterdir())
        if storage_dir.exists():
            if files_in_storage:
                st.write("### Files in storage:")
                for file in files_in_storage:
                    st.write(file.name)
            else:
                st.write("No files in storage.")
        else:
            st.write("Storage folder does not exist.")
        clear_btn = st.button("Clear storage")
        if clear_btn:
            with st.spinner("Processing..."):
                for file in files_in_storage:
                    os.remove(str(storage_dir / file.name))
                # for file in persist_files:
                #     st.write(str(persist_directory_dir_db / file.name))
                #     os.chmod(str(persist_directory_dir_db / file.name), stat.S_IWRITE)
                #     os.remove(str(persist_directory_dir_db / file.name))
                try:
                    shutil.rmtree(persist_directory_dir_db)
                except PermissionError:
                    time.sleep(1)  # Add a delay here
                    shutil.rmtree(persist_directory_dir_db)

                st.success("Cleared storage")
                st.write("No files in storage.")