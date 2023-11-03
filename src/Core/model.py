from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores.chroma import Chroma
from .prompts import prompt

from dotenv import load_dotenv,find_dotenv
import os


####################STAGE 0 LOAD CONFIG ############################
load_dotenv(find_dotenv(),override=True)
CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
model = HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
database = Chroma(persist_directory="../chroma_db", embedding_function=model)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, openai_api_key=OPEN_AI_API_KEY)
#####################STAGE 1 BUILDING VECTOR DB########################
def load_pdf(file_path):
    loader = PyPDFLoader(file_path= file_path)
    docs = loader.load()
    return docs


def split_documents_into_chunks(docs, chunk_size=800, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(docs)

    # Create a Chroma collection from the document chunks
    collection = Chroma.from_documents(documents=chunks, embedding_function=model, persist_directory="../chroma_db")

    # Create a Chroma collection from the document chunks
    for chunk in chunks:
        collection.add(
            metadatas=chunk.metadata,
            documents=chunk.page_content
        )

    # returning the document chunks
    return chunks
def get_similar_chunks(query, db=database, k=4):
    chunks = db.similarity_search_with_score(query=query, k=k)
    return chunks

def get_response_from_query(query, chunks):
    docs = " ".join([d[0].page_content for d in chunks])
    chain = LLMChain(llm=llm, prompt=prompt)

    output = chain.run({'question': query, 'docs': docs})
    return output

