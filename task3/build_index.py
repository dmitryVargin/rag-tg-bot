import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "task2/knowledge_base/"

def create_vector_db():
    start_total = time.time()

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    start_indexing = time.time()

    vector_db = FAISS.from_documents(docs, embeddings)

    end_indexing = time.time()
    indexing_time = end_indexing - start_indexing

    vector_db.save_local("faiss_index")

    end_total = time.time()
    total_time = end_total - start_total


if __name__ == "__main__":
    create_vector_db()
