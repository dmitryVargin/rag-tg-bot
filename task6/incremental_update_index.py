import os
import time
import logging
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    filename='task6/update_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATA_PATH = "task2/knowledge_base/"
INDEX_PATH = "faiss_index"

def get_existing_sources(vector_db):
    if not vector_db or not vector_db.docstore:
        return set()
    
    sources = set()
    for doc_id in vector_db.index_to_docstore_id.values():
        doc = vector_db.docstore.search(doc_id)
        if doc and 'source' in doc.metadata:
            sources.add(os.path.abspath(doc.metadata['source']))
    return sources

def update_vector_db_incremental():
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    try:
        if os.path.exists(INDEX_PATH):
            vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            existing_sources = get_existing_sources(vector_db)
            logging.info(f"Загружен существующий индекс. Найдено источников: {len(existing_sources)}")
        else:
            vector_db = None
            existing_sources = set()
            logging.info("Существующий индекс не найден. Будет создан новый.")

        loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        all_documents = loader.load()
        
        new_documents = []
        for doc in all_documents:
            abs_path = os.path.abspath(doc.metadata['source'])
            if abs_path not in existing_sources:
                new_documents.append(doc)
        
        if not new_documents:
            log_msg = "Обновление не требуется: новых файлов не обнаружено."
            print(log_msg)
            logging.info(log_msg)
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        new_docs_chunks = text_splitter.split_documents(new_documents)

        if vector_db:
            vector_db.add_documents(new_docs_chunks)
            logging.info(f"Добавлено {len(new_docs_chunks)} чанков из {len(new_documents)} новых файлов.")
        else:
            vector_db = FAISS.from_documents(new_docs_chunks, embeddings)
            logging.info(f"Создан новый индекс с {len(new_docs_chunks)} чанками.")

        vector_db.save_local(INDEX_PATH)
        
        duration = time.time() - start_time
        log_msg = f"УСПЕХ: Индекс обновлен (инкрементально). Добавлено файлов: {len(new_documents)}, Время: {duration:.2f}с"
        print(log_msg)
        logging.info(log_msg)

    except Exception as e:
        err_msg = f"ОШИБКА при обновлении индекса: {e}"
        print(err_msg)
        logging.error(err_msg, exc_info=True)

if __name__ == "__main__":
    update_vector_db_incremental()
