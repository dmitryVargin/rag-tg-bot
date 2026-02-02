from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

query = "Кто такой Reed Guru?"

results = db.similarity_search(query, k=3)

print(f"\nРезультаты поиска по запросу: '{query}'\n")
for i, res in enumerate(results):
    print(f"Отрывок №{i+1}:")
    print(res.page_content)
    print("-" * 30)
