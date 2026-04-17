import os
from uuid import uuid4
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def init_pinecone(api_key, index_name):
    if not api_key or not index_name:
        return None

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def sync_to_pinecone(index, chunks, model, file_id="0", namespace="documind_ns"):
    if index is None or model is None:
        return False

    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"Log: Namespace already empty or error: {e}")

    vectors = []
    for i, chunk in enumerate(chunks):
        if chunk.page_content.strip():
            vector = model.embed_query(chunk.page_content)
            vectors.append({
                "id": f"doc-{i}-{uuid4().hex}",
                "values": vector,
                "metadata": {"text": chunk.page_content}
            })
    
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        return True
    return False
