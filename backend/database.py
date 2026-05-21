import os
from uuid import uuid4
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def init_pinecone(api_key: str, index_name: str):
    if not api_key:
        raise ValueError("Pinecone API key must be provided.")
    if not index_name:
        raise ValueError("Pinecone index name must be provided.")

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def sync_to_pinecone(index, chunks, model, file_id="0", namespace="documind_ns"):
    if index is None or model is None:
        return False

    # Use deterministic chunk IDs to overwrite existing data rather than performing a destructive global delete
    vectors = []
    for i, chunk in enumerate(chunks):
        if chunk.page_content.strip():
            vector = model.embed_query(chunk.page_content)
            vectors.append({
                "id": f"doc-{file_id}-{i}",
                "values": vector,
                "metadata": {"text": chunk.page_content}
            })
    
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        return True
    return False
