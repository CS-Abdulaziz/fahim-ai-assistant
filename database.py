import streamlit as st
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def init_pinecone(api_key, index_name):

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def sync_to_pinecone(index, chunks, model, namespace="documind_ns"):
    
    vectors = []
    for i, chunk in enumerate(chunks):
        if chunk.page_content.strip():
            vector = model.embed_query(chunk.page_content)
            vectors.append({
                "id": f"vec_{i}_{st.session_state.get('file_id', '0')}",
                "values": vector,
                "metadata": {"text": chunk.page_content}
            })
    
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
        return True
    return False