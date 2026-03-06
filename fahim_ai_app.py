import streamlit as st
import openai
from utils import process_uploaded_file
from database import load_embedding_model, init_pinecone, sync_to_pinecone

st.set_page_config(page_title="Fahim AI", page_icon="🧠", layout="centered")

st.markdown("""
            
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    .stChatInputContainer {
        direction: rtl;
    }
    div[data-testid="stChatMessageContent"] {
        text-align: right;
        direction: rtl;
    }
    .stButton>button {
        float: right;
    }
    </style>
    """, unsafe_allow_html=True)

try:

    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

except KeyError as e:

    st.error(f"Secret Key {e} not found")
    st.stop()

client = openai.OpenAI(

    base_url="https://openrouter.ai/api/v1", 
    api_key=OPENROUTER_API_KEY

)

embed_model = load_embedding_model()
vector_db = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)

st.title("🧠 Fahim AI")

file = st.file_uploader("ارفع الملف", type=["pdf", "docx", "txt", "pptx", "ppt"])

if file:

    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != file.name:

        with st.spinner("جار معالجة الملف..."):
            chunks = process_uploaded_file(file)
            if sync_to_pinecone(vector_db, chunks, embed_model):
                st.session_state.last_uploaded = file.name
                st.success("تم معالجة الملف بنجاح!")
            else:
                st.error("خطأ في معالجة الملف!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اسألني عن محتوى الملف..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching..."):
        query_vector = embed_model.embed_query(prompt)
        search_results = vector_db.query(
            namespace="documind_ns", 
            vector=query_vector, 
            top_k=3, 
            include_metadata=True
        )
        context_text = "\n".join([item['metadata']['text'] for item in search_results['matches']])

    with st.chat_message("assistant"):
        try:
            ai_response = client.chat.completions.create(

                model="google/gemini-2.5-flash-lite",
                messages=[
                    {
                        "role": "system", 
                        "content": (

                            "You are a professional document analyzer. "
                            "Use the provided Context to answer. "
                            "1. Respond in the same language as the user's question (Arabic or English). "
                            "2. If the question is in Arabic, use formal Arabic. "
                            "3. Maintain technical terms in their original language if necessary. "
                            "4. If the user gave you somthing not in the file search about something very close to it in the file and use it as a context to answer. "
                            f"\n\nContext: {context_text}"
                        )
                    },

                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            final_text = ai_response.choices[0].message.content
            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
        except Exception as e:
            st.error(f"Error: {e}")