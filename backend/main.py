import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai

from database import init_pinecone, sync_to_pinecone, load_embedding_model
from utils import process_uploaded_file

load_dotenv()

app = FastAPI(title="Fahim AI Backend")

# Enable CORS for Stitch Frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = None
if OPENROUTER_API_KEY:
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

embedding_model = load_embedding_model()
vector_db = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)

def success_response(answer: str) -> JSONResponse:
    return JSONResponse(content={"answer": answer})

def error_response(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"answer": message})

def build_fallback_answer(prompt: str, context_text: str) -> str:
    if context_text:
        return f"AI service is currently unavailable. Relevant context for '{prompt}':\n\n{context_text[:1200]}"

    return f"AI service is currently unavailable, and no indexed document context was found for: {prompt}"

def get_context_text(prompt: str) -> str:
    if vector_db is None:
        return ""

    query_vector = embedding_model.embed_query(prompt)
    search_results = vector_db.query(
        namespace="documind_ns",
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    matches = search_results.get("matches", [])
    return "\n".join(
        item.get("metadata", {}).get("text", "")
        for item in matches
        if item.get("metadata", {}).get("text")
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return error_response("Uploaded file is empty.", 400)

        chunks = process_uploaded_file(file_bytes, filename=file.filename)

        upload_ok = sync_to_pinecone(
            index=vector_db, 
            chunks=chunks, 
            model=embedding_model, 
            file_id=file.filename
        )

        if upload_ok:
            return success_response(f"File '{file.filename}' uploaded and indexed successfully.")

        return error_response("Failed to index document.", 500)
    except Exception as e:
        return error_response(str(e), 500)

@app.post("/chat")
async def chat_with_doc(prompt: str = Form(...)):
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return error_response("Prompt is required.", 400)

    try:
        context_text = get_context_text(cleaned_prompt)

        if client is None:
            return success_response(build_fallback_answer(cleaned_prompt, context_text))

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
                {"role": "user", "content": cleaned_prompt}
            ],
            temperature=0.3
        )

        answer = ai_response.choices[0].message.content or "The AI returned an empty response."
        return success_response(answer)

    except Exception as e:
        return success_response(build_fallback_answer(cleaned_prompt, locals().get("context_text", "")))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
