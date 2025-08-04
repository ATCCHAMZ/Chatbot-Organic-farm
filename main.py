import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI app must be at the top ---
app = FastAPI(title="Business Chatbot API")

# --- Environment variable check ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in environment.")

# --- Embeddings (must match the one used when creating vector store) ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load persistent vector store ---
vectorstore = Chroma(
    persist_directory="zimfresh_db",
    embedding_function=embeddings
)

# --- Groq client ---
groq_client = Groq(api_key=GROQ_API_KEY)

class Query(BaseModel):
    question: str

def groq_llm_call(prompt: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful business assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.exception("Groq LLM call failed")
        raise HTTPException(status_code=502, detail=f"Groq LLM error: {str(e)}")

@app.post("/ask")
async def ask_question(query: Query):
    try:
        docs = vectorstore.similarity_search(query.question, k=3)
    except Exception:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=500, detail="Vector DB error")

    if not docs:
        return {"answer": "Sorry, I don't have information on that."}

    context = "\n".join(d.page_content for d in docs)
    prompt = f"""
Answer the question using ONLY the context below:
{context}
Question: {query.question}
"""
    answer = groq_llm_call(prompt)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Business Chatbot is live!"}
