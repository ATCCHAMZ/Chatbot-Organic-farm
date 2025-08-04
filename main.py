import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from langchain_community.vectorstores import Chroma

# Load environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Init FastAPI
app = FastAPI(title="Business Chatbot API")

# Load vector database (already persisted)
vectorstore = Chroma(persist_directory="zimfresh_db")

# Init Groq client
client = Groq(api_key=GROQ_API_KEY)

class Query(BaseModel):
    question: str

def groq_llm_call(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful business assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return completion.choices[0].message.content

@app.post("/ask")
async def ask_question(query: Query):
    # Search knowledge base
    docs = vectorstore.similarity_search(query.question, k=3)
    context = "\n".join([d.page_content for d in docs])

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
