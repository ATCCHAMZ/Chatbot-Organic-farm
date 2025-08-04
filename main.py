import os
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from groq import Groq
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("business-chatbot")

# ---------- Startup checks ----------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("Missing environment variable 'GROQ_API_KEY'")
    raise RuntimeError("GROQ_API_KEY must be set in the environment.")

# Optional: embed this app with Retry strategy or a pool if needed
client = Groq(api_key=GROQ_API_KEY)

# IMPORTANT: If you originally built the 'zimfresh_db' store using a specific embedding model
# then you should pass that same embedding_function here. For most persistent builds, this isn't needed.
vectorstore = Chroma(persist_directory="zimfresh_db")

class Query(BaseModel):
    question: str

def groq_llm_call(prompt: str) -> str:
    """
    Sends the prompt to Groq LLM and returns the assistant's reply.
    On API key failure or quota exceedance, it will raise an Exception.
    """
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful business assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,    # Using 0.0 for deterministic behavior
        max_tokens=None     # Let Groq decide optimal token usage
    )
    return resp.choices[0].message.content

@app.post("/ask")
async def ask_question(query: Query):
    logger.info("Received /ask request: %r", query.question)
    try:
        docs = vectorstore.similarity_search(query.question, k=3)
    except Exception:
        logger.exception("Vector similarity search failed")
        raise HTTPException(status_code=500, detail="Internal vector store error")

    if not docs:
        logger.warning("No relevant documents found for question: %s", query.question)
        return {"answer": "I'm not sure—please try asking something else."}

    # Join with visible breaklines so user knows each chunk
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = f"Answer using ONLY the context below:\n\n{context}\n\nQuestion: {query.question}\nAnswer:"
    logger.debug("Prompt for LLM:\n%s", prompt)

    try:
        answer = groq_llm_call(prompt)
    except Exception:
        logger.exception("Groq LLM request failed")
        raise HTTPException(status_code=502, detail="LLM service unavailable or invalid API key")

    return {"answer": answer}

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    incoming = form.get("Body", "").strip()
    logger.info("Incoming WhatsApp message: %r", incoming)

    if not incoming:
        reply = "Sorry, I didn't catch that."
    else:
        try:
            docs = vectorstore.similarity_search(incoming, k=3)
        except Exception:
            logger.exception("WhatsApp vector search failed")
            reply = "Apologies, technical issue retrieving data."
        else:
            if docs:
                context = "\n\n---\n\n".join(d.page_content for d in docs)
                prompt = f"Answer using ONLY the context below:\n\n{context}\n\nQuestion: {incoming}\nAnswer:"
                try:
                    reply = groq_llm_call(prompt)
                except Exception:
                    logger.exception("Groq call failed for WhatsApp")
                    reply = "Sorry, something went wrong with our AI service."
            else:
                logger.warning("No docs found for WhatsApp message")
                reply = "I’m not sure—maybe try asking in a different way?"

    xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<Response><Message>{reply}</Message></Response>"""
    return PlainTextResponse(content=xml, media_type="application/xml")

@app.get("/")
def health_check():
    return {"message": "Business Chatbot is live!"}
