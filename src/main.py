# src/main.py — FastAPI sans sources dans la réponse
import os
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse, JSONResponse

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from vector_db import load_vector_db

# --- .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

# === Hyperparamètres FIXES ===
K = 3
TEMPERATURE = 0.2

# ---- helpers ----
def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

# ---- Vector store & chaîne RAG ----
vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": K})

llm = ChatMistralAI(
    mistral_api_key=API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "Vous êtes un assistant médical utile et courtois. Répondez aux questions de l'utilisateur "
     "en vous basant uniquement sur le contexte fourni. Si la réponse ne se trouve pas dans le contexte, "
     "indiquez-le poliment. Après avoir répondu, proposez 1–2 questions de suivi pertinentes. "
     "Commencez par une formule de politesse et terminez avec une formule de courtoisie. "
     "Ne donnez pas d'avis médical personnalisé."),
    ("human", "Contexte:\n{context}\n\nQuestion: {question}")
])

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# ---- FastAPI app ----
app = FastAPI(title="Chatbot Médical (FastAPI)", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
    answer = rag_chain.invoke(q)
    return ChatResponse(answer=answer)

@app.get("/chat/stream")
async def chat_stream(question: str = Query(..., description="Question utilisateur")):
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        async for chunk in rag_chain.astream(q):
            yield f"data: {chunk}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/")
def root():
    return JSONResponse(
        {
            "name": "Chatbot Médical (FastAPI)",
            "endpoints": {
                "health": "/health",
                "chat (POST)": "/chat",
                "chat stream (GET, SSE)": "/chat/stream?question=...",
            },
            "params": {"K": K, "TEMPERATURE": TEMPERATURE, "MODEL_NAME": MODEL_NAME},
        }
    )