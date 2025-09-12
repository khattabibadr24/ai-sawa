# src/main.py — FastAPI (RAG) avec routage salutations/au revoir/qui êtes-vous
import os
import re
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse, JSONResponse
from langchain_mistralai import MistralAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from vector_db import load_vector_db  # <= import relatif

# --- .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

# === Hyperparamètres FIXES ===
K = 3
TEMPERATURE = 0.0

# Limitation de concurrence & fallback modèles
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
FALLBACK_MODELS = [MODEL_NAME, "mistral-small", "mistral-tiny"]

# === Réponses fixes EXACTES ===
SALUTATION_REPLY = "Bonjour, comment puis je vous aidez ?"
GOODBYE_REPLY = "Merci au revoir"
WHOAMI_REPLY = (
    "Bonjour, je suis votre  SAWACARE AI CareManager, votre assistant intelligent dédié à l’aidance. "
    "Je combine technologie et bienveillance pour vous aider à mieux gérer vos responsabilités d’aidant, "
    "à trouver rapidement des réponses fiables et à bénéficier d’un accompagnement continu et personnalisé."
)

# Détection (priorité : AU REVOIR > QUI ÊTES-VOUS > SALUTATIONS)
GOODBYE_RE = re.compile(r"\b(au revoir|bonne (journée|soirée|nuit)|à bientôt|à plus|merci|bye|ciao|tchao|see you|take care)\b", re.I)
WHOAMI_RE  = re.compile(r"(vous ?êtes qui|qui ?êtes[- ]?vous|t.?es qui|tu es qui|c.?est quoi ce bot|who are you)", re.I)
HELLO_RE   = re.compile(r"^\s*(bonjour|salut|hello|hi|hey|coucou|salam|bonsoir)\b", re.I)

def shortcut_reply(user_text: str) -> Optional[str]:
    t = (user_text or "").strip()
    if not t:
        return None
    if GOODBYE_RE.search(t):
        return GOODBYE_REPLY
    if WHOAMI_RE.search(t):
        return WHOAMI_REPLY
    if HELLO_RE.search(t):
        return SALUTATION_REPLY
    return None

# ---- helpers ----
def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

# ---- Vector store & retriever ----
vector_db = load_vector_db()

# base: MMR pour éviter 3 chunks trop similaires
base_retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.7}
)

# filtre: retire les passages peu liés à la question
emb = MistralAIEmbeddings(mistral_api_key=API_KEY, model="mistral-embed")
compressor = EmbeddingsFilter(embeddings=emb, similarity_threshold=0.30)  # 0.25–0.35 est une bonne plage
retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)
# ---- Prompt (sans règles de salutation puisque gérées en amont) ----
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "Vous êtes un assistant médical utile et courtois. "
     "Répondez STRICTEMENT à partir du CONTEXTE fourni. "
     "Si l'information n'est pas dans le contexte ou est insuffisante, dites clairement : "
     "« Je n’ai pas assez d’informations dans le contexte pour répondre. » "
     "N’inventez jamais. "
     "Après avoir répondu (quand il y a du contexte pertinent), proposez 1–2 questions de suivi pertinentes. "
     "Commencez par une formule de politesse et terminez avec une formule de courtoisie. "
     "Ne donnez pas d’avis médical personnalisé."
    ),
    ("human", "Contexte:\n{context}\n\nQuestion: {question}")
])

# ---- Chaîne RAG (constructeur par modèle pour fallback) ----
def make_llm(model_name: str) -> ChatMistralAI:
    return ChatMistralAI(
        mistral_api_key=API_KEY,
        model=model_name,
        temperature=TEMPERATURE,
        top_p=1.0, 
        max_retries=1,  # on gère nous-mêmes le fallback
    )

def build_chain_for(model_name: str):
    llm_local = make_llm(model_name)
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm_local
        | StrOutputParser()
    )

def generate_with_fallback(question: str) -> str:
    last_err = None
    for m in FALLBACK_MODELS:
        try:
            chain = build_chain_for(m)
            return chain.invoke(question)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code in (429, 503):
                last_err = e
                continue
            raise
    raise HTTPException(
        status_code=503,
        detail="Le service de génération est temporairement saturé. Réessayez dans quelques instants.",
    )

async def sse_with_fallback(question: str):
    for m in FALLBACK_MODELS:
        try:
            chain = build_chain_for(m)
            for chunk in chain.stream(question):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
            return
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code in (429, 503):
                continue
            raise
    yield "data: [ERROR] Service saturé, réessayez plus tard.\n\n"
    yield "data: [DONE]\n\n"

# ---- FastAPI app ----
app = FastAPI(title="Chatbot Médical (FastAPI)", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre si besoin
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
async def chat(req: ChatRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Court-circuit salutation / au revoir / whoami
    fixed = shortcut_reply(q)
    if fixed is not None:
        return ChatResponse(answer=fixed)

    async with SEMAPHORE:
        answer = generate_with_fallback(q)
    return ChatResponse(answer=answer)

@app.get("/chat/stream")
async def chat_stream(question: str = Query(..., description="Question utilisateur")):
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Court-circuit en streaming
    fixed = shortcut_reply(q)
    if fixed is not None:
        async def fixed_gen():
            yield f"data: {fixed}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        return StreamingResponse(fixed_gen(), media_type="text/event-stream")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        async with SEMAPHORE:
            async for evt in sse_with_fallback(q):
                yield evt.encode("utf-8")
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
            "params": {
                "K": K,
                "TEMPERATURE": TEMPERATURE,
                "MODEL_NAME": MODEL_NAME,
                "MAX_CONCURRENCY": MAX_CONCURRENCY,
            },
        }
    )
