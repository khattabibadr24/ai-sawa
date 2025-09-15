import os
import re
import json
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse, JSONResponse

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from vector_db import load_vector_db  # import relatif

# --- .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

# Lis les deux variantes possibles
API_KEY = os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

# === Hyperparamètres FIXES ===
K = 3
TEMPERATURE = 0.0

# Concurrence & fallback
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
FALLBACK_MODELS = [MODEL_NAME, "mistral-small", "mistral-tiny"]

# Fenêtre de mémoire locale (serveur)
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))  # nombre de tours (user+assistant) conservés

# === Réponses fixes EXACTES ===
SALUTATION_REPLY = "Bonjour, comment puis-je vous aider ?"
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
def require_api_key():
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Clé Mistral absente. Définis API_KEY ou MISTRAL_API_KEY dans le .env."
        )

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def to_messages(raw_hist: Optional[List[Dict]], keep_last: int = MAX_TURNS) -> List[BaseMessage]:
    """
    Convertit une liste [{role:'user'|'ai'|'assistant', content:'...'}, ...]
    en messages LangChain. Ne garde que les `keep_last` derniers messages.
    """
    msgs: List[BaseMessage] = []
    for m in (raw_hist or []):
        role = (m.get("role") or "").strip().lower()
        content = m.get("content", "")
        if not content:
            continue
        if role in ("user", "human"):
            msgs.append(HumanMessage(content=content))
        elif role in ("ai", "assistant", "bot"):
            msgs.append(AIMessage(content=content))
    return msgs[-keep_last:]


# ---- Vector store & retriever ----
vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": K})


# ---- Prompt principal ----
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es Sawa, l’assistant virtuel de l’application Sawa. "
     "Ta mission est d’aider les personnes à simplifier leurs tâches et à les guider pas à pas. "
     "Réponds STRICTEMENT à partir du CONTEXTE fourni (issu du fichier de données / base). "
     "Si l’information n’est pas dans le contexte ou est insuffisante, dis clairement : "
     "« Je n’ai pas assez d’informations dans le contexte pour répondre. » "
     "N’invente jamais. "
     "Analyse soigneusement la question de l’utilisateur et réponds précisément en t’appuyant sur le contexte disponible. "
     "Assure une continuité de discussion en tenant compte de l’historique (notamment le dernier échange) lorsqu’il est fourni. "
     "Quand il y a du contexte pertinent, termine toujours par 1–2 questions de suivi pour faciliter la poursuite de l’échange. "
     "N’utilise PAS de formules de politesse (ex. « Bonjour », « Merci », « Cordialement ») à chaque message : garde un ton respectueux et direct. "
     "N’emploie des salutations que si l’utilisateur en utilise explicitement ou pour clore définitivement la conversation. "
     "Ne donne pas d’avis médical personnalisé."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Contexte (extraits de la base) :\n{context}\n\nQuestion : {question}")
])


# ---- LLM factory ----
def make_llm(model_name: str) -> ChatMistralAI:
    return ChatMistralAI(
        mistral_api_key=API_KEY,
        model=model_name,
        temperature=TEMPERATURE,
        max_retries=1,  # on gère nous-mêmes le fallback
    )


# --- NEW: question rewriter (history-aware) ---
def build_question_rewriter(model_name: str):
    llm_local = make_llm(model_name)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu reformules la dernière question utilisateur en une question autonome, "
         "en t’appuyant sur l'historique pour lever les pronoms, ellipses et références implicites. "
         "Si l'historique est vide, renvoie la question telle quelle. "
         "Réponds UNIQUEMENT par la question réécrite, sans explication."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    return rewrite_prompt | llm_local | StrOutputParser()


# ---- Chaîne RAG (constructeur par modèle pour fallback) ----
def build_chain_for(model_name: str):
    llm_local = make_llm(model_name)
    rewriter = build_question_rewriter(model_name)

    # Pipeline:
    # 1) standalone_q = rewriter(chat_history, question)
    # 2) context = retriever(standalone_q)
    # 3) prompt(system + chat_history + human(question, context)) -> llm
    return (
        {
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        # standalone_q = rewriter(chat_history, question)
        | RunnablePassthrough.assign(
            standalone_q=RunnableLambda(lambda x: {
                "question": x["question"],
                "chat_history": x["chat_history"],
            }) | rewriter
        )
        # context = retriever(standalone_q)
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["standalone_q"]) | retriever | RunnableLambda(format_docs)
        )
        # prompt final = system + chat_history + human(question + context)
        | prompt_template
        | llm_local
        | StrOutputParser()
    )


def generate_with_fallback(payload: Dict) -> str:
    """
    payload = {"question": str, "chat_history": List[BaseMessage]}
    """
    require_api_key()
    last_err = None
    for m in FALLBACK_MODELS:
        try:
            chain = build_chain_for(m)
            return chain.invoke(payload)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code if getattr(e, "response", None) else None
            # Tente les modèles suivants seulement sur 429/5xx
            if code in (429, 500, 502, 503, 504):
                last_err = e
                continue
            # Erreurs d'auth/autorisations → message clair
            if code in (401, 403):
                raise HTTPException(
                    status_code=502,
                    detail="Appel Mistral refusé (401/403). Vérifie la clé API et les droits du modèle."
                )
            # Autres erreurs HTTP → propage en 502
            raise HTTPException(status_code=502, detail=f"Erreur Mistral ({code}).")
        except Exception as e:
            # Erreurs réseau/transients : essaie modèle suivant
            last_err = e
            continue
    # Si on a épuisé tous les modèles
    raise HTTPException(
        status_code=503,
        detail="Le service de génération est temporairement indisponible (fallback épuisé)."
    )


# === Mémoire serveur optionnelle (par session_id) ===
SESSIONS: Dict[str, List[BaseMessage]] = defaultdict(list)

def get_server_history(session_id: Optional[str]) -> List[BaseMessage]:
    if not session_id:
        return []
    hist = SESSIONS.get(session_id, [])
    return hist[-MAX_TURNS:]

def append_server_history(session_id: Optional[str], user_msg: str, ai_msg: str):
    if not session_id:
        return
    SESSIONS[session_id].append(HumanMessage(content=user_msg))
    SESSIONS[session_id].append(AIMessage(content=ai_msg))
    # Trim
    if len(SESSIONS[session_id]) > 2 * MAX_TURNS:
        SESSIONS[session_id] = SESSIONS[session_id][-2 * MAX_TURNS:]


# ---- FastAPI app ----
app = FastAPI(title="Chatbot Médical (FastAPI)", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Schémas d'E/S ----
class ChatHistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatHistoryItem]] = None  # [{role, content}]
    session_id: Optional[str] = None  # NEW: mémoire côté serveur

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

    # Raccourcis
    fixed = shortcut_reply(q)
    if fixed is not None:
        append_server_history(req.session_id, q, fixed)
        return ChatResponse(answer=fixed)

    # Historique : priorité au serveur si session_id fourni, sinon on prend celui envoyé par le client
    server_hist = get_server_history(req.session_id)
    client_hist = to_messages([h.model_dump() for h in (req.chat_history or [])])
    msgs = server_hist if server_hist else client_hist

    payload = {"question": q, "chat_history": msgs}

    async with SEMAPHORE:
        answer = generate_with_fallback(payload)

    append_server_history(req.session_id, q, answer)
    return ChatResponse(answer=answer)


@app.get("/chat/stream")
async def chat_stream(
    question: str = Query(..., description="Question utilisateur"),
    chat_history: Optional[str] = Query(None, description="JSON minifié [{role,content},...]"),
    session_id: Optional[str] = Query(None, description="Identifiant de session pour mémoire serveur"),
):
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Raccourcis en streaming
    fixed = shortcut_reply(q)
    if fixed is not None:
        async def fixed_gen():
            yield f"data: {fixed}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        append_server_history(session_id, q, fixed)
        return StreamingResponse(fixed_gen(), media_type="text/event-stream")

    # Parse éventuel historique JSON depuis la query (usage test)
    raw_hist_list: Optional[List[Dict]] = None
    if chat_history:
        try:
            raw_hist_list = json.loads(chat_history)
            if not isinstance(raw_hist_list, list):
                raw_hist_list = None
        except json.JSONDecodeError:
            raw_hist_list = None

    server_hist = get_server_history(session_id)
    client_hist = to_messages(raw_hist_list or [])
    msgs = server_hist if server_hist else client_hist

    payload = {"question": q, "chat_history": msgs}

    async def event_gen() -> AsyncGenerator[bytes, None]:
        acc: List[str] = []
        async with SEMAPHORE:
            # Fallback manual en streaming, pour pouvoir mémoriser la réponse finale
            tried_any = False
            for m in FALLBACK_MODELS:
                tried_any = True
                try:
                    chain = build_chain_for(m)
                    for chunk in chain.stream(payload):
                        # stream() renvoie des morceaux de texte (str)
                        acc.append(chunk)
                        yield f"data: {chunk}\n\n".encode("utf-8")
                    full_answer = "".join(acc)
                    append_server_history(session_id, q, full_answer)
                    yield b"data: [DONE]\n\n"
                    return
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code if getattr(e, "response", None) else None
                    if code in (429, 500, 502, 503, 504):
                        # tente le modèle suivant
                        continue
                    msg = "Clé API invalide ou non autorisée." if code in (401, 403) else f"Erreur Mistral ({code})."
                    yield f"data: [ERROR] {msg}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                    return
                except Exception:
                    # tente le modèle suivant
                    continue

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/")
def root():
    return JSONResponse(
        {
            "name": "Chatbot Médical (FastAPI)",
            "version": "1.4.0",
            "endpoints": {
                "health": "/health",
                "chat (POST)": "/chat",
                "chat stream (GET, SSE)": "/chat/stream?question=...&session_id=...",
            },
            "params": {
                "K": K,
                "TEMPERATURE": TEMPERATURE,
                "MODEL_NAME": MODEL_NAME,
                "MAX_CONCURRENCY": MAX_CONCURRENCY,
                "MAX_TURNS": MAX_TURNS,
            },
        }
    )
