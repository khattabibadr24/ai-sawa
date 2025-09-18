import os
import re
import json
import time
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, List, Dict
from collections import defaultdict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from vector_db import load_vector_db  # import relatif

# ---------------------------------------
# Logging
# ---------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sawacare-api")

# ---------------------------------------
# Regex & Réponses fixes
# ---------------------------------------
SALUTATION_REPLY = "Bonjour, comment puis-je vous aider ?"
GOODBYE_REPLY = "Merci au revoir"
WHOAMI_REPLY = (
    "Bonjour, je suis votre SAWACARE AI CareManager, votre assistant intelligent dédié à l'aidance. "
    "Je combine technologie et bienveillance pour vous aider à mieux gérer vos responsabilités d'aidant, "
    "à trouver rapidement des réponses fiables et à bénéficier d'un accompagnement continu et personnalisé."
)

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

# ---------------------------------------
# Config & Initialisation
# ---------------------------------------
@lru_cache(maxsize=1)
def init_config() -> Dict:
    """Charge la config depuis .env (à la racine du projet)"""
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ENV_PATH = PROJECT_ROOT / ".env"
    load_dotenv(ENV_PATH, override=True)
    return {
        'API_KEY': os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY"),
        'MODEL_NAME': os.getenv("MODEL_NAME", "mistral-small-latest"),
        'K': 3,
        'TEMPERATURE': float(os.getenv("TEMPERATURE", "0.0")),
        'MAX_CONCURRENCY': int(os.getenv("MAX_CONCURRENCY", "4")),
        'MAX_TURNS': int(os.getenv("MAX_TURNS", "8")),
        'FALLBACK_MODELS': [
            os.getenv("MODEL_NAME", "mistral-small-latest"),
            "mistral-small",
            "mistral-tiny"
        ]
    }

config = init_config()

def require_api_key():
    if not config['API_KEY']:
        raise HTTPException(status_code=500, detail="Clé Mistral absente. Définis API_KEY ou MISTRAL_API_KEY dans le .env.")

@lru_cache(maxsize=1)
def get_retriever():
    """Initialise le retriever (Qdrant)"""
    try:
        vector_db = load_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": config['K']})
        return retriever
    except Exception as e:
        log.exception("Erreur lors du chargement de la base vectorielle")
        raise HTTPException(status_code=500, detail=f"Erreur base vectorielle: {e}")

retriever = get_retriever()

# ---------------------------------------
# Prompt principal
# ---------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es Sawa, l'assistant virtuel de l'application Sawa. "
     "Ta mission est d'aider les personnes à simplifier leurs tâches et à les guider pas à pas. "
     "Réponds STRICTEMENT à partir du CONTEXTE fourni (issu du fichier de données / base). "
     "Si l'information n'est pas dans le contexte ou est insuffisante, dis clairement : "
     "« Je n'ai pas assez d'informations dans le contexte pour répondre. » "
     "N'invente jamais. "
     "Analyse soigneusement la question de l'utilisateur et réponds précisément en t'appuyant sur le contexte disponible. "
     "Assure une continuité de discussion en tenant compte de l'historique (notamment le dernier échange) lorsqu'il est fourni. "
     "Quand il y a du contexte pertinent, termine toujours par 1–2 questions de suivi pour faciliter la poursuite de l'échange. "
     "N'utilise PAS de formules de politesse (ex. « Bonjour », « Merci », « Cordialement ») à chaque message : garde un ton respectueux et direct. "
     "N'emploie des salutations que si l'utilisateur en utilise explicitement ou pour clore définitivement la conversation. "
     "Ne donne surtout pas d'avis médical personnalisé."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Contexte (extraits de la base) :\n{context}\n\nQuestion : {question}")
])

# ---------------------------------------
# LLM factory / Chains
# ---------------------------------------
def make_llm(model_name: str) -> ChatMistralAI:
    return ChatMistralAI(
        mistral_api_key=config['API_KEY'],
        model=model_name,
        temperature=config['TEMPERATURE'],
        max_retries=1,
    )

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def to_messages(raw_hist: Optional[List[Dict]], keep_last: Optional[int] = None) -> List[BaseMessage]:
    if keep_last is None:
        keep_last = config['MAX_TURNS']
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

def build_question_rewriter(model_name: str):
    llm_local = make_llm(model_name)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tu reformules la dernière question utilisateur en une question autonome, "
         "en t'appuyant sur l'historique pour lever les pronoms, ellipses et références implicites. "
         "Si l'historique est vide, renvoie la question telle quelle. "
         "Réponds UNIQUEMENT par la question réécrite, sans explication."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    return rewrite_prompt | llm_local | StrOutputParser()

def build_chain_for(model_name: str):
    llm_local = make_llm(model_name)
    rewriter = build_question_rewriter(model_name)
    return (
        {
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | RunnablePassthrough.assign(
            standalone_q=RunnableLambda(lambda x: {
                "question": x["question"],
                "chat_history": x["chat_history"],
            }) | rewriter
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["standalone_q"]) | retriever | RunnableLambda(format_docs)
        )
        | prompt_template
        | llm_local
        | StrOutputParser()
    )

def generate_with_fallback(payload: Dict) -> str:
    require_api_key()
    last_err: Optional[Exception] = None
    for m in config['FALLBACK_MODELS']:
        try:
            chain = build_chain_for(m)
            return chain.invoke(payload)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code if getattr(e, "response", None) else None
            if code in (429, 500, 502, 503, 504):
                last_err = e
                continue
            if code in (401, 403):
                raise HTTPException(status_code=401, detail="Appel Mistral refusé (401/403). Vérifie la clé API et les droits du modèle.")
            raise HTTPException(status_code=502, detail=f"Erreur Mistral ({code}).")
        except Exception as e:
            # Essaye le modèle suivant
            last_err = e
            continue
    log.exception("Fallback épuisé", exc_info=last_err)
    raise HTTPException(status_code=503, detail="Le service de génération est temporairement indisponible (fallback épuisé).")

# ---------------------------------------
# FastAPI App
# ---------------------------------------
app = FastAPI(title="SAWACARE AI CareManager API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Schémas Pydantic
# ---------------------------------------
class Msg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|human|ai|bot)$")
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Msg]] = None

class ChatResponse(BaseModel):
    response: str
    history: List[Msg]
    used_fallback_model: Optional[str] = None

# ---------------------------------------
# Routes
# ---------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/config")
def get_config():
    cfg = dict(config)
    cfg.pop("API_KEY", None)
    return cfg

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint de chat RAG.
    L'historique est géré directement via le champ 'history' dans la requête.
    """
    # 1) Raccourcis
    fixed = shortcut_reply(req.message)
    if fixed:
        # Prépare l'historique de sortie
        hist_in = [m.dict() for m in (req.history or [])]
        hist_out_dicts = hist_in + [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": fixed}
        ]
        # Limite l'historique
        hist_out = [Msg(**h) for h in hist_out_dicts][-2*config['MAX_TURNS']:]
        
        return ChatResponse(
            response=fixed,
            history=hist_out,
            used_fallback_model=None
        )

    # 2) Prépare l'historique pour LangChain
    lc_history = to_messages([m.dict() for m in (req.history or [])])

    # 3) Payload chaîne
    payload = {
        "question": req.message,
        "chat_history": lc_history
    }

    # 4) Génération
    resp_text = generate_with_fallback(payload)

    # 5) Prépare l'historique de sortie
    hist_in = [m.dict() for m in (req.history or [])]
    hist_out_dicts = hist_in + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": resp_text}
    ]
    # Limite l'historique à MAX_TURNS paires (approx)
    hist_out = [Msg(**h) for h in hist_out_dicts][-2*config['MAX_TURNS']:]

    return ChatResponse(
        response=resp_text,
        history=hist_out,
        used_fallback_model=None
    )

# ---------------------------------------
# Exécution locale
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)