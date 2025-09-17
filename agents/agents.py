import re
import httpx
from typing import Optional, Dict
from fastapi import HTTPException

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.config import API_KEY, TEMPERATURE, FALLBACK_MODELS, require_api_key

# === Réponses fixes EXACTES ===
SALUTATION_REPLY = "Bonjour, comment puis-je vous aider ?"
GOODBYE_REPLY = "Merci au revoir"
WHOAMI_REPLY = (
    "Bonjour, je suis votre  SAWACARE AI CareManager, votre assistant intelligent dédié à l'aidance. "
    "Je combine technologie et bienveillance pour vous aider à mieux gérer vos responsabilités d'aidant, "
    "à trouver rapidement des réponses fiables et à bénéficier d'un accompagnement continu et personnalisé."
)

# Détection (priorité : AU REVOIR > QUI ÊTES-VOUS > SALUTATIONS)
GOODBYE_RE = re.compile(r"\b(au revoir|bonne (journée|soirée|nuit)|à bientôt|à plus|merci|bye|ciao|tchao|see you|take care)\b", re.I)
WHOAMI_RE = re.compile(r"(vous ?êtes qui|qui ?êtes[- ]?vous|t.?es qui|tu es qui|c.?est quoi ce bot|who are you)", re.I)
HELLO_RE = re.compile(r"^\s*(bonjour|salut|hello|hi|hey|coucou|salam|bonsoir)\b", re.I)

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

# --- LLM Factory ---
def make_llm(model_name: str) -> ChatMistralAI:
    return ChatMistralAI(
        mistral_api_key=API_KEY,
        model=model_name,
        temperature=TEMPERATURE,
        max_retries=1,  # on gère nous-mêmes le fallback
    )

# --- Document Formatting ---
def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

# ---- Prompt principal ----
def get_main_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
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
         "Ne donne pas d'avis médical personnalisé."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Contexte (extraits de la base) :\n{context}\n\nQuestion : {question}")
    ])

# --- NEW: question rewriter (history-aware) ---
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

# ---- Chaîne RAG (constructeur par modèle pour fallback) ----
def build_chain_for(model_name: str, retriever):
    llm_local = make_llm(model_name)
    rewriter = build_question_rewriter(model_name)
    prompt_template = get_main_prompt_template()

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

def generate_with_fallback(payload: Dict, retriever) -> str:
    """
    payload: {"question": str, "chat_history": List[BaseMessage]}
    retriever: Vector database retriever
    """
    require_api_key()
    last_err = None
    for m in FALLBACK_MODELS:
        try:
            chain = build_chain_for(m, retriever)
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
