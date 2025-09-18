import os
import re
import json
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict
import time

import streamlit as st
import httpx
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from vector_db import load_vector_db  # import relatif

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SAWACARE AI CareManager",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour le design moderne
st.markdown("""
<style>
    /* Style général */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Messages de chat */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    
    .system-message {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        font-style: italic;
    }
    
    /* Sidebar */
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Chat input */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
    }
    
    /* Métriques */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration et initialisation ---
@st.cache_resource
def init_config():
    """Initialise la configuration depuis le .env"""
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ENV_PATH = PROJECT_ROOT / ".env"
    load_dotenv(ENV_PATH, override=True)
    
    return {
        'API_KEY': os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY"),
        'MODEL_NAME': os.getenv("MODEL_NAME", "mistral-small-latest"),
        'K': 3,
        'TEMPERATURE': 0.0,
        'MAX_CONCURRENCY': int(os.getenv("MAX_CONCURRENCY", "4")),
        'MAX_TURNS': int(os.getenv("MAX_TURNS", "8")),
        'FALLBACK_MODELS': [
            os.getenv("MODEL_NAME", "mistral-small-latest"),
            "mistral-small", 
            "mistral-tiny"
        ]
    }

config = init_config()

# === Réponses fixes EXACTES ===
SALUTATION_REPLY = "Bonjour, comment puis-je vous aider ?"
GOODBYE_REPLY = "Merci au revoir"
WHOAMI_REPLY = (
    "Bonjour, je suis votre SAWACARE AI CareManager, votre assistant intelligent dédié à l'aidance. "
    "Je combine technologie et bienveillance pour vous aider à mieux gérer vos responsabilités d'aidant, "
    "à trouver rapidement des réponses fiables et à bénéficier d'un accompagnement continu et personnalisé."
)

# Détection (priorité : AU REVOIR > QUI ÊTES-VOUS > SALUTATIONS)
GOODBYE_RE = re.compile(r"\b(au revoir|bonne (journée|soirée|nuit)|à bientôt|à plus|merci|bye|ciao|tchao|see you|take care)\b", re.I)
WHOAMI_RE  = re.compile(r"(vous ?êtes qui|qui ?êtes[- ]?vous|t.?es qui|tu es qui|c.?est quoi ce bot|who are you)", re.I)
HELLO_RE   = re.compile(r"^\s*(bonjour|salut|hello|hi|hey|coucou|salam|bonsoir)\b", re.I)

def shortcut_reply(user_text: str) -> Optional[str]:
    """Vérifie si le message correspond à un raccourci"""
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

# ---- Helpers ----
def require_api_key():
    """Vérifie la présence de la clé API"""
    if not config['API_KEY']:
        st.error("❌ Clé Mistral absente. Définis API_KEY ou MISTRAL_API_KEY dans le .env.")
        st.stop()

def format_docs(docs) -> str:
    """Formate les documents récupérés"""
    return "\n\n".join(d.page_content for d in docs)

def to_messages(raw_hist: Optional[List[Dict]], keep_last: int = None) -> List[BaseMessage]:
    """Convertit l'historique en messages LangChain"""
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

# ---- Vector store & retriever ----
@st.cache_resource
def init_vector_db():
    """Initialise la base de données vectorielle"""
    try:
        vector_db = load_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": config['K']})
        return retriever
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la base vectorielle: {e}")
        st.stop()

retriever = init_vector_db()

# ---- Prompt principal ----
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

# ---- LLM factory ----
@st.cache_resource
def make_llm(model_name: str) -> ChatMistralAI:
    """Crée un objet LLM Mistral"""
    return ChatMistralAI(
        mistral_api_key=config['API_KEY'],
        model=model_name,
        temperature=config['TEMPERATURE'],
        max_retries=1,
    )

# --- Question rewriter ---
def build_question_rewriter(model_name: str):
    """Construit le réécrivan de questions"""
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

# ---- Chaîne RAG ----
def build_chain_for(model_name: str):
    """Construit la chaîne RAG pour un modèle donné"""
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
    """Génère une réponse avec système de fallback"""
    require_api_key()
    last_err = None
    
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
                st.error("🔑 Appel Mistral refusé (401/403). Vérifie la clé API et les droits du modèle.")
                st.stop()
            st.error(f"🚨 Erreur Mistral ({code}).")
            st.stop()
        except Exception as e:
            last_err = e
            continue
    
    st.error("🚫 Le service de génération est temporairement indisponible (fallback épuisé).")
    st.stop()

# ---- Interface Streamlit ----
def main():
    # Header principal
    st.markdown("""
        <div class="main-header">
            <h1>🏥 SAWACARE AI CareManager</h1>
            <p>Votre assistant intelligent dédié à l'aidance</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar avec informations
    with st.sidebar:
        st.markdown("### 📊 Configuration")
        st.markdown(f"""
        <div class="sidebar-content">
            <strong>Modèle:</strong> {config['MODEL_NAME']}<br>
            <strong>Documents K:</strong> {config['K']}<br>
            <strong>Température:</strong> {config['TEMPERATURE']}<br>
            <strong>Max tours:</strong> {config['MAX_TURNS']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Raccourcis")
        st.markdown("""
        <div class="sidebar-content">
            • <strong>Salutations:</strong> bonjour, salut, hello<br>
            • <strong>Qui êtes-vous:</strong> qui êtes-vous, you are<br>
            • <strong>Au revoir:</strong> au revoir, bye, merci
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour vider l'historique
        if st.button("🗑️ Vider l'historique", use_container_width=True):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.rerun()
        
        # Statistiques de session
        if 'messages' in st.session_state and st.session_state.messages:
            msg_count = len(st.session_state.messages)
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            st.markdown("### 📈 Statistiques")
            st.markdown(f"""
            <div class="sidebar-content">
                <strong>Total messages:</strong> {msg_count}<br>
                <strong>Vos messages:</strong> {user_msgs}<br>
                <strong>Mes réponses:</strong> {assistant_msgs}
            </div>
            """, unsafe_allow_html=True)

    # Initialisation de l'état de session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique des messages
    st.markdown("### 💬 Conversation")
    
    # Container pour les messages avec scroll
    messages_container = st.container()
    
    with messages_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 Vous:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Sawa:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

    # Zone de saisie fixée en bas
    st.markdown("---")
    
    # Utilisation de colonnes pour le layout du chat
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Posez votre question...",
            key="user_input",
            placeholder="Tapez votre message ici...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Envoyer", use_container_width=True)

    # Traitement du message
    if send_button and user_input:
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Vérifier les raccourcis
        fixed_response = shortcut_reply(user_input)
        
        if fixed_response:
            # Réponse raccourci
            st.session_state.messages.append({"role": "assistant", "content": fixed_response})
            
            # Afficher la réponse immédiatement
            st.markdown(f"""
            <div class="system-message">
                <strong>⚡ Réponse automatique:</strong><br>
                {fixed_response}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Réponse RAG avec spinner
            with st.spinner("🤔 Je réfléchis..."):
                try:
                    # Convertir l'historique
                    chat_history = to_messages([
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages[:-1]  # Exclure le dernier message utilisateur
                    ])
                    
                    # Préparer le payload
                    payload = {
                        "question": user_input,
                        "chat_history": chat_history
                    }
                    
                    # Générer la réponse
                    response = generate_with_fallback(payload)
                    
                    # Ajouter la réponse à l'historique
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Success message temporaire
                    success_placeholder = st.success("✅ Réponse générée!")
                    time.sleep(1)
                    success_placeholder.empty()
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération: {str(e)}")
        
        # Rerun pour afficher le nouveau message
        st.rerun()



if __name__ == "__main__":
    main()