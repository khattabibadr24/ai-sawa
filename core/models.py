from typing import Optional, List, Dict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from collections import defaultdict

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

# === Mémoire serveur optionnelle (par session_id) ===
SESSIONS: Dict[str, List[BaseMessage]] = defaultdict(list)

def to_messages(raw_hist: Optional[List[Dict]], keep_last: int = 8) -> List[BaseMessage]:
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

def get_server_history(session_id: Optional[str], max_turns: int = 8) -> List[BaseMessage]:
    if not session_id:
        return []
    hist = SESSIONS.get(session_id, [])
    return hist[-max_turns:]

def append_server_history(session_id: Optional[str], user_msg: str, ai_msg: str, max_turns: int = 8):
    if not session_id:
        return
    SESSIONS[session_id].append(HumanMessage(content=user_msg))
    SESSIONS[session_id].append(AIMessage(content=ai_msg))
    # Trim
    if len(SESSIONS[session_id]) > 2 * max_turns:
        SESSIONS[session_id] = SESSIONS[session_id][-2 * max_turns:]
