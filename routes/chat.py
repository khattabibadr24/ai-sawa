import json
from typing import AsyncGenerator, Optional, List, Dict

import httpx
from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse

from core.config import SEMAPHORE, FALLBACK_MODELS, MAX_TURNS
from core.models import (
    ChatRequest, ChatResponse, to_messages, 
    get_server_history, append_server_history
)
from agents.agents import (
    shortcut_reply, build_chain_for, generate_with_fallback
)
from services.vector_db_service import get_retriever

# Initialize router
router = APIRouter()

# Initialize retriever
retriever = get_retriever()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Raccourcis
    fixed = shortcut_reply(q)
    if fixed is not None:
        append_server_history(req.session_id, q, fixed, MAX_TURNS)
        return ChatResponse(answer=fixed)

    # Historique : priorité au serveur si session_id fourni, sinon on prend celui envoyé par le client
    server_hist = get_server_history(req.session_id, MAX_TURNS)
    client_hist = to_messages([h.model_dump() for h in (req.chat_history or [])], MAX_TURNS)
    msgs = server_hist if server_hist else client_hist

    payload = {"question": q, "chat_history": msgs}

    async with SEMAPHORE:
        answer = generate_with_fallback(payload, retriever)

    append_server_history(req.session_id, q, answer, MAX_TURNS)
    return ChatResponse(answer=answer)

@router.get("/chat/stream")
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
        append_server_history(session_id, q, fixed, MAX_TURNS)
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

    server_hist = get_server_history(session_id, MAX_TURNS)
    client_hist = to_messages(raw_hist_list or [], MAX_TURNS)
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
                    chain = build_chain_for(m, retriever)
                    for chunk in chain.stream(payload):
                        # stream() renvoie des morceaux de texte (str)
                        acc.append(chunk)
                        yield f"data: {chunk}\n\n".encode("utf-8")
                    full_answer = "".join(acc)
                    append_server_history(session_id, q, full_answer, MAX_TURNS)
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
