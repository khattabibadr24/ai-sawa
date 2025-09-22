from typing import AsyncGenerator, Optional
import uuid

from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse

from core.models import ChatRequest, ChatResponse
from services.vector_db_service import get_retriever
from services.sawa_service import process_user_message

# Initialize router
router = APIRouter()

# Lazy-loaded retriever
def get_retriever_instance():
    return get_retriever()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Generate session_id if not provided
    session_id = req.session_id or str(uuid.uuid4())

    answer = process_user_message(q, get_retriever_instance(), session_id)

    return ChatResponse(answer=answer)

@router.get("/chat/stream")
async def chat_stream(
    question: str = Query(..., description="Question utilisateur"),
    session_id: Optional[str] = Query(None, description="Identifiant de session pour mémoire serveur"),
):
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Generate session_id if not provided
    session_id = session_id or str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        try:
            answer = process_user_message(q, get_retriever_instance(), session_id)
            
            # Stream the response (for now, we'll send it all at once)
            # TODO: Implement proper streaming with the new agent system
            yield f"data: {answer}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
                
        except Exception as e:
            error_msg = f"Erreur lors du traitement: {str(e)}"
            yield f"data: [ERROR] {error_msg}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
