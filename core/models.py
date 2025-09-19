from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class ChatHistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatHistoryItem]] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

class IntentionResponse(BaseModel):
    type: Literal["direct_response", "sawa_agent"] = Field(
        description="Type of response - either direct response or SAWA agent processing needed"
    )
    content: str = Field(
        description="Either the direct response text or the query for SAWA agent"
    )

