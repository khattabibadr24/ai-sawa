from typing import Optional, Literal
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str

class ChatResponse(BaseModel):
    answer: str

class IntentionResponse(BaseModel):
    type: Literal["direct_response", "sawa_agent"] = Field(
        description="Type of response - either direct response or SAWA agent processing needed"
    )
    content: str = Field(
        description="Either the direct response text or the query for SAWA agent"
    )
