from typing import List, Tuple
from langchain_core.messages import BaseMessage

from services.mongodb_service import MongoDBHistoryService

mongodb_service = MongoDBHistoryService()


def save_interaction(session_id: str, user_message: str, ai_message: str):
    if session_id:
        mongodb_service.save_interaction(session_id, user_message, ai_message)


def get_history(session_id: str, limit: int = 5) -> List[BaseMessage]:
    if not session_id:
        return []
    return mongodb_service.get_history(session_id, limit)

def search_qdrant(query: str, retriever, k: int = 3) -> Tuple[List[str], int]:
    try:
        docs = retriever.invoke(query)
        documents = [doc.page_content for doc in docs]
        return documents, len(documents)
    except Exception:
        return [], 0
