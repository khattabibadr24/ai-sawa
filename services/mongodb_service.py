from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from core.config import MONGODB_URL, MONGODB_DATABASE


class MongoDBHistoryService:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection_name = "chat_history"
        self._connect()

    def _connect(self):
        if self.client is None:
            mongo_url = MONGODB_URL
            self.client = MongoClient(mongo_url)
            db_name = MONGODB_DATABASE 
            self.db = self.client[db_name]

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    def save_interaction(self, session_id: str, user_message: str, ai_message: str):
        try:
            interaction = {
                "session_id": session_id,
                "user_message": user_message,
                "ai_message": ai_message,
                "timestamp": datetime.utcnow(),
            }
            self.db[self.collection_name].insert_one(interaction)
        except Exception as e:
            print(f"Error saving interaction to MongoDB: {e}")

    def get_history(self, session_id: str, limit: int = 5) -> List[BaseMessage]:
        try:
            interactions = list(
                self.db[self.collection_name]
                .find({"session_id": session_id})
                .sort("timestamp", DESCENDING)
                .limit(limit)
            )

            # Convert to LangChain messages (reverse to chronological order)
            messages: List[BaseMessage] = []
            for interaction in reversed(interactions):
                user_msg = interaction.get("user_message")
                ai_msg = interaction.get("ai_message")
                if user_msg is not None:
                    messages.append(HumanMessage(content=user_msg))
                if ai_msg is not None:
                    messages.append(AIMessage(content=ai_msg))

            return messages
        except Exception as e:
            print(f"Error retrieving history from MongoDB: {e}")
            return []
