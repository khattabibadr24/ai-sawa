from agents.agents import analyze_user_intention, sawa_agent
from core.tools import get_history, save_interaction, search_qdrant

def process_user_message(user_input: str, retriever, session_id: str = None) -> str:
    # Step 1: Get chat history from MongoDB
    chat_history = get_history(session_id, limit=5) if session_id else []
    print(f"CHAT HISTORY: {chat_history}")
    
    # Step 2: Analyze user intention
    intention_result = analyze_user_intention(user_input, chat_history)
    
    # Step 3: Generate response
    if intention_result.type == "direct_response":
        print("[DIRECT RESPONSE]")
        answer = intention_result.content
    elif intention_result.type == "sawa_agent":
        print("[SAWA AGENT]")
        query = intention_result.content
        context_docs, doc_count = search_qdrant(query, retriever)
        print(f"QDRANT RESULT: ({context_docs}, {doc_count})")
        answer = sawa_agent(query, context_docs, chat_history)
    else:
        answer = "Je ne comprends pas votre demande. Pouvez-vous la reformuler ?"
    
    # Step 4: Save interaction to MongoDB
    if session_id:
        save_interaction(session_id, user_input, answer)
    
    return answer
