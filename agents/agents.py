from typing import List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from core.config import API_KEY, TEMPERATURE, FALLBACK_MODELS
from core.models import IntentionResponse
from core.tools import get_history, save_interaction, search_qdrant

def make_llm(model_name: str = None) -> ChatMistralAI:
    model = model_name or FALLBACK_MODELS[0]
    return ChatMistralAI(
        mistral_api_key=API_KEY,
        model=model,
        temperature=TEMPERATURE,
        max_retries=1,
    )

def analyze_user_intention(user_input: str, chat_history: List[BaseMessage]) -> IntentionResponse:
    try:
        llm = make_llm()
        
        structured_llm = llm.with_structured_output(IntentionResponse)
        
        prompt = ChatPromptTemplate.from_messages([
        ("system",
             """Tu es un assistant intelligent qui analyse les intentions des utilisateurs pour l'application SAWA dédiée à l'aidance.

                Ton rôle est de:
                1. Identifier si le message est une salutation, au revoir, ou question d'identité
                2. Fournir une réponse directe appropriée OU indiquer que le SAWA agent doit traiter la question

                Pour les SALUTATIONS (bonjour, salut, hello, hi, hey, coucou, salam, bonsoir):
                - Retourne: type="direct_response", content="Bonjour, comment puis-je vous aider ?"

                Pour les AU REVOIR (au revoir, bonne journée/soirée/nuit, à bientôt, à plus, merci, bye, ciao, tchao):
                - Retourne: type="direct_response", content="Merci au revoir"

                Pour les QUESTIONS D'IDENTITÉ (vous êtes qui, qui êtes-vous, t'es qui, tu es qui, c'est quoi ce bot):
                - Retourne: type="direct_response", content="Bonjour, je suis votre SAWACARE AI CareManager, votre assistant intelligent dédié à l'aidance. Je combine technologie et bienveillance pour vous aider à mieux gérer vos responsabilités d'aidant, à trouver rapidement des réponses fiables et à bénéficier d'un accompagnement continu et personnalisé."

                Pour TOUTES LES AUTRES QUESTIONS sur l'aidance, la santé, les soins, etc.:
                - Retourne: type="sawa_agent", content="[la question de l'utilisateur, reformulée si nécessaire pour être plus claire]"

                IMPORTANT: Tu dois TOUJOURS retourner une structure avec 'type' et 'content'."""),
        MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])
        
        result = (prompt | structured_llm).invoke({
            "user_input": user_input,
            "chat_history": chat_history or []
        })
        
        return result
        
    except Exception:
        return IntentionResponse(
            type="sawa_agent",
            content=user_input
        )

def sawa_agent(question: str, context_docs: list[str], chat_history: List[BaseMessage]) -> str:
    try:
        llm = make_llm()
        context = "\n\n".join(context_docs) if context_docs else "Aucun contexte pertinent trouvé."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Tu es Sawa, l'assistant virtuel de l'application Sawa dédié à l'aidance. "
             "Tu as accès aux 5 dernières interactions avec l'utilisateur pour comprendre le contexte de la conversation. "
             "Utilise cet historique pour:\n"
             "- Comprendre les références implicites (il, elle, ça, cette situation, etc.)\n"
             "- Maintenir la continuité de la conversation\n"
             "- Éviter de répéter des informations déjà données récemment\n"
             "- Personnaliser tes réponses selon le contexte établi\n\n"
             "RÈGLES STRICTES - TU DOIS LES RESPECTER ABSOLUMENT:\n"
             "1. RÉPONDS UNIQUEMENT à partir du CONTEXTE fourni et de l'HISTORIQUE de conversation\n"
             "2. Si le contexte est vide ou insuffisant, dis explicitement : 'Je n'ai pas d'informations suffisantes dans ma base de connaissances pour répondre à cette question.'\n"
             "3. N'utilise JAMAIS tes connaissances générales ou externes\n"
             "4. N'invente JAMAIS d'informations\n"
             "5. Ne fais PAS de suppositions au-delà du contexte fourni\n"
             "6. Si une partie de la question peut être répondue avec le contexte et une autre non, indique clairement quelle partie tu ne peux pas traiter\n"
             "7. Propose des questions de suivi UNIQUEMENT si elles sont pertinentes par rapport au contexte disponible\n"
             "8. Ne donne pas d'avis médical personnalisé\n\n"
             "Quand tu n'as pas assez d'informations, suggère à l'utilisateur de reformuler sa question ou de consulter un professionnel de santé si nécessaire."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", 
             "Question actuelle: {question}\n\n"
             "Contexte trouvé dans la base de connaissances:\n{context}"
            )
        ])
        
        result = (prompt | llm).invoke({
            "question": question,
            "context": context,
            "chat_history": chat_history or []
        })
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        print(f"ERROR in sawa_agent: {str(e)}")
        return "Je rencontre des difficultés techniques pour traiter votre question. Pouvez-vous la reformuler ?"

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
