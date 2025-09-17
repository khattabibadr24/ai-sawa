from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from core.config import (
    K, TEMPERATURE, MODEL_NAME, MAX_CONCURRENCY, MAX_TURNS
)
from routes.chat import router as chat_router

app = FastAPI(title="AI-SAWA Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, tags=["chat"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return JSONResponse(
        {
            "name": "AI-SAWA Chatbot",
            "version": "2.0.0",
            "description": "Assistant intelligent dédié à l'aidance",
            "endpoints": {
                "health": "/health",
                "chat (POST)": "/chat",
                "chat stream (GET, SSE)": "/chat/stream?question=...&session_id=...",
            },
            "configuration": {
                "K": K,
                "TEMPERATURE": TEMPERATURE,
                "MODEL_NAME": MODEL_NAME,
                "MAX_CONCURRENCY": MAX_CONCURRENCY,
                "MAX_TURNS": MAX_TURNS,
            },
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
