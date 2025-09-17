import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# --- .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

# Lis les deux variantes possibles
API_KEY = os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

# === Hyperparamètres FIXES ===
K = 3
TEMPERATURE = 0.0

# Concurrence & fallback
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
FALLBACK_MODELS = [MODEL_NAME, "mistral-small", "mistral-tiny"]

# Fenêtre de mémoire locale (serveur)
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))  # nombre de tours (user+assistant) conservés

# --- Vector Database Configuration ---
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_data_collection")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "/home/khattabi/Desktop/qdrant_local")
RECREATE_COLLECTION = os.getenv("RECREATE_COLLECTION", "0") in ("1", "true", "True")

# --- Debug Configuration ---
DEBUG = os.getenv("DEBUG", "1") not in ("0", "false", "False")

def log(*args):
    if DEBUG:
        print(*args)

def require_api_key():
    from fastapi import HTTPException
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Clé Mistral absente. Définis API_KEY ou MISTRAL_API_KEY dans le .env."
        )
