# --- CPU ONLY & silence CUDA warnings (doit être AVANT tout import torch/sentence-transformers) ---
import os, warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # masque les GPU pour PyTorch
warnings.filterwarnings(
    "ignore",
    message="CUDA initialization",
    category=UserWarning,
    module="torch.cuda",
)

from pathlib import Path
from dotenv import load_dotenv

# --- Debug flag ---
DEBUG = os.getenv("DEBUG", "1") not in ("0", "false", "False")

def log(*args):
    if DEBUG:
        print(*args)

# --- Paths & env ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # dossier AI-sawa
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("API_KEY")  # doit être la clé Mistral
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_data_collection")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "/home/khattabi/Desktop/qdrant_local")
RECREATE_COLLECTION = os.getenv("RECREATE_COLLECTION", "0") in ("1", "true", "True")

# --- Imports qui peuvent tirer torch après qu'on ait forcé le CPU ---
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from process_data import load_and_chunk_data_with_openai_embeddings  # ton chunking sémantique

import uuid

def get_local_qdrant_client() -> QdrantClient:
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    log(f"[qdrant] Path local: {QDRANT_LOCAL_PATH}")
    return QdrantClient(path=QDRANT_LOCAL_PATH)   # mode local (persistant)

def _ensure_collection(client: QdrantClient, embeddings: MistralAIEmbeddings):
    if RECREATE_COLLECTION and client.collection_exists(collection_name=COLLECTION_NAME):
        log(f"[qdrant] Suppression de la collection existante '{COLLECTION_NAME}' (RECREATE_COLLECTION=1)")
        client.delete_collection(collection_name=COLLECTION_NAME)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        log(f"[qdrant] Création de la collection '{COLLECTION_NAME}'...")
        try:
            dim = len(embeddings.embed_query("dimension-check"))
        except Exception as e:
            raise RuntimeError(
                f"[embeddings] Impossible d'obtenir la dimension (API_KEY invalide ? réseau ?) : {e}"
            ) from e

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            ),
            # Optionnels : paramètres d’index
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
        )
        log(f"[qdrant] Collection créée: dim={dim}, distance=COSINE")
    else:
        log(f"[qdrant] Collection '{COLLECTION_NAME}' déjà présente.")

def create_and_save_vector_db(chunks):
    if not API_KEY:
        raise ValueError("API_KEY non trouvée dans .env (doit contenir la clé Mistral).")
    if not chunks:
        raise ValueError("Aucun chunk à indexer (liste vide).")

    log("[embeddings] Initialisation MistralAIEmbeddings…")
    embeddings = MistralAIEmbeddings(mistral_api_key=API_KEY, model="mistral-embed")

    log("[qdrant] Initialisation client local…")
    client = get_local_qdrant_client()
    _ensure_collection(client, embeddings)

    log("[vectorstore] Création du QdrantVectorStore…")
    vector_db = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # IDs déterministes pour éviter les doublons lors des relances
    ids = []
    for i, d in enumerate(chunks):
        cid = (getattr(d, "metadata", {}) or {}).get("chunk_id", f"auto_{i}")
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, str(cid))))

    log(f"[vectorstore] Ajout de {len(chunks)} documents (batch_size=128)…")
    try:
        vector_db.add_documents(chunks, ids=ids, batch_size=128)
    except UnexpectedResponse as e:
        raise RuntimeError(f"[qdrant] Erreur d'insertion: {e}") from e

    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    log(f"[vectorstore] Insertion OK ✓  Points dans la collection: {count}")
    return vector_db

def load_vector_db():
    if not API_KEY:
        raise ValueError("API_KEY non trouvée dans .env (doit contenir la clé Mistral).")

    embeddings = MistralAIEmbeddings(mistral_api_key=API_KEY, model="mistral-embed")
    client = get_local_qdrant_client()

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        raise FileNotFoundError(
            f"La collection '{COLLECTION_NAME}' n'existe pas. "
            f"Lance d'abord create_and_save_vector_db(chunks)."
        )

    log("[vectorstore] Chargement de la base Qdrant locale…")
    return Qdrant(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

if __name__ == "__main__":
    json_file_path = PROJECT_ROOT / "data" / "medical_data.json"
    if not json_file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {json_file_path}")

    log(f"[main] Lecture: {json_file_path}")
    chunks = load_and_chunk_data_with_openai_embeddings(str(json_file_path))  # active les prints de chunking
    print(f"Total chunks prêts à indexer: {len(chunks)}")

    vector_db = create_and_save_vector_db(chunks)
    print("Vector DB prête.")
