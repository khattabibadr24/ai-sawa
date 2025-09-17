import os
import warnings
import uuid
from pathlib import Path

# --- CPU ONLY & silence CUDA warnings (doit être AVANT tout import torch/sentence-transformers) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings(
    "ignore",
    message="CUDA initialization",
    category=UserWarning,
    module="torch.cuda",
)

# --- Imports qui peuvent tirer torch après qu'on ait forcé le CPU ---
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from core.config import (
    API_KEY, COLLECTION_NAME, QDRANT_LOCAL_PATH, 
    RECREATE_COLLECTION, log, K
)

from core.process_data import load_and_chunk_data

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

def get_retriever():
    vector_db = load_vector_db()
    return vector_db.as_retriever(search_kwargs={"k": K})

def initialize_vector_db_from_data(json_file_path: str):
    if not Path(json_file_path).exists():
        raise FileNotFoundError(f"Fichier introuvable: {json_file_path}")

    log(f"[main] Lecture: {json_file_path}")
    chunks = load_and_chunk_data(str(json_file_path))
    log(f"Total chunks prêts à indexer: {len(chunks)}")

    vector_db = create_and_save_vector_db(chunks)
    log("Vector DB prête.")
    return vector_db
