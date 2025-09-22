import os
import warnings
import uuid
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from core.config import (
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, COLLECTION_NAME, 
    RECREATE_COLLECTION, log, K
)
from core.process_data_openai import load_and_chunk_data_with_openai_embeddings

# --- CPU ONLY & silence CUDA warnings (doit être AVANT tout import torch/sentence-transformers) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings(
    "ignore",
    message="CUDA initialization",
    category=UserWarning,
    module="torch.cuda",
)

_retriever_instance = None

def get_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    return QdrantClient(url=qdrant_url)

def _ensure_collection(client: QdrantClient, embeddings: OpenAIEmbeddings):
    if RECREATE_COLLECTION and client.collection_exists(collection_name=COLLECTION_NAME):
        log(f"[qdrant] Suppression de la collection existante '{COLLECTION_NAME}' (RECREATE_COLLECTION=1)")
        client.delete_collection(collection_name=COLLECTION_NAME)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        log(f"[qdrant] Création de la collection '{COLLECTION_NAME}'...")
        try:
            dim = len(embeddings.embed_query("dimension-check"))
        except Exception as e:
            raise RuntimeError(
                f"[embeddings] Impossible d'obtenir la dimension (OPENAI_API_KEY invalide ? réseau ?) : {e}"
            ) from e

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            ),
            # Optionnels : paramètres d'index
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
        )
        log(f"[qdrant] Collection créée: dim={dim}, distance=COSINE")
    else:
        log(f"[qdrant] Collection '{COLLECTION_NAME}' déjà présente.")

def create_and_save_vector_db(chunks):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trouvée dans .env (doit contenir la clé OpenAI).")
    if not chunks:
        raise ValueError("Aucun chunk à indexer (liste vide).")

    log(f"[embeddings] Initialisation OpenAIEmbeddings avec le modèle: {OPENAI_EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL
    )

    log("[qdrant] Initialisation client Qdrant…")
    client = get_qdrant_client()
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
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trouvée dans .env (doit contenir la clé OpenAI).")

    log(f"[vectorstore] Chargement avec le modèle d'embedding: {OPENAI_EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL
    )
    client = get_qdrant_client()

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        raise FileNotFoundError(
            f"La collection '{COLLECTION_NAME}' n'existe pas. "
            f"Assure-toi que la base vectorielle a été initialisée."
        )

    return Qdrant(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

def get_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        vector_db = load_vector_db()
        _retriever_instance = vector_db.as_retriever(search_kwargs={"k": K})
    return _retriever_instance

def initialize_vector_db_from_data(json_file_path: str):
    if not Path(json_file_path).exists():
        raise FileNotFoundError(f"Fichier introuvable: {json_file_path}")
    
    log(f"[main] Lecture: {json_file_path}")
    chunks = load_and_chunk_data_with_openai_embeddings(str(json_file_path))
    log(f"Total chunks prêts à indexer: {len(chunks)}")

    vector_db = create_and_save_vector_db(chunks)
    log("Vector DB prête.")
    return vector_db