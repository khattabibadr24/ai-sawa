import os
from pathlib import Path
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient, models
from process_data import load_and_chunk_data
import uuid
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # dossier AI-sawa
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)
API_KEY = os.getenv("API_KEY")

COLLECTION_NAME = "medical_data_collection"
QDRANT_LOCAL_PATH = "/home/khattabi/Desktop/qdrant_local"

def get_local_qdrant_client():
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    return QdrantClient(path=QDRANT_LOCAL_PATH)   # mode local (persistant)

def create_and_save_vector_db(chunks):
    if not API_KEY:
        raise ValueError("API_KEY non trouvée dans les variables d'environnement.")
    if not chunks:
        raise ValueError("Aucun chunk à indexer (liste vide).")

    embeddings = MistralAIEmbeddings(mistral_api_key=API_KEY, model="mistral-embed")

    print("Initialisation du client Qdrant en mode local...")
    client = get_local_qdrant_client()

    # Crée la collection si besoin (dimension = taille des embeddings)
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Création de la collection '{COLLECTION_NAME}'...")
        dim = len(embeddings.embed_query("test"))
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    print("Création du vectorstore (sans recréer de client interne)...")
    vector_db = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # IDs déterministes pour éviter les doublons lors des relances
    ids = []
    for i, d in enumerate(chunks):
        cid = (d.metadata.get("chunk_id") if hasattr(d, "metadata") else f"auto_{i}")
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, str(cid))))

    print(f"Ajout de {len(chunks)} documents (batchs) ...")
    vector_db.add_documents(chunks, ids=ids, batch_size=128)

    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Insertion OK. Points dans la collection: {count}")
    return vector_db

def load_vector_db():
    if not API_KEY:
        raise ValueError("API_KEY non trouvée dans les variables d'environnement.")
    embeddings = MistralAIEmbeddings(mistral_api_key=API_KEY, model="mistral-embed")
    client = get_local_qdrant_client()

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        raise FileNotFoundError(
            f"La collection '{COLLECTION_NAME}' n'existe pas encore. "
            f"Lance d'abord create_and_save_vector_db(chunks)."
        )

    print("Chargement de la base Qdrant locale...")
    return Qdrant(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    json_file_path = PROJECT_ROOT / "data" / "medical_data.json"
    if not json_file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {json_file_path}")

    chunks = load_and_chunk_data(str(json_file_path))
    print(f"Total chunks prêts à indexer: {len(chunks)}")
    vector_db = create_and_save_vector_db(chunks)
