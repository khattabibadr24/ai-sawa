import json
from typing import List, Dict, Any, Tuple
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Charger les variables d'environnement depuis .env (dossier parent)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

# Compatibilité LangChain (anciens/nouveaux imports)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore


# ========== CONFIGURATION MODÈLE PAR DÉFAUT ==========
DEFAULT_MODEL_CONFIG = {
    "model_name": "text-embedding-3-large", 
    "dimensions": 3072,
    "max_tokens": 8192,
    "price_per_1k": 0.00013,
    "description": "OpenAI text-embedding-3-large - Le plus performant"
}


def initialize_openai_client(api_key: str = None) -> Tuple[OpenAI, Dict[str, Any]]:
    """
    Initialise le client OpenAI avec la clé API et la configuration du modèle depuis .env.
    """
    if api_key:
        client = OpenAI(api_key=api_key)
        print("🔑 Utilisation de la clé API fournie en paramètre")
    else:
        # Charge depuis .env ou variable d'environnement OPENAI_API_KEY
        api_key_from_env = os.getenv('OPENAI_API_KEY')
        if not api_key_from_env:
            raise ValueError(
                "❌ Clé API OpenAI non trouvée!\n"
                "💡 Créez un fichier .env avec:\n"
                "   OPENAI_API_KEY=sk-proj-votre-clé-ici\n"
                "   OPENAI_EMBEDDING_MODEL=text-embedding-3-large\n"
                "📄 Ou définissez la variable: export OPENAI_API_KEY='sk-proj-...'"
            )
        
        client = OpenAI(api_key=api_key_from_env)
        print(f"🔑 Clé API chargée depuis .env (sk-...{api_key_from_env[-4:]})")
    
    # Configuration du modèle depuis .env ou valeur par défaut
    model_name = os.getenv('OPENAI_EMBEDDING_MODEL', DEFAULT_MODEL_CONFIG["model_name"])
    
    model_config = DEFAULT_MODEL_CONFIG.copy()
    model_config["model_name"] = model_name
    
    print(f"🤖 Modèle configuré: {model_name}")
    print(f"📐 Dimensions: {model_config['dimensions']}")
    print(f"💰 Prix estimé: ${model_config['price_per_1k']}/1k tokens")
    
    return client, model_config




def get_openai_embeddings_batch(
    client: OpenAI, 
    texts: List[str], 
    model_name: str = "text-embedding-3-large",  # Par défaut le modèle haute performance
    batch_size: int = 50,  # Optimisé pour text-embedding-3-large
    max_retries: int = 3,
    delay_between_batches: float = 1.5  # Plus conservateur pour le modèle large
) -> List[List[float]]:
    """
    Obtient les embeddings OpenAI par lots avec gestion d'erreurs et encodage UTF-8.
    
    Args:
        client: Client OpenAI initialisé
        texts: Liste des textes à vectoriser
        model_name: Nom du modèle OpenAI
        batch_size: Taille des lots pour les requêtes
        max_retries: Nombre maximum de tentatives
        delay_between_batches: Délai entre les lots (secondes)
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"🚀 Génération des embeddings OpenAI en {total_batches} lots de {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"📦 Lot {batch_num}/{total_batches} ({len(batch_texts)} textes)...")
        
        # Nettoyer et encoder correctement les textes pour éviter les erreurs d'encodage
        cleaned_batch = []
        for text in batch_texts:
            # S'assurer que le texte est une chaîne Unicode valide
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            elif not isinstance(text, str):
                text = str(text)
            
            # Nettoyer les caractères de contrôle problématiques
            text = text.replace('\x00', ' ')  # Caractère null
            text = text.replace('\r\n', '\n')  # Normaliser les retours à la ligne
            text = text.replace('\r', '\n')
            
            # S'assurer que le texte n'est pas vide après nettoyage
            if text.strip():
                cleaned_batch.append(text.strip())
            else:
                cleaned_batch.append("contenu vide")  # Placeholder pour éviter les erreurs
        
        for attempt in range(max_retries):
            try:
                # Appel à l'API OpenAI avec textes nettoyés
                response = client.embeddings.create(
                    model=model_name,
                    input=cleaned_batch,
                    encoding_format="float"
                )
                
                # Extraire les embeddings
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"✅ Lot {batch_num} traité avec succès")
                break
                
            except UnicodeEncodeError as e:
                print(f"⚠️  Erreur d'encodage lot {batch_num}, tentative {attempt + 1}: {e}")
                # Essayer de nettoyer davantage les textes
                cleaned_batch = []
                for text in batch_texts:
                    # Encoder/décoder pour nettoyer
                    try:
                        text_clean = text.encode('utf-8', errors='ignore').decode('utf-8')
                        # Remplacer les caractères problématiques
                        text_clean = text_clean.encode('ascii', errors='ignore').decode('ascii')
                        if text_clean.strip():
                            cleaned_batch.append(text_clean)
                        else:
                            cleaned_batch.append("contenu nettoye")
                    except:
                        cleaned_batch.append("contenu problematique")
                        
                if attempt == max_retries - 1:
                    raise e
                        
            except Exception as e:
                print(f"⚠️  Erreur lot {batch_num}, tentative {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponentiel
                else:
                    print(f"❌ Échec définitif pour le lot {batch_num}")
                    raise e
        
        # Délai entre les lots pour respecter les limites de taux
        if i + batch_size < len(texts):
            time.sleep(delay_between_batches)
    
    return all_embeddings


def load_and_chunk_data_with_openai_embeddings(
    file_path: str,
    openai_api_key: str = None,
    min_content_length: int = 5,
    batch_size: int = 50
) -> List[Document]:
    """
    Version avec embeddings OpenAI configuré via .env.
    
    Args:
        file_path: Chemin vers le fichier JSON
        openai_api_key: Clé API OpenAI (ou None pour fichier .env)
        min_content_length: Longueur minimale du contenu (caractères)
        batch_size: Taille des lots pour les requêtes API
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        print("⚠️  Erreur d'encodage UTF-8, tentative avec latin-1...")
        with open(file_path, "r", encoding="latin-1") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Erreur lecture fichier {file_path}: {e}")
        raise

    # Initialisation du client OpenAI et récupération de la config
    try:
        client, model_config = initialize_openai_client(openai_api_key)
        print("✅ Client OpenAI initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur initialisation OpenAI: {e}")
        raise

    model_name = model_config["model_name"]

    all_docs: List[Document] = []
    
    # Filtrage des documents valides avec critères stricts
    contexts = []
    valid_items = []
    filtered_count = {"empty_content": 0, "missing_content": 0, "missing_id": 0, "too_short": 0}
    
    print(f"🔍 Filtrage avec longueur minimale: {min_content_length} caractères...")
    
    for item in data:
        doc_id = item.get("id")
        content = item.get("content")
        
        # Vérifications strictes
        if doc_id is None or doc_id == "":
            filtered_count["missing_id"] += 1
            continue
            
        if content is None:
            filtered_count["missing_content"] += 1
            continue
            
        content_clean = content.strip() if isinstance(content, str) else str(content).strip()
        
        if not content_clean:
            filtered_count["empty_content"] += 1
            continue
            
        if len(content_clean) < min_content_length:
            filtered_count["too_short"] += 1
            continue
            
        contexts.append(content_clean)
        valid_items.append(item)

    print(f"📊 Résultats du filtrage:")
    print(f"   • Documents valides retenus: {len(contexts)}")
    print(f"   • Exclus - ID manquant: {filtered_count['missing_id']}")  
    print(f"   • Exclus - Contenu manquant: {filtered_count['missing_content']}")
    print(f"   • Exclus - Contenu vide: {filtered_count['empty_content']}")
    print(f"   • Exclus - Trop court (< {min_content_length} car): {filtered_count['too_short']}")
    
    if not contexts:
        print("❌ Aucun document valide trouvé!")
        return all_docs

    # Génération des embeddings via OpenAI
    try:
        embeddings = get_openai_embeddings_batch(
            client=client,
            texts=contexts,
            model_name=model_name,
            batch_size=batch_size
        )
        
        print(f"✅ {len(embeddings)} embeddings générés avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur génération embeddings: {e}")
        raise

    # Création des documents avec embeddings
    print("📝 Création des documents avec embeddings...")
    for i, (item, embedding) in enumerate(zip(valid_items, embeddings)):
        doc_id = item.get("id")
        content = contexts[i]
        
        meta = {
            "source": str(doc_id),
            "chunk_id": f"{doc_id}_0",
            "start_index": 0,
            "end_index": len(content),
            "total_chars": len(content),
            "embedding_model": model_name,
            "embedding_dimensions": len(embedding),
            "model_config": model_config,
            "provider": "openai"
        }
        
        # Stocker l'embedding comme liste Python (compatible JSON)
        meta["embedding"] = embedding
        
        all_docs.append(Document(page_content=content, metadata=meta))

    return all_docs




if __name__ == "__main__":
    # Test simple pour vérifier le fonctionnement
    json_file_path = "data/medical_data.json"
    
    try:
        print("🚀 Test de génération des embeddings OpenAI")
        chunks_with_embeddings = load_and_chunk_data_with_openai_embeddings(
            json_file_path,
            min_content_length=10,
            batch_size=50
        )
        
        print(f"✅ Succès: {len(chunks_with_embeddings)} documents avec embeddings générés!")
        
        # Exemple du premier chunk
        if chunks_with_embeddings:
            doc = chunks_with_embeddings[0]
            print(f"📋 Premier chunk - ID: {doc.metadata.get('chunk_id')}")
            print(f"🧠 Modèle: {doc.metadata.get('embedding_model')}")
            print(f"📐 Dimensions: {doc.metadata.get('embedding_dimensions')}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Vérifiez votre fichier .env avec OPENAI_API_KEY")
