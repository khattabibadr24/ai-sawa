import json
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import os
import locale
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Charger les variables d'environnement depuis .env (dossier parent)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)
# Ajoutez ceci temporairement au début du main (après les imports)
print("🔍 DIAGNOSTIC FICHIER .ENV:")
import os
print(f"📁 Répertoire courant: {os.getcwd()}")
print(f"📄 Fichier .env existe? {os.path.exists('../.env')}")
print(f"📄 Contenu du répertoire parent: {os.listdir('..')}")

# Test de chargement
from dotenv import load_dotenv
result = load_dotenv(dotenv_path="../.env")
print(f"✅ Chargement .env réussi: {result}")
print(f"🔑 OPENAI_API_KEY trouvée: {'Oui' if os.getenv('OPENAI_API_KEY') else 'Non'}")

# Configurer l'encodage par défaut pour éviter les problèmes
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass  # Utiliser les paramètres par défaut

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


def analyze_json_data(file_path: str) -> Dict[str, Any]:
    """
    Analyse complète du fichier JSON pour diagnostiquer les documents manquants.
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
    
    total_docs = len(data)
    empty_content = 0
    missing_content = 0
    missing_id = 0
    valid_docs = 0
    content_lengths = []
    
    empty_ids = []
    missing_content_ids = []
    missing_id_indices = []
    very_short_content = []  # < 10 caractères
    
    for i, item in enumerate(data):
        doc_id = item.get("id")
        content = item.get("content")
        
        # Vérifier l'ID
        if doc_id is None or doc_id == "":
            missing_id += 1
            missing_id_indices.append(i)
            continue
            
        # Vérifier le contenu
        if content is None:
            missing_content += 1
            missing_content_ids.append(doc_id)
            continue
            
        content_clean = content.strip() if isinstance(content, str) else str(content).strip()
        
        # Nettoyage supplémentaire pour éviter les problèmes d'encodage
        if content_clean:
            try:
                # Nettoyer les caractères de contrôle
                content_clean = content_clean.replace('\x00', ' ')
                content_clean = content_clean.replace('\r\n', '\n').replace('\r', '\n')
                # Vérifier que c'est encodable en UTF-8
                content_clean.encode('utf-8')
            except UnicodeEncodeError:
                # Si problème d'encodage, nettoyer davantage
                content_clean = content_clean.encode('utf-8', errors='ignore').decode('utf-8')
        
        if not content_clean:
            empty_content += 1
            empty_ids.append(doc_id)
        elif len(content_clean) < 10:  # Très court
            very_short_content.append((doc_id, len(content_clean), content_clean[:50]))
            valid_docs += 1  # On garde quand même les très courts
            content_lengths.append(len(content_clean))
        else:
            valid_docs += 1
            content_lengths.append(len(content_clean))
    
    return {
        "total_docs": total_docs,
        "valid_docs": valid_docs,
        "empty_content": empty_content,
        "missing_content": missing_content,
        "missing_id": missing_id,
        "very_short_content": len(very_short_content),
        "empty_ids": empty_ids[:10],
        "missing_content_ids": missing_content_ids[:10], 
        "missing_id_indices": missing_id_indices[:10],
        "very_short_examples": very_short_content[:5],
        "avg_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
        "min_content_length": min(content_lengths) if content_lengths else 0,
        "max_content_length": max(content_lengths) if content_lengths else 0,
    }


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
    batch_size: int = 50,
    estimate_cost: bool = True
) -> List[Document]:
    """
    Version avec embeddings OpenAI configuré via .env.
    
    Args:
        file_path: Chemin vers le fichier JSON
        openai_api_key: Clé API OpenAI (ou None pour fichier .env)
        min_content_length: Longueur minimale du contenu (caractères)
        batch_size: Taille des lots pour les requêtes API
        estimate_cost: Afficher l'estimation du coût
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

    # Estimation du coût
    if estimate_cost:
        # Estimation approximative (1 token ≈ 4 caractères pour l'anglais/français)
        total_chars = sum(len(text) for text in contexts)
        estimated_tokens = total_chars // 4
        estimated_cost = (estimated_tokens / 1000) * model_config["price_per_1k"]
        
        print(f"💰 ESTIMATION DU COÛT:")
        print(f"   • Caractères totaux: {total_chars:,}")
        print(f"   • Tokens estimés: {estimated_tokens:,}")
        print(f"   • Coût estimé: ${estimated_cost:.4f}")
        
        if estimated_cost > 1.0:
            confirmation = input("⚠️  Coût > $1. Continuer? (y/n): ")
            if confirmation.lower() != 'y':
                print("❌ Arrêt sur demande utilisateur")
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


def save_embeddings_to_file(docs: List[Document], output_path: str):
    """
    Sauvegarde les embeddings dans un fichier JSON pour réutilisation.
    """
    data_to_save = []
    for doc in docs:
        data_to_save.append({
            "chunk_id": doc.metadata["chunk_id"],
            "source": doc.metadata["source"], 
            "content": doc.page_content,
            "embedding": doc.metadata["embedding"],
            "metadata": {k: v for k, v in doc.metadata.items() if k != "embedding"}
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Embeddings sauvegardés dans: {output_path}")


if __name__ == "__main__":
    # ============= DIAGNOSTIC D'ENCODAGE =============
    print("🔍 DIAGNOSTIC D'ENCODAGE SYSTÈME")
    print("=" * 50)
    
    print(f"📊 Encodage par défaut: {locale.getpreferredencoding()}")
    print(f"🌍 Locale: {locale.getlocale()}")
    
    # Test d'encodage avec caractères français
    test_text = "Médecine française avec accents: àáâãäåæçèéêëìíîï"
    try:
        test_text.encode('utf-8')
        print(f"✅ Test UTF-8 réussi: {test_text[:30]}...")
    except UnicodeEncodeError as e:
        print(f"❌ Erreur UTF-8: {e}")
    
    json_file_path = "/home/khattabi/Desktop/AI-sawa/data/medical_data.json"
    
    # Alternative: chemin relatif depuis src/
    # json_file_path = "../data/medical_data.json"
    
    # ============= ANALYSE PRÉLIMINAIRE =============
    print("🔍 ANALYSE PRÉLIMINAIRE DES DONNÉES")
    print("=" * 50)
    
    analysis = analyze_json_data(json_file_path)
    
    print(f"📊 Documents totaux dans le JSON: {analysis['total_docs']}")
    print(f"✅ Documents potentiellement valides: {analysis['valid_docs']}")
    print(f"❌ Documents avec contenu vide: {analysis['empty_content']}")
    print(f"❌ Documents sans champ 'content': {analysis['missing_content']}")
    print(f"❌ Documents sans ID: {analysis['missing_id']}")
    print(f"⚠️  Documents très courts (< 10 car): {analysis['very_short_content']}")
    print(f"📏 Longueur moyenne du contenu: {analysis['avg_content_length']:.0f} caractères")
    print(f"📏 Longueur min/max: {analysis['min_content_length']} / {analysis['max_content_length']}")
    
    if analysis['empty_ids']:
        print(f"🔍 Exemples d'IDs avec contenu vide: {analysis['empty_ids']}")
    if analysis['missing_content_ids']:
        print(f"🔍 Exemples d'IDs sans contenu: {analysis['missing_content_ids']}")
    if analysis['very_short_examples']:
        print(f"🔍 Exemples de contenus très courts:")
        for doc_id, length, preview in analysis['very_short_examples']:
            print(f"     ID {doc_id}: {length} car -> '{preview}'")
    
    # ============= GÉNÉRATION DES EMBEDDINGS OPENAI =============
    print("\n🚀 GÉNÉRATION DES EMBEDDINGS OPENAI")
    print("=" * 50)
    print("🤖 Utilisation du modèle: text-embedding-3-large (configuré via .env)")
    print("📐 Dimensions: 3072 | 💰 Prix: $0.00013/1k tokens")
    
    try:
        # Génération des embeddings avec OpenAI (modèle configuré via .env)
        chunks_with_embeddings = load_and_chunk_data_with_openai_embeddings(
            json_file_path,
            # openai_api_key="sk-...",  # Optionnel si .env configuré
            min_content_length=10,
            batch_size=50,  # Optimisé pour text-embedding-3-large
            estimate_cost=True
        )
        
        print(f"\n✅ SUCCÈS: {len(chunks_with_embeddings)} chunks avec embeddings générés!")
        
        # ============= STATISTIQUES FINALES =============
        print(f"\n📈 STATISTIQUES FINALES:")
        print(f"   • Documents originaux: {analysis['total_docs']}")
        print(f"   • Documents avec embeddings: {len(chunks_with_embeddings)}")
        print(f"   • Documents exclus: {analysis['total_docs'] - len(chunks_with_embeddings)}")
        
        # Exemples des premiers chunks
        print(f"\n📄 EXEMPLES DE CHUNKS:")
        for i, doc in enumerate(chunks_with_embeddings[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  📋 ID: {doc.metadata.get('chunk_id')}")
            print(f"  🏷️  Source: {doc.metadata.get('source')}")
            print(f"  📏 Taille: {doc.metadata.get('total_chars')} caractères")
            print(f"  🧠 Modèle: {doc.metadata.get('embedding_model')} ({doc.metadata.get('provider')})")
            print(f"  📐 Dimensions: {doc.metadata.get('embedding_dimensions')}")
            preview = doc.page_content[:120].replace('\n', ' ').replace('\r', ' ')
            print(f"  👀 Aperçu: {preview}...")
        
        # Optionnel: sauvegarder les embeddings
        # save_embeddings_to_file(chunks_with_embeddings, "openai_embeddings_output.json")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        print("💡 Vérifiez:")
        print("   • Votre fichier .env avec:")
        print("     OPENAI_API_KEY=sk-proj-...")
        print("     OPENAI_EMBEDDING_MODEL=text-embedding-3-large")
        print("   • Votre installation: pip install openai python-dotenv")