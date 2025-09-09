# Chatbot Médical

Ce projet implémente un chatbot médical interactif utilisant Streamlit pour l'interface utilisateur, Langchain pour le traitement du langage naturel et Qdrant pour la recherche de similarité vectorielle.

## Fonctionnalités

- **Traitement des données JSON**: Charge et segmente les données médicales à partir d'un fichier JSON.
- **Génération d'Embeddings**: Convertit le texte des données médicales en représentations vectorielles (embeddings) à l'aide de l'API Mistral AI.
- **Base de données Vectorielle Qdrant**: Stocke les embeddings pour une recherche rapide de similarité. Qdrant est utilisé en mode service.
- **Recherche de Similarité Cosinus**: Trouve les documents les plus pertinents en fonction de la question de l'utilisateur.
- **Interface Utilisateur Streamlit**: Fournit une interface conviviale pour interagir avec le chatbot.
- **Historique de Conversation**: Maintient un historique des 5 dernières interactions.

## Configuration du Projet

### Prérequis

Assurez-vous d'avoir Python 3.8+ installé sur votre système.

### Installation

1.  **Cloner le dépôt (si applicable) ou décompresser le fichier fourni**:
    ```bash
    unzip medical_chatbot.zip
    cd medical_chatbot
    ```

2.  **Installer les dépendances Python**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration de l'API Key Mistral AI**:
    Créez un fichier `.env` à la racine du projet (`medical_chatbot/`) et ajoutez votre clé API Mistral AI ainsi que le nom du modèle:
    ```
    API_KEY=votre_cle_api_mistral
    MODEL_NAME=mistral-small-latest
    HF_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN_HERE # Optionnel, pour le tokenizer
    ```
    Remplacez `votre_cle_api_mistral` par votre véritable clé API.

## Utilisation

1.  **Démarrer le service Qdrant**: Avant de générer la base de données vectorielle ou de lancer l'application, vous devez démarrer le service Qdrant. Vous pouvez le faire en utilisant Docker (recommandé) ou en installant Qdrant localement.

    **Option 1: Utilisation de Docker (recommandé)**
    Assurez-vous que Docker est installé et en cours d'exécution sur votre système. Ensuite, exécutez la commande suivante dans votre terminal:
    ```bash
    docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
    ```
    Cette commande démarre un conteneur Qdrant, expose les ports 6333 (pour l'API) et 6334 (pour l'interface web), et monte un volume `qdrant_data` pour persister vos données.

    **Option 2: Installation locale (avancé)**
    Suivez les instructions sur le site officiel de Qdrant pour installer et démarrer le service localement.

2.  **Préparer les données**: Assurez-vous que votre fichier `medical_data.json` est situé dans le répertoire `data/` et qu'il contient des objets JSON avec les clés `id` et `context`.

3.  **Générer la base de données vectorielle**: Une fois le service Qdrant démarré, exécutez le script suivant pour traiter vos données, générer les embeddings et les stocker dans Qdrant. Cela peut prendre un certain temps en fonction de la taille de vos données et de votre connexion internet.
    ```bash
    python3 src/vector_db.py
    ```

4.  **Lancer l'application Streamlit**:
    ```bash
    streamlit run src/main.py
    ```

    L'application s'ouvrira automatiquement dans votre navigateur web.

## Structure du Projet

```
medical_chatbot/
├── .env
├── README.md
├── requirements.txt
├── data/
│   └── medical_data.json
├── models/
│   └── qdrant_index/  # Ce dossier sera créé par Qdrant si vous utilisez le mode embarqué, sinon les données seront dans le volume Docker.
└── src/
    ├── main.py
    ├── process_data.py
    └── vector_db.py
```

## Prochaines Étapes (Développement en cours)

- Intégration du LLM pour la génération de réponses.
- Implémentation de prompts structurés pour la politesse et la continuité de conversation.
- Affichage en streaming des réponses du LLM.
- Gestion de l'historique de conversation pour les 5 derniers messages.


