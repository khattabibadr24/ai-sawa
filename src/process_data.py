
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_data(file_path: str):
    """
    Charge les données depuis un fichier JSON et les divise en chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    for item in data:
        doc_id = item.get('id')
        context = item.get('content')
        if context:
            chunks = text_splitter.create_documents([context])
            for i, chunk in enumerate(chunks):
                chunk.metadata = {"source": doc_id, "chunk_id": f"{doc_id}_{i}"}
                all_chunks.append(chunk)
                
    return all_chunks

if __name__ == "__main__":
    # Exemple d'utilisation
    json_file_path = "/home/khattabi/Desktop/AI-sawa/data/medical_data.json"
    chunks = load_and_chunk_data(json_file_path)
    print(f"Nombre total de chunks générés : {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]): # Afficher les 5 premiers chunks pour vérification
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.metadata.get('chunk_id')}")


