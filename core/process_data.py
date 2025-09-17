import json
import re
import numpy as np

from typing import List, Tuple

# Compatibilité LangChain (anciens/nouveaux imports)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore

from sentence_transformers import SentenceTransformer


# -------- Helpers sémantiques --------

def _l2_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-12)

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (_l2_norm(u) * _l2_norm(v)))


def _sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Découpe en phrases en conservant les indices (start, end).
    Le pattern capture jusqu'à la ponctuation finale ou la fin de texte.
    """
    # Ponctuation élargie . ? ! … و العربية ؟
    pattern = r'[^\.!\?…\n]+(?:[\.!\?…]+|\Z)'
    spans = []
    for m in re.finditer(pattern, text, flags=re.UNICODE):
        start, end = m.span()
        sent = text[start:end].strip()
        if sent:
            spans.append((start, end, sent))
    if not spans:  # fallback si texte très court
        spans = [(0, len(text), text)]
    return spans


def _semantic_segments(
    sentences: List[str],
    embeddings: np.ndarray,
    max_chars: int = 900,
    min_chars: int = 200,
    similarity_threshold: float = 0.72,
) -> List[List[int]]:
    """
    Regroupe les indices de phrases en segments sémantiques.
    - Nouveau chunk si (similarité avec le centroïde courant < seuil)
      ou si on dépasse max_chars.
    - Fusion des tout petits segments (< min_chars) avec leur voisin.
    """
    if len(sentences) == 1:
        return [[0]]

    segments: List[List[int]] = []
    current: List[int] = [0]
    cur_chars = len(sentences[0])
    centroid = embeddings[0].copy()

    for i in range(1, len(sentences)):
        sim = _cosine(centroid, embeddings[i])
        next_len = cur_chars + 1 + len(sentences[i])  # +1 espace

        should_split = (sim < similarity_threshold) or (next_len > max_chars)
        if should_split:
            segments.append(current)
            current = [i]
            cur_chars = len(sentences[i])
            centroid = embeddings[i].copy()
        else:
            current.append(i)
            # maj centroïde (moyenne incrémentale simple)
            centroid = (centroid * (len(current) - 1) + embeddings[i]) / len(current)
            cur_chars = next_len

    if current:
        segments.append(current)

    # Fusion des petits segments
    merged: List[List[int]] = []
    buf: List[int] = []
    for seg in segments:
        if not buf:
            buf = seg
            continue
        # longueur du buffer en chars
        # (approxime en joignant avec espaces)
        def seg_len(ixs: List[int]) -> int:
            return sum(len(sentences[j]) for j in ixs) + max(0, len(ixs) - 1)

        if seg_len(buf) < min_chars:
            buf = buf + seg  # fusion avec suivant
        else:
            merged.append(buf)
            buf = seg
    if buf:
        merged.append(buf)

    # Si le tout dernier est encore trop petit, fusionne en arrière si possible
    def seg_len(ixs: List[int]) -> int:
        return sum(len(sentences[j]) for j in ixs) + max(0, len(ixs) - 1)

    if len(merged) >= 2 and seg_len(merged[-1]) < min_chars:
        merged[-2] = merged[-2] + merged[-1]
        merged.pop()

    return merged


# -------- Pipeline principal --------

def load_and_chunk_data(
    file_path: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_chars: int = 900,
    min_chars: int = 200,
    similarity_threshold: float = 0.72,
):
    """
    Charge un JSON [{id, content}, ...] et segmente chaque content en chunks SÉMANTIQUES.
    Retourne une liste de Document LangChain avec metadata:
        - source: id du doc d'origine
        - chunk_id: <id>_<index>
        - start_index, end_index: offsets caractère dans le texte source
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceTransformer(model_name)

    all_docs: List[Document] = []

    for item in data:
        doc_id = item.get("id")
        context: str = (item.get("content") or "").strip()
        if not context:
            continue

        # 1) Phrases + spans (indices)
        spans = _sentence_spans(context)
        sentences = [s for _, _, s in spans]

        # 2) Embeddings de phrases
        embs = np.asarray(model.encode(sentences, normalize_embeddings=True))

        # 3) Segmentation sémantique (listes d'indices de phrases)
        groups = _semantic_segments(
            sentences,
            embs,
            max_chars=max_chars,
            min_chars=min_chars,
            similarity_threshold=similarity_threshold,
        )

        # 4) Reconstruction des chunks + indices start/end
        for i, idx_group in enumerate(groups):
            start_char = spans[idx_group[0]][0]
            end_char = spans[idx_group[-1]][1]
            chunk_text = context[start_char:end_char].strip()

            meta = {
                "source": doc_id,
                "chunk_id": f"{doc_id}_{i}",
                "start_index": start_char,
                "end_index": end_char,
            }
            all_docs.append(Document(page_content=chunk_text, metadata=meta))

    return all_docs


if __name__ == "__main__":
    # Exemple d'utilisation
    json_file_path = "/home/khattabi/Desktop/AI-sawa/data/medical_data.json"
    chunks = load_and_chunk_data(
        json_file_path,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_chars=900,          # taille max d'un chunk (approx chars)
        min_chars=200,          # taille min; fusionne les trop petits
        similarity_threshold=0.72,  # plus haut = segments plus "cohérents", plus nombreux
    )
    print(f"Nombre total de chunks générés : {len(chunks)}")
    for i, doc in enumerate(chunks[:5]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {doc.metadata.get('chunk_id')}")
        print(f"  start_index: {doc.metadata.get('start_index')}, end_index: {doc.metadata.get('end_index')}")
        print(f"  Aperçu: {doc.page_content[:120].replace('\\n',' ')}...")
