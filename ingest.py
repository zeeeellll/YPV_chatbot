# ingest.py
import os
import pickle
from openai import embeddings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from utils import load_text_from_file, chunk_text
import numpy as np

# Config
DATA_DIR = os.path.dirname(r"C:\Users\MAHADEV\Downloads\1st.pdf") # directory containing your pdf
PDF_FILE = r"C:\Users\MAHADEV\Downloads\1st.pdf" # specific pdf file
INDEX_DIR = "vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast; change if you want better embeddings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(INDEX_DIR, exist_ok=True)

def main():
    # 1) load model
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2) read the PDF file and chunk
    docs = []  # list of dicts: {"text":..., "source":..., "meta":...}
    try:
        print(f"Attempting to read file: {PDF_FILE}")
        text = load_text_from_file(PDF_FILE)
        print(f"Successfully read file. Text length: {len(text)}")
        fname = os.path.basename(PDF_FILE)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Created {len(chunks)} chunks")
        docs.extend([{
            "text": c,
            "source": fname,
            "chunk_id": i
        } for i, c in enumerate(chunks)])
    except Exception as e:
        print(f"Error reading file {PDF_FILE}: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3️⃣ Create embeddings in batches
    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # 4️⃣ Create FAISS index (cosine similarity)
    import numpy as np
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # ✅ normalize + convert dtype
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index.add(embeddings) # type: ignore
    
    # 5) persist index + metadata
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(docs, f)

    print("Saved FAISS index and metadata.")

if __name__ == "__main__":
    main()

