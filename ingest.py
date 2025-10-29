# # ingest.py
# import os
# import pickle
# from openai import embeddings
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# import faiss
# from utils import load_text_from_file, chunk_text
# import numpy as np

# # Config
# DATA_DIR = os.path.dirname(r"C:\Users\MAHADEV\Downloads\1st.pdf") # directory containing your pdf
# PDF_FILE = r"C:\Users\MAHADEV\Downloads\1st.pdf" # specific pdf file
# INDEX_DIR = "vector_store"
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast; change if you want better embeddings
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50

# os.makedirs(INDEX_DIR, exist_ok=True)

# def main():
#     # 1) load model
#     model = SentenceTransformer(EMBED_MODEL_NAME)

#     # 2) read the PDF file and chunk
#     docs = []  # list of dicts: {"text":..., "source":..., "meta":...}
#     try:
#         print(f"Attempting to read file: {PDF_FILE}")
#         text = load_text_from_file(PDF_FILE)
#         print(f"Successfully read file. Text length: {len(text)}")
#         fname = os.path.basename(PDF_FILE)
#         chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
#         print(f"Created {len(chunks)} chunks")
#         docs.extend([{
#             "text": c,
#             "source": fname,
#             "chunk_id": i
#         } for i, c in enumerate(chunks)])
#     except Exception as e:
#         print(f"Error reading file {PDF_FILE}: {e}")
#         import traceback
#         traceback.print_exc()
#         return

#     # 3️⃣ Create embeddings in batches
#     texts = [d["text"] for d in docs]
#     print(f"Embedding {len(texts)} chunks...")
#     embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

#     # 4️⃣ Create FAISS index (cosine similarity)
#     import numpy as np
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)

#     # ✅ normalize + convert dtype
#     embeddings = embeddings.astype("float32")
#     faiss.normalize_L2(embeddings)
#     index.add(embeddings) # type: ignore
    
#     # 5) persist index + metadata
#     faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
#     with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
#         pickle.dump(docs, f)

#     print("Saved FAISS index and metadata.")

# if __name__ == "__main__":
#     main()


# ingest.py
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from utils import load_text_from_file, chunk_text
import numpy as np
from datetime import datetime

# Config
INDEX_DIR = "vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Initialize model once (can be reused)
_model = None

def get_model():
    """Lazy load the embedding model"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def ingest_document(file_path, user_id=None):
    """
    Ingest a single uploaded document and create/update vector store
    
    Args:
        file_path: Path to the uploaded document (PDF, TXT, DOCX, etc.)
        user_id: Optional user identifier to create separate indexes per user
    
    Returns:
        dict: Status and metadata about the ingestion
    """
    # Create user-specific or shared index directory
    if user_id:
        index_dir = os.path.join(INDEX_DIR, f"user_{user_id}")
    else:
        index_dir = INDEX_DIR
    
    os.makedirs(index_dir, exist_ok=True)
    
    # Load model
    model = get_model()
    
    # Read and chunk the uploaded file
    docs = []
    try:
        print(f"Processing uploaded file: {file_path}")
        text = load_text_from_file(file_path)
        print(f"Successfully read file. Text length: {len(text)}")
        
        fname = os.path.basename(file_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Created {len(chunks)} chunks")
        
        # Create document metadata
        docs.extend([{
            "text": c,
            "source": fname,
            "chunk_id": i,
            "upload_timestamp": datetime.now().isoformat()
        } for i, c in enumerate(chunks)])
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "file": file_path
        }
    
    # Create embeddings
    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Prepare embeddings for FAISS
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    
    # Load existing index if it exists, or create new one
    index_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        # Load existing index and metadata
        print("Loading existing index...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            existing_docs = pickle.load(f)
        
        # Add new embeddings to existing index
        index.add(embeddings)
        docs = existing_docs + docs
        print(f"Added {len(texts)} new chunks to existing index")
    else:
        # Create new index
        print("Creating new index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"Created new index with {len(texts)} chunks")
    
    # Save updated index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(docs, f)
    
    print("Saved FAISS index and metadata.")
    
    return {
        "success": True,
        "file": fname,
        "chunks_added": len(texts),
        "total_chunks": len(docs),
        "index_dir": index_dir
    }

def clear_user_index(user_id=None):
    """Clear the vector store for a specific user or the shared store"""
    if user_id:
        index_dir = os.path.join(INDEX_DIR, f"user_{user_id}")
    else:
        index_dir = INDEX_DIR
    
    index_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    print(f"Cleared index for {'user ' + user_id if user_id else 'shared store'}")

# Example usage for Flask/FastAPI integration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_path> [user_id]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    user_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = ingest_document(file_path, user_id)
    print("\nIngestion Result:")
    print(result)