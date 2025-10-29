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


# app.py
import streamlit as st
import os
import pickle
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils import load_text_from_file, chunk_text

# Config
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    """Load the sentence transformer model (cached)"""
    return SentenceTransformer(EMBED_MODEL_NAME)

def process_document(uploaded_file):
    """Process uploaded document and create FAISS index"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and chunk text
        with st.spinner("Reading document..."):
            text = load_text_from_file(tmp_path)
            st.success(f"Document loaded successfully! Length: {len(text)} characters")
        
        with st.spinner("Creating chunks..."):
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            st.info(f"Created {len(chunks)} chunks")
        
        # Create document metadata
        docs = [{
            "text": chunk,
            "source": uploaded_file.name,
            "chunk_id": i
        } for i, chunk in enumerate(chunks)]
        
        # Load model
        if st.session_state.model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.model = load_model()
        
        # Create embeddings
        with st.spinner("Creating embeddings..."):
            texts = [d["text"] for d in docs]
            embeddings = st.session_state.model.encode(
                texts, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
        
        # Create FAISS index
        with st.spinner("Building search index..."):
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            
            # Normalize and add to index
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return index, docs
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, []

def search_similar_chunks(query, top_k=3):
    """Search for similar chunks in the index"""
    if st.session_state.index is None or st.session_state.model is None:
        return []
    
    # Encode query
    query_embedding = st.session_state.model.encode(
        [query], 
        convert_to_numpy=True
    ).astype("float32")
    
    # Normalize
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = st.session_state.index.search(query_embedding, top_k)
    
    # Return results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(st.session_state.docs):
            results.append({
                "rank": i + 1,
                "score": float(dist),
                "text": st.session_state.docs[idx]["text"],
                "source": st.session_state.docs[idx]["source"],
                "chunk_id": st.session_state.docs[idx]["chunk_id"]
            })
    
    return results

def main():
    st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
    
    st.title("📄 Document Question & Answer")
    st.markdown("Upload a document and ask questions about its content!")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload a PDF, TXT, or DOCX file"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                index, docs = process_document(uploaded_file)
                if index is not None:
                    st.session_state.index = index
                    st.session_state.docs = docs
                    st.success("✅ Document processed successfully!")
                    st.balloons()
        
        st.divider()
        
        # Display document info
        if st.session_state.docs:
            st.subheader("📊 Document Info")
            st.metric("Document Name", st.session_state.docs[0]["source"])
            st.metric("Total Chunks", len(st.session_state.docs))
            st.metric("Total Characters", sum(len(d["text"]) for d in st.session_state.docs))
    
    # Main content area
    if st.session_state.index is None:
        st.info("👈 Please upload a document from the sidebar to get started!")
        st.markdown("""
        ### How to use:
        1. Upload a PDF, TXT, or DOCX file from the sidebar
        2. Click "Process Document" to create the search index
        3. Ask questions about your document in the text box below
        4. View relevant excerpts and answers
        """)
    else:
        st.success("✅ Document loaded and ready for questions!")
        
        # Question input
        st.subheader("🔍 Ask a Question")
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic of this document?",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            top_k = st.selectbox("Results:", [3, 5, 10], index=0)
        
        if query:
            with st.spinner("Searching..."):
                results = search_similar_chunks(query, top_k=top_k)
            
            if results:
                st.subheader(f"📝 Top {len(results)} Relevant Excerpts")
                
                for result in results:
                    with st.expander(
                        f"**Result #{result['rank']}** - Similarity: {result['score']:.3f}",
                        expanded=(result['rank'] == 1)
                    ):
                        st.markdown(f"**Source:** {result['source']} (Chunk {result['chunk_id']})")
                        st.markdown("---")
                        st.write(result['text'])
                
                # Generate answer based on top results
                st.subheader("💡 Generated Answer")
                context = "\n\n".join([r['text'] for r in results[:3]])
                st.info(f"""
                Based on the top relevant excerpts from your document:
                
                {context[:1000]}{'...' if len(context) > 1000 else ''}
                
                *Note: This is a simple retrieval system. For better answers, integrate with an LLM like OpenAI's GPT or Claude.*
                """)
            else:
                st.warning("No relevant results found. Try rephrasing your question.")

if __name__ == "__main__":
    main()