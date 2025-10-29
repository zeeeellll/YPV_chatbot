import os
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from prompt_templates import RAG_PROMPT_TEMPLATE
from dotenv import load_dotenv

# CONFIGURATION
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_DIR = "vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6

# LOAD MODELS AND DATA

st.sidebar.title("⚙️ Settings")
st.sidebar.write("Loading embedding model and FAISS index...")

try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    st.sidebar.success("Models loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading index or model: {e}")
    st.stop()

# HELPER FUNCTIONS

def retrieve(query: str, top_k=6):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        results.append(metadata[idx])
    return results

def call_openai_generation(prompt: str, max_tokens=512, temperature=0.2):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o", "gpt-4-turbo"
        messages=[
            {"role": "system", "content": "You are a RAG chatbot."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip() #type: ignore

# STREAMLIT UI

st.title("RAG Chatbot")
st.markdown(
    "This chatbot retrieves relevant information from stored documents and "
    "generates responses based on the provided context."
)
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

# Header
st.title("💬 RAG Chatbot")
st.caption("Ask any question — responses are strictly based on your uploaded data.")

# Chat input box
user_question = st.text_area("Ask your question:", placeholder="e.g. What is energy healing?", height=100)

# Retrieval settings (optional slider)
TOP_K = 3
top_k = st.slider("Number of documents to retrieve (Top K):", 1, 10, TOP_K)

# RAG prompt template

# Action button
if st.button("🔍 Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant context..."):
            retrieved = retrieve(user_question, top_k=top_k)

        if not retrieved:
            st.error("No relevant context found.")
        else:
            # Build context
            context_blocks = [
                f"Source: {r.get('source')} | chunk_id: {r.get('chunk_id')}\n{r.get('text')}"
                for r in retrieved
            ]
            context_text = "\n\n---\n\n".join(context_blocks)

            # Build prompt
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context_text,
                user_question=user_question,
            )

            # Generate answer
            with st.spinner("Generating response..."):
                answer = call_openai_generation(prompt, max_tokens=700, temperature=0.2)

            # Display answer
            st.success("✅ Response generated!")
            st.markdown("### 🧠 Answer:")
            st.write(answer)

            # Optional: show retrieved context
            with st.expander("📚 View Retrieved Context"):
                for i, r in enumerate(retrieved, 1):
                    st.markdown(f"**{i}. Source:** {r.get('source')} | Chunk: {r.get('chunk_id')}")
                    st.write(r.get('text'))
                    st.divider()