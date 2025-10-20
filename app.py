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

load_dotenv()

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
            {"role": "system", "content": "You are a healing assistant using YPV knowledge."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip() #type: ignore

# STREAMLIT UI

st.title("🧘‍♀️ YPV Healer RAG Chatbot")
st.markdown(
    "This chatbot retrieves relevant healing information from stored documents and "
    "generates responses based on the provided context."
)

# Patient info
st.subheader("🩺 Patient Information")
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Name", "")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
with col2:
    history = st.text_area("Patient History", "")
conditions = st.text_input("Known Conditions (comma-separated)", "")

# Question
st.subheader("💬 Ask a Healing Question")
user_question = st.text_area("Enter your question:", placeholder="e.g. How to heal chronic back pain?")
top_k = st.slider("Number of documents to retrieve (Top K):", 1, 10, TOP_K)

# Submit
if st.button("🔍 Get Healing Guidance"):
    if not user_question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving relevant context..."):
            retrieved = retrieve(user_question, top_k=top_k)

        if not retrieved:
            st.error("No relevant documents found.")
        else:
            # Build context
            context_blocks = [
                f"Source: {r.get('source')} | chunk_id: {r.get('chunk_id')}\n{r.get('text')}"
                for r in retrieved
            ]
            context_text = "\n\n---\n\n".join(context_blocks)

            # Patient summary
            patient_summary = (
                f"Patient name: {name}\nAge: {age}\nHistory: {history}\n"
                f"Known conditions: {conditions}\n"
            )

            # Build RAG prompt
            prompt = RAG_PROMPT_TEMPLATE.format(
                system_instructions=(
                    "You are a healer assistant. Use ONLY the information from the provided CONTEXT "
                    "to make recommendations. If the answer is not in the context, say you couldn't "
                    "find clear guidance."
                ),
                context=context_text,
                patient_info=patient_summary,
                user_question=user_question,
            )

            # Generate answer
            with st.spinner("Generating healing guidance..."):
                answer = call_openai_generation(prompt, max_tokens=700, temperature=0.2)

            st.success("✅ Response generated!")
            st.markdown("### 🧘‍♂️ Healing Recommendation:")
            st.write(answer)

            # Optional: show retrieved docs
            with st.expander("📚 View Retrieved Context"):
                for i, r in enumerate(retrieved, 1):
                    st.markdown(f"**{i}. Source:** {r.get('source')} | Chunk: {r.get('chunk_id')}")
                    st.write(r.get('text'))
                    st.divider()
