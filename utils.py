# utils.py
import os
from PyPDF2 import PdfReader

def load_text_from_file(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(path)
        texts = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)
    else:
        raise ValueError("Unsupported file type: " + ext)


def chunk_text(text: str, chunk_size=500, overlap=50):
    # naive chunker by words
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks
