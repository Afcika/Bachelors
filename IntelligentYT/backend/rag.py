"""
Phase 5: RAG for Q&A - Stable Version for Streamlit
"""

import os
import json
import time
from typing import List, Dict
from loguru import logger
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

TOP_K = 5
RELEVANCE_THRESHOLD = 0.32

def load_transcript(video_id: str):
    path = f"data/{video_id}/{video_id}_transcript.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(segments: List[Dict]):
    """Build FAISS index"""
    texts = [seg["text"] for seg in segments]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, embeddings


def retrieve_relevant_segments(question: str, segments: List[Dict], index=None, embeddings=None):
    """Safe retrieval function"""
    if index is None or embeddings is None:
        # Rebuild index if not available
        index, embeddings = build_vector_store(segments)
    
    # Embed question
    question_emb = embedding_model.encode([question])[0].astype('float32').reshape(1, -1)
    faiss.normalize_L2(question_emb)
    
    # Search
    distances, indices = index.search(question_emb, TOP_K)
    
    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx < len(segments):
            seg = segments[idx].copy()
            seg["relevance_score"] = float(distances[0][i])
            retrieved.append(seg)
    
    return retrieved


#  - If the question is unrelated to the video, say: "This question is not related to the video content, but also if user asks question that in not takled about video but it is connected to material say "It is not exacply taked about in video but ...(text answer)""
#     - If you don't have enough information, say: "I don't have enough information from the video to answer this but here is info i gathered from onther sources (and aswer the question)."
#     - if the qeustion is totaly not connected not even a little bit lets say video is about cars  perosn asks what is the name of certain flower say "not video related qeustion"

def generate_answer(question: str, retrieved: List[Dict], video_title: str):
    avg_relevance = sum(seg.get("relevance_score", 0) for seg in retrieved) / len(retrieved) if retrieved else 0.0

    context = "\n".join([f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}" for seg in retrieved])

    system_prompt = f"""You are an assistant that answers questions **only** about the video titled "{video_title}" .

    Rules:
    - answer any question user asks you 
    = if user asks video related question answer it normally 
    - if questiion is not video realted say "It is not mention in here but (then give real ansewer besides video, it can be your opinion)"
    - Include timestamps when referring to specific parts.
    - Be honest and clear."""

    

    user_content = f"Question: {question}\n\nRelevant video excerpts:\n{context}\n\nRelevance: {avg_relevance:.3f}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# For testing outside Streamlit
if __name__ == "__main__":
    print("RAG module loaded successfully.")