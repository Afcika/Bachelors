"""
Phase 3: Topic Segmentation (Chapter Detection)
This module detects topic changes and creates intelligent chapters.
"""

import os
import json
from typing import List, Dict
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Good balance of speed and quality (fast on CPU)
SIMILARITY_THRESHOLD = 0.65            # Lower = more chapters, Higher = fewer chapters
MIN_CHAPTER_LENGTH = 30                # Minimum seconds per chapter

def load_transcript(json_path: str) -> Dict:
    """Load transcript from Phase 2"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.success(f"Loaded transcript with {len(data['segments'])} segments")
    return data


def split_into_sentences(segments: List[Dict]) -> List[Dict]:
    """Keep original segments but prepare for embedding"""
    return segments  # We already have good segments from Whisper


def get_sentence_embeddings(sentences: List[str], model_name: str = EMBEDDING_MODEL):
    """Generate embeddings using sentence-transformers"""
    logger.info(f"Generating embeddings with model: {model_name}")
    model = SentenceTransformer(model_name)
    
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=32)
    logger.success(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings


def detect_topic_boundaries(segments: List[Dict], embeddings: np.ndarray):
    """Detect chapter boundaries using cosine similarity + sliding window"""
    logger.info("Detecting topic boundaries using cosine similarity...")

    boundaries = [0]  # First chapter always starts at 0
    
    for i in range(1, len(embeddings) - 1):
        # Compare current sentence with previous few sentences (sliding window)
        prev_emb = embeddings[i-1].reshape(1, -1)
        curr_emb = embeddings[i].reshape(1, -1)
        
        similarity = cosine_similarity(prev_emb, curr_emb)[0][0]
        
        # If similarity drops significantly → new topic
        if similarity < SIMILARITY_THRESHOLD:
            # Also check minimum chapter length to avoid too many small chapters
            current_start = segments[i]["start"]
            last_boundary_start = segments[boundaries[-1]]["start"]
            
            if current_start - last_boundary_start > MIN_CHAPTER_LENGTH:
                boundaries.append(i)

    boundaries.append(len(segments))  # Add the end
    return boundaries


def create_chapters(segments: List[Dict], boundaries: List[int], full_text: str) -> List[Dict]:
    """Group segments into chapters and generate simple titles"""
    chapters = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        chapter_segments = segments[start_idx:end_idx]
        
        if not chapter_segments:
            continue
            
        chapter_start = chapter_segments[0]["start"]
        chapter_end = chapter_segments[-1]["end"]
        
        # Combine text for this chapter
        chapter_text = " ".join([seg["text"] for seg in chapter_segments]).strip()
        
        # Simple title: take first 8-12 words (we can improve later with LLM)
        words = chapter_text.split()
        title = " ".join(words[:12])
        if len(words) > 12:
            title += "..."
        
        chapter = {
            "chapter_id": i + 1,
            "start": round(chapter_start, 2),
            "end": round(chapter_end, 2),
            "duration": round(chapter_end - chapter_start, 2),
            "title": title,
            "text": chapter_text,
            "segment_count": len(chapter_segments)
        }
        
        chapters.append(chapter)
    
    logger.success(f"Created {len(chapters)} chapters")
    return chapters


def save_chapters(chapters: List[Dict], video_id: str):
    """Save chapters as JSON"""
    output_dir = os.path.join("data", video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    chapters_data = {
        "video_id": video_id,
        "chapter_count": len(chapters),
        "chapters": chapters,
        "created_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    }
    
    json_path = os.path.join(output_dir, f"{video_id}_chapters.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chapters_data, f, ensure_ascii=False, indent=4)
    
    logger.success(f"✅ Chapters saved to: {json_path}")
    return json_path


# ====================== TEST FUNCTION ======================
if __name__ == "__main__":
    print("🚀 Starting Phase 3 - Topic Segmentation Test...\n")
    
    video_id = "erjMgola4fQ"
    transcript_path = f"data/{video_id}/{video_id}_transcript.json"
    
    if not os.path.exists(transcript_path):
        print(f"❌ Transcript not found: {transcript_path}")
        print("Please run Phase 2 first.")
    else:
        try:
            data = load_transcript(transcript_path)
            segments = data["segments"]
            full_text = data["full_text"]
            
            # Prepare sentences for embedding
            sentences = [seg["text"] for seg in segments]
            
            # Get embeddings
            embeddings = get_sentence_embeddings(sentences)
            
            # Detect boundaries
            boundaries = detect_topic_boundaries(segments, embeddings)
            
            # Create chapters
            chapters = create_chapters(segments, boundaries, full_text)
            
            # Save
            chapters_file = save_chapters(chapters, video_id)
            
            # Show sample output
            print(f"\n📋 Created {len(chapters)} chapters:")
            for ch in chapters[:5]:   # show first 5
                print(f"   Chapter {ch['chapter_id']}: [{ch['start']:.1f}s - {ch['end']:.1f}s] {ch['title']}")
            
            print(f"\n✅ Phase 3 Test PASSED!")
            print(f"Chapters saved at: {chapters_file}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()