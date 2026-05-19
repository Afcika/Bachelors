"""
Phase 2: Accurate Transcription Module using faster-whisper
"""

import os
import json
import time
from pathlib import Path
from loguru import logger
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Configuration - Change these if needed
MODEL_SIZE = "large-v3-turbo"   # Options: "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"
DEVICE = "cpu"                  # Change to "cuda" if you have NVIDIA GPU and want it faster
COMPUTE_TYPE = "float32"        # "float16" is faster on GPU, "float32" safer on CPU

def normalize_audio(audio_path: str) -> str:
    """Simple audio normalization (makes volume consistent)"""
    try:
        logger.info("🔧 Normalizing audio volume...")
        audio = AudioSegment.from_mp3(audio_path)
        # Normalize to -16 dBFS (good standard for speech)
        normalized_audio = audio.normalize(headroom=-16.0)
        
        normalized_path = audio_path.replace(".mp3", "_normalized.mp3")
        normalized_audio.export(normalized_path, format="mp3")
        
        logger.success("✅ Audio normalized")
        return normalized_path
    except Exception as e:
        logger.warning(f"⚠️ Normalization skipped: {e}")
        return audio_path  # return original if fails


def transcribe_audio(audio_path: str, language: str = None):
    """
    Main transcription function.
    Returns: list of segments + full text
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"🎤 Starting transcription with model: {MODEL_SIZE} on {DEVICE}")
    
    # Optional: normalize audio first (helps with quiet recordings)
    audio_to_use = normalize_audio(audio_path)

    # Load the Whisper model
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root="models"   # saves model in your project/models folder
    )

    start_time = time.time()

    # Run transcription
    segments, info = model.transcribe(
        audio_to_use,
        language=language,           # None = auto detect
        beam_size=5,
        word_timestamps=False,       # Set True if you want word-level (slower)
        vad_filter=True,             # Removes silence automatically
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Convert segments to list of dictionaries
    transcript_segments = []
    full_text = ""

    for segment in segments:
        segment_dict = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        }
        transcript_segments.append(segment_dict)
        full_text += segment.text + " "

    full_text = full_text.strip()

    processing_time = time.time() - start_time

    logger.success(f"✅ Transcription completed in {processing_time:.1f} seconds")
    logger.info(f"Total segments: {len(transcript_segments)} | Total words ≈ {len(full_text.split())}")

    # Clean up normalized file if created
    if audio_to_use != audio_path and os.path.exists(audio_to_use):
        try:
            os.remove(audio_to_use)
        except:
            pass

    return transcript_segments, full_text, info.language


def save_transcript(transcript_segments: list, full_text: str, video_id: str, language: str):
    """Save transcript as nice JSON file"""
    output_dir = os.path.join("data", video_id)
    os.makedirs(output_dir, exist_ok=True)

    transcript_data = {
        "video_id": video_id,
        "language": language,
        "full_text": full_text,
        "segments": transcript_segments,
        "segment_count": len(transcript_segments),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    json_path = os.path.join(output_dir, f"{video_id}_transcript.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=4)

    logger.success(f"✅ Transcript saved to: {json_path}")
    return json_path


# ====================== TEST FUNCTION ======================
if __name__ == "__main__":
    print("🚀 Starting Phase 2 Transcription Test...\n")
    
    # Use the same video we downloaded in Phase 1
    video_id = "erjMgola4fQ"
    audio_path = f"data/{video_id}_A1 English Listening Practice - Language Learning.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        print("Please run Phase 1 first to download the audio.")
    else:
        try:
            segments, full_text, lang = transcribe_audio(audio_path)
            
            print(f"\n📋 Transcription Results:")
            print(f"Language: {lang}")
            print(f"Number of segments: {len(segments)}")
            print(f"Total text length: {len(full_text)} characters")
            
            # Print first 3 segments as example
            print("\nFirst 3 segments:")
            for seg in segments[:3]:
                print(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
            
            # Save the transcript
            json_file = save_transcript(segments, full_text, video_id, lang)
            
            print(f"\n✅ Phase 2 Test PASSED!")
            print(f"Transcript saved at: {json_file}")
            
        except Exception as e:
            print(f"\n❌ Transcription failed: {e}")
            import traceback
            traceback.print_exc()