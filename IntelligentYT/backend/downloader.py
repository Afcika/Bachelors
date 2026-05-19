"""
Phase 1: Robust YouTube Video Downloader
This module downloads only audio and gets all metadata safely.
"""

import os
from loguru import logger
import yt_dlp
from datetime import timedelta

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

def get_video_metadata(url: str):
    """
    Gets title, channel, duration, thumbnail, etc. WITHOUT downloading.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,   # fast metadata only
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            metadata = {
                "video_id": info.get("id"),
                "title": info.get("title", "Unknown Title"),
                "channel": info.get("uploader", "Unknown Channel"),
                "duration": info.get("duration"),                    # in seconds
                "duration_formatted": str(timedelta(seconds=info.get("duration", 0))),
                "thumbnail": info.get("thumbnail"),
                "view_count": info.get("view_count"),
                "upload_date": info.get("upload_date"),
                "url": url,
            }
            
            logger.success(f"✅ Metadata extracted: {metadata['title']}")
            return metadata
            
    except Exception as e:
        logger.error(f"❌ Failed to get metadata: {e}")
        raise Exception(f"Could not get video info. Maybe private video or wrong link? Error: {e}")


def download_audio(url: str, metadata: dict):
    """
    Downloads ONLY audio (best quality) and saves it in data/ folder.
    Returns the full path to the audio file.
    """
    # Clean title for filename (removes bad characters)
    safe_title = "".join(c for c in metadata["title"] if c.isalnum() or c in " -_").strip()
    safe_title = safe_title[:80]  # not too long
    
    output_path = os.path.join("data", f"{metadata['video_id']}_{safe_title}")
    
    ydl_opts = {
        'format': 'bestaudio/best',           # best audio only
        'outtmpl': f"{output_path}.%(ext)s",  # save path
        'quiet': False,                       # show progress
        'no_warnings': True,
        'progress_hooks': [lambda d: logger.info(f"Downloading... {d.get('status', '')}")],
        'postprocessors': [{                  # convert to mp3
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    try:
        logger.info(f"📥 Starting audio download: {metadata['title']}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        audio_file = f"{output_path}.mp3"
        
        if os.path.exists(audio_file):
            logger.success(f"✅ Audio downloaded successfully: {audio_file}")
            return audio_file
        else:
            raise Exception("Audio file not found after download")
            
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise Exception(f"Download failed: {e}")


# ====================== TEST FUNCTION ======================
if __name__ == "__main__":
    # Change this URL to test different videos
    test_url = "https://www.youtube.com/watch?v=erjMgola4fQ"   # Rick Astley (safe test)
    
    print("🚀 Starting Phase 1 Test...\n")
    
    try:
        metadata = get_video_metadata(test_url)
        print("\n📋 METADATA:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        audio_path = download_audio(test_url, metadata)
        print(f"\n🎵 Audio saved at: {audio_path}")
        print("\n✅ Phase 1 Test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")