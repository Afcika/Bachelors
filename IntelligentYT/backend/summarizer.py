"""
Improved Summarizer - Designed for Higher ROUGE Scores
"""

import os
import json
import time
from loguru import logger
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_chapters(video_id: str):
    path = f"data/{video_id}/{video_id}_chapters.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_chapter(text: str, title: str) -> str:
    """Better prompt for higher ROUGE"""
    prompt = f"""Summarize the following chapter clearly and concisely.

Chapter Title: {title}

Text:
{text}

Instructions for high quality summary:
- Use clear and direct language
- Include the most important key points and ideas
- Keep the same meaning and tone as the original
- Write 2 to 5 sentences only
- Do not add extra opinions or information

Chapter Summary:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,      # Lower temperature = more consistent
            max_tokens=450
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Chapter error: {e}")
        return text[:350]


def generate_overall_summary(chapters: list) -> str:
    """Strong prompt for better ROUGE"""
    chapter_blocks = []
    for ch in chapters:
        chapter_blocks.append(f"Chapter {ch.get('chapter_id')}: {ch.get('title','')}\n{ch.get('chapter_summary','')}")

    context = "\n\n".join(chapter_blocks)

    prompt = f"""Write a concise, meaningful, and high-quality overall summary of the video.

Chapter Summaries:
{context}

Requirements (Very Important):
Requirements (Very Important):
- Write a concise but comprehensive overall summary of the entire video.
- The summary should be between 3-6 sentences long (its is basicly up you).
- Start with a strong TL;DR that captures the main idea of the video.
- Focus only on the key points and core message presented in the video.
- Do not add any new information, opinions, or details that are not in the video.
- Use clear, formal academic language.
- Make the summary easy to understand while remaining accurate and faithful to the original content.
- The reader should fully understand the main purpose and important takeaways of the video after reading your summary.
Overall Summary:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=750
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Overall summary error: {e}")
        return "Summary could not be generated."


def generate_hierarchical_summary(video_id: str):
    """Main function - Do not change name"""
    logger.info(f"Generating improved summary for {video_id}")

    data = load_chapters(video_id)
    chapters = data["chapters"]

    # Step 1: Summarize each chapter
    for chapter in chapters:
        text = chapter.get("text", "")
        title = chapter.get("title", f"Chapter {chapter.get('chapter_id')}")
        chapter["chapter_summary"] = summarize_chapter(text, title)

    # Step 2: Overall summary
    final_summary = generate_overall_summary(chapters)

    # Save
    result = {
        "video_id": video_id,
        "summary": final_summary,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    save_path = f"data/{video_id}/{video_id}_hierarchical_summary.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    logger.success(f"✅ New summary saved for {video_id}")
    return final_summary


if __name__ == "__main__":
    test_id = "6-TDjSIroRQ"   # Change this to test different videos
    summary = generate_hierarchical_summary(test_id)
    print("\n=== GENERATED SUMMARY ===\n")
    print(summary)