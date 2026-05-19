"""
Intelligent YouTube Video Summarizer - Stable Version
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.downloader import get_video_metadata, download_audio
from backend.transcriber import transcribe_audio, save_transcript
from backend.segmenter import load_transcript as load_transcript_seg, get_sentence_embeddings, detect_topic_boundaries, create_chapters, save_chapters
from backend.summarizer import generate_hierarchical_summary
from backend.rag import build_vector_store, retrieve_relevant_segments, generate_answer

st.set_page_config(page_title="Intelligent YT Summarizer", layout="wide")

st.title("🎥 Intelligent YouTube Video Summarizer")
st.markdown("**Bachelor Thesis Project**")

with st.sidebar:
    st.header("Settings")
    summary_style = st.selectbox("Summary Style", ["Student", "Expert", "Short", "Detailed"])

tab1, tab2, tab3 = st.tabs(["📥 Input Video", "📊 Results", "💬 Q&A"])

with tab1:
    st.subheader("Enter YouTube URL")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("🚀 Process Video", type="primary"):
        if not url:
            st.error("Please enter a YouTube URL")
        else:
            try:
                with st.spinner("Processing video..."):
                    progress = st.progress(0)
                    status = st.empty()

                    status.text("1/5 → Fetching metadata...")
                    metadata = get_video_metadata(url)
                    video_id = metadata["video_id"]
                    progress.progress(20)

                    status.text("2/5 → Downloading audio...")
                    audio_path = download_audio(url, metadata)
                    progress.progress(40)

                    status.text("3/5 → Transcribing...")
                    segments, full_text, lang = transcribe_audio(audio_path)
                    save_transcript(segments, full_text, video_id, lang)
                    progress.progress(60)

                    status.text("4/5 → Creating chapters...")
                    transcript_data = load_transcript_seg(f"data/{video_id}/{video_id}_transcript.json")
                    sentences = [seg["text"] for seg in transcript_data["segments"]]
                    embeddings = get_sentence_embeddings(sentences)
                    boundaries = detect_topic_boundaries(transcript_data["segments"], embeddings)
                    chapters = create_chapters(transcript_data["segments"], boundaries, full_text)
                    save_chapters(chapters, video_id)
                    progress.progress(80)

                    status.text("5/5 → Generating summary...")
                    final_summary = generate_hierarchical_summary(video_id)
                    progress.progress(100)

                    # Save to session state
                    st.session_state.video_id = video_id
                    st.session_state.metadata = metadata
                    st.session_state.summary = final_summary
                    st.session_state.chapters = chapters
                    st.session_state.segments = transcript_data["segments"]

                    st.success("✅ Processing completed!")

            except Exception as e:
                st.error(f"❌ Error: {e}")

with tab2:
    if 'metadata' in st.session_state:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state.metadata.get('thumbnail'), use_column_width=True)
        with col2:
            st.subheader(st.session_state.metadata.get('title', ''))
            st.write(f"**Channel:** {st.session_state.metadata.get('channel', '')}")

        st.subheader("📝 Summary")
        st.write(st.session_state.get('summary', 'No summary yet'))

        st.subheader("📑 Chapters")
        for ch in st.session_state.get('chapters', []):
            st.write(f"**Chapter {ch.get('chapter_id')}**: {ch.get('title')} [{ch.get('start',0):.1f}s - {ch.get('end',0):.1f}s]")
    else:
        st.info("Process a video first")

with tab3:
    st.subheader("💬 Ask Questions")
    if 'video_id' in st.session_state:
        question = st.text_input("Ask a question about the video:")

        if st.button("Get Answer"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    # Rebuild vector store safely
                    transcript_data = load_transcript_seg(f"data/{st.session_state.video_id}/{st.session_state.video_id}_transcript.json")
                    index, embeddings = build_vector_store(transcript_data["segments"])
                    
                    retrieved = retrieve_relevant_segments(question, transcript_data["segments"], index, embeddings)
                    answer = generate_answer(question, retrieved, st.session_state.metadata.get('title', 'Video'))
                    
                    st.write("**Answer:**")
                    st.write(answer)
            else:
                st.warning("Please enter a question")
    else:
        st.info("Process a video first")

st.caption("Intelligent YouTube Summarizer | Bachelor Thesis")