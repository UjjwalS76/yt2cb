import streamlit as st
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from typing import Optional, Dict, List

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatAnyscale

# Ensure the Perplexity API key is set in Streamlit secrets
os.environ["ANYSCALE_API_KEY"] = st.secrets["ANYSCALE_API_KEY"]

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info_from_transcript(transcript_list) -> Dict:
    """Gets basic video information from the transcript metadata."""
    try:
        transcript = transcript_list.find_transcript(["en"])
        video_info = transcript.video_metadata
        return {
            "title": video_info.get("title", "Unknown Title"),
            "duration": video_info.get("duration", 0),
        }
    except NoTranscriptFound:
        st.warning("English transcript not found. Attempting to use other available transcripts.")
        try:
            transcript = transcript_list.find_transcript([])
            video_info = transcript.video_metadata
            return {
                "title": video_info.get("title", "Unknown Title"),
                "duration": video_info.get("duration", 0),
            }
        except Exception as e:
            st.warning(f"Could not fetch video metadata: {e}")
            return {"title": "Unknown Title", "duration": 0}
    except Exception as e:
        st.warning(f"Could not fetch video metadata: {e}")
        return {"title": "Unknown Title", "duration": 0}

def load_video_transcript(video_url: str) -> Optional[List[Document]]:
    """Loads and processes the YouTube video transcript."""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Could not extract video ID from URL.")
            return None

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print(f"Transcript List: {transcript_list}")

        video_info = get_video_info_from_transcript(transcript_list)
        st.write(f"📺 **Video Title:** {video_info['title']}")

        try:
            transcript = transcript_list.find_generated_transcript(["en"])
            st.write("Using generated English transcript.")
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_manually_created_transcript(["en"])
                st.write("Using manually created English transcript.")
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_generated_transcript([])
                    transcript = transcript.translate("en")
                    st.write("Using translated generated transcript.")
                except NoTranscriptFound:
                    try:
                        transcript = transcript_list.find_manually_created_transcript([])
                        transcript = transcript.translate("en")
                        st.write("Using translated manually created transcript.")
                    except Exception as e:
                        st.error(f"No suitable transcript found: {e}")
                        return None

        transcript_pieces = transcript.fetch()
        transcript_text = " ".join([t["text"] for t in transcript_pieces])

        doc = Document(
            page_content=transcript_text,
            metadata={
                "source": video_id,
                "title": video_info["title"],
                "url": video_url,
            },
        )

        st.success("Transcript loaded successfully!")
        return [doc]

    except Exception as e:
        st.error(f"Error loading transcript: {e}")
        return None

def setup_qa_chain(transcript_docs):
    """Sets up the QA chain with FAISS vector store and Perplexity model."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(transcript_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = ChatAnyscale(
            model_name="mistralai/llama-3-sonar-small-32k-online", temperature=0.7
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=False,
            verbose=False,
        )

        return qa_chain

    except Exception as e:
        st.error(f"Error in setup_qa_chain: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit app."""
    st.title("💬 YouTube Video Chat Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "video_loaded" not in st.session_state:
        st.session_state.video_loaded = False
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    with st.sidebar:
        st.markdown(
            """
        ### How to use:
        1. Enter a YouTube video URL.
        2. Wait for transcript processing.
        3. Ask questions about the video.

        ### Notes:
        - Video must have English captions.
        - Links must be from youtube.com or youtu.be.
        """
        )

    video_url = st.text_input(
        "Enter YouTube Video URL:", help="Paste the full YouTube video URL here"
    )

    if video_url and st.button("Load Video", type="primary"):
        st.session_state.video_loaded = False
        st.session_state.chat_history = []

        with st.spinner("Processing video..."):
            transcript = load_video_transcript(video_url)
            if transcript:
                qa_chain = setup_qa_chain(transcript)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.video_loaded = True
                    st.success("Ready to chat about the video!")

    if st.session_state.video_loaded and st.session_state.qa_chain:
        st.markdown("### Chat")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask about the video..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": prompt})
                        st.write(response["answer"])
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response["answer"]}
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    elif not video_url:
        st.info("👆 Start by entering a YouTube video URL above")

if __name__ == "__main__":
    main()
