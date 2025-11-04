import streamlit as st
import requests
import whisper
import os
from pathlib import Path
import tempfile
from typing import Optional
import yt_dlp
import time

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api"

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = OLLAMA_API_URL):
        self.base_url = base_url
        self.generate_url = f"{base_url}/generate"
    
    def generate(self, model: str, prompt: str, stream: bool = False):
        """Generate response from Ollama"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Ollama: {e}")
            return None
    
    def list_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=10)
            response.raise_for_status()
            return response.json().get('models', [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching models: {e}")
            return []

class AudioTranscriber:
    """Handles audio transcription using Whisper"""
    
    def __init__(self):
        self.model = None
        self.model_size = "base"
    
    def load_model(self, model_size: str = "base"):
        """Load Whisper model"""
        try:
            with st.spinner(f"Loading Whisper {model_size} model..."):
                self.model = whisper.load_model(model_size)
                self.model_size = model_size
            return True
        except Exception as e:
            st.error(f"Error loading Whisper model: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Optional[dict]:
        """Transcribe audio file"""
        if self.model is None:
            st.error("Whisper model not loaded!")
            return None
        
        try:
            with st.spinner("Transcribing audio... This may take a few minutes."):
                result = self.model.transcribe(audio_path)
            return result
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return None

class PodcastDownloader:
    """Downloads audio from podcast URLs (YouTube, Spotify, etc.)"""
    
    @staticmethod
    def download_audio(url: str, output_dir: str) -> Optional[str]:
        """Download audio from URL using yt-dlp"""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
            }
            
            with st.spinner("Downloading audio from URL..."):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                    # Change extension to mp3
                    audio_file = os.path.splitext(filename)[0] + '.mp3'
                    
                    if os.path.exists(audio_file):
                        return audio_file
                    return None
        except Exception as e:
            st.error(f"Error downloading audio: {e}")
            return None

class PodcastSummarizer:
    """Main class for podcast summarization"""
    
    def __init__(self, ollama_client: OllamaClient, transcriber: AudioTranscriber):
        self.ollama_client = ollama_client
        self.transcriber = transcriber
    
    def create_summary_prompt(self, transcript: str, summary_type: str) -> str:
        """Create prompt for Ollama based on summary type"""
        
        base_prompt = f"""You are an expert podcast summarizer. Below is a transcript of a podcast episode.

TRANSCRIPT:
{transcript}

"""
        
        if summary_type == "Brief Overview":
            prompt = base_prompt + """Please provide a BRIEF SUMMARY (2-3 paragraphs) that captures:
1. The main topic and purpose of the podcast
2. Key points discussed
3. Main takeaways

Keep it concise and engaging."""
        
        elif summary_type == "Detailed Summary":
            prompt = base_prompt + """Please provide a DETAILED SUMMARY that includes:
1. Introduction and context
2. Main topics and subtopics discussed in order
3. Key insights and important points
4. Notable quotes or examples
5. Conclusions and takeaways

Be thorough but organized."""
        
        elif summary_type == "Key Points Only":
            prompt = base_prompt + """Please extract and list the KEY POINTS from this podcast:
- List 8-12 main points
- Each point should be 1-2 sentences
- Focus on actionable insights and important information
- Use bullet points for clarity"""
        
        elif summary_type == "Q&A Format":
            prompt = base_prompt + """Please create a Q&A SUMMARY by:
1. Identifying the main questions or topics discussed
2. Providing clear answers for each
3. Format as: "Q: [question]" followed by "A: [answer]"
4. Include 5-8 key Q&As"""
        
        else:  # Custom
            prompt = base_prompt + """Please provide a comprehensive summary of this podcast episode."""
        
        return prompt
    
    def generate_summary(self, transcript: str, summary_type: str, model: str) -> Optional[str]:
        """Generate summary using Ollama"""
        prompt = self.create_summary_prompt(transcript, summary_type)
        
        with st.spinner("Generating summary with AI..."):
            response = self.ollama_client.generate(model, prompt)
            
            if response and 'response' in response:
                return response['response']
            return None

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = AudioTranscriber()
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = OllamaClient()
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = PodcastSummarizer(
            st.session_state.ollama_client,
            st.session_state.transcriber
        )
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'whisper_loaded' not in st.session_state:
        st.session_state.whisper_loaded = False

def main():
    st.set_page_config(
        page_title="Podcast Summarizer",
        page_icon="🎙️",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🎙️ Podcast Summarizer with Ollama")
    st.markdown("Transcribe and summarize podcasts using Whisper + Ollama")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Whisper model selection
        st.subheader("🎤 Transcription Settings")
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        if not st.session_state.whisper_loaded:
            if st.button("Load Whisper Model", type="primary"):
                if st.session_state.transcriber.load_model(whisper_model):
                    st.session_state.whisper_loaded = True
                    st.success("✅ Whisper model loaded!")
                    st.rerun()
        else:
            st.success(f"✅ Whisper {st.session_state.transcriber.model_size} loaded")
            if st.button("Change Model"):
                st.session_state.whisper_loaded = False
                st.session_state.transcriber.model = None
                st.rerun()
        
        st.divider()
        
        # Ollama model selection
        st.subheader("🤖 Summarization Settings")
        models = st.session_state.ollama_client.list_models()
        if models:
            model_names = [m['name'] for m in models]
            selected_model = st.selectbox(
                "Ollama Model",
                model_names,
                index=0 if model_names else None
            )
        else:
            st.warning("No Ollama models found")
            selected_model = st.text_input("Enter model name", "llama2")
        
        summary_type = st.selectbox(
            "Summary Type",
            ["Brief Overview", "Detailed Summary", "Key Points Only", "Q&A Format"],
            help="Choose how you want the podcast summarized"
        )
        
        st.divider()
        st.caption("Made with ❤️ using Whisper & Ollama")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🔗 Download from URL", "📝 Results"])
    
    with tab1:
        st.header("Upload Audio File")
        st.markdown("Upload an audio file (MP3, WAV, M4A, etc.) to transcribe and summarize")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
            help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🎯 Process Audio", type="primary", use_container_width=True):
                    if not st.session_state.whisper_loaded:
                        st.error("Please load Whisper model first from the sidebar!")
                    else:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Transcribe
                            result = st.session_state.transcriber.transcribe_audio(tmp_path)
                            
                            if result:
                                st.session_state.transcript = result['text']
                                st.success("✅ Transcription complete!")
                                
                                # Generate summary
                                summary = st.session_state.summarizer.generate_summary(
                                    st.session_state.transcript,
                                    summary_type,
                                    selected_model
                                )
                                
                                if summary:
                                    st.session_state.summary = summary
                                    st.success("✅ Summary generated!")
                                    st.info("📝 Check the 'Results' tab to view transcript and summary")
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
    
    with tab2:
        st.header("Download from URL")
        st.markdown("Enter a podcast URL (YouTube, SoundCloud, etc.) to download and process")
        
        url_input = st.text_input(
            "Podcast URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports YouTube, SoundCloud, and many other platforms"
        )
        
        if url_input:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🎯 Download & Process", type="primary", use_container_width=True):
                    if not st.session_state.whisper_loaded:
                        st.error("Please load Whisper model first from the sidebar!")
                    else:
                        # Create temp directory
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            # Download audio
                            audio_file = PodcastDownloader.download_audio(url_input, tmp_dir)
                            
                            if audio_file and os.path.exists(audio_file):
                                st.success(f"✅ Downloaded: {Path(audio_file).name}")
                                
                                # Transcribe
                                result = st.session_state.transcriber.transcribe_audio(audio_file)
                                
                                if result:
                                    st.session_state.transcript = result['text']
                                    st.success("✅ Transcription complete!")
                                    
                                    # Generate summary
                                    summary = st.session_state.summarizer.generate_summary(
                                        st.session_state.transcript,
                                        summary_type,
                                        selected_model
                                    )
                                    
                                    if summary:
                                        st.session_state.summary = summary
                                        st.success("✅ Summary generated!")
                                        st.info("📝 Check the 'Results' tab to view transcript and summary")
                            else:
                                st.error("Failed to download audio. Please check the URL.")
    
    with tab3:
        st.header("Results")
        
        if st.session_state.transcript or st.session_state.summary:
            # Summary section
            if st.session_state.summary:
                st.subheader("📊 AI-Generated Summary")
                st.markdown(st.session_state.summary)
                
                # Download summary button
                st.download_button(
                    label="💾 Download Summary",
                    data=st.session_state.summary,
                    file_name="podcast_summary.txt",
                    mime="text/plain"
                )
                
                st.divider()
            
            # Transcript section
            if st.session_state.transcript:
                with st.expander("📄 View Full Transcript", expanded=False):
                    st.text_area(
                        "Transcript",
                        value=st.session_state.transcript,
                        height=400,
                        label_visibility="collapsed"
                    )
                    
                    # Download transcript button
                    st.download_button(
                        label="💾 Download Transcript",
                        data=st.session_state.transcript,
                        file_name="podcast_transcript.txt",
                        mime="text/plain"
                    )
            
            # Regenerate summary option
            st.divider()
            if st.button("🔄 Regenerate Summary with Current Settings"):
                if st.session_state.transcript:
                    summary = st.session_state.summarizer.generate_summary(
                        st.session_state.transcript,
                        summary_type,
                        selected_model
                    )
                    if summary:
                        st.session_state.summary = summary
                        st.rerun()
        else:
            st.info("👈 Upload an audio file or enter a URL to get started!")
            
            # Show example
            with st.expander("ℹ️ How it works"):
                st.markdown("""
                ### Process Flow:
                
                1. **Input Audio**: Upload a file or provide a URL
                2. **Transcription**: Whisper AI converts speech to text
                3. **Summarization**: Ollama LLM generates an intelligent summary
                4. **Review**: View and download your results
                
                ### Tips:
                - **Whisper Model Size**: 
                  - `tiny/base`: Fast, good for clear audio
                  - `small/medium`: Better accuracy
                  - `large`: Best quality, slowest
                
                - **Summary Types**:
                  - `Brief Overview`: Quick 2-3 paragraph summary
                  - `Detailed Summary`: Comprehensive breakdown
                  - `Key Points Only`: Bullet point highlights
                  - `Q&A Format`: Question and answer style
                
                - **Supported URLs**: YouTube, SoundCloud, and 1000+ sites
                """)

if __name__ == "__main__":
    main()
