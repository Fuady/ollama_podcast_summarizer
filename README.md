# 🎙️ Podcast Summarizer with Ollama & Whisper

An AI-powered application that transcribes and summarizes podcasts using OpenAI's Whisper for speech-to-text and Ollama for intelligent summarization. Upload audio files or provide URLs to get instant, high-quality summaries!

## 🌟 Features

- **🎤 Audio Transcription**: Convert speech to text using Whisper AI
- **🤖 AI Summarization**: Generate intelligent summaries with Ollama
- **📁 File Upload**: Support for MP3, WAV, M4A, OGG, FLAC formats
- **🔗 URL Download**: Automatically download audio from YouTube, SoundCloud, and 1000+ sites
- **📊 Multiple Summary Styles**: Brief, Detailed, Key Points, Q&A formats
- **💾 Export Results**: Download transcripts and summaries as text files
- **⚡ GPU Acceleration**: Optional GPU support for faster processing

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)

---

## 🔄 How It Works

```
Audio Input → Whisper Transcription → Ollama Summarization → Summary Output
```

1. **Input**: You provide audio (file or URL)
2. **Whisper**: Converts audio to text transcript
3. **Ollama**: Analyzes transcript and generates summary
4. **Output**: View and download transcript + summary

**Note**: Ollama alone cannot process audio - it needs text input. That's why we use Whisper to convert audio to text first!

---

## 🔧 Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   python --version
   ```

2. **Ollama** (for AI summarization)
   - Download: https://ollama.ai/download
   - Install and pull a model:
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```

3. **FFmpeg** (for audio processing)
   
   **Windows:**
   - Download from: https://ffmpeg.org/download.html
   - Or use Chocolatey: `choco install ffmpeg`
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux:**
   ```bash
   sudo apt install ffmpeg  # Ubuntu/Debian
   sudo yum install ffmpeg  # CentOS/RHEL
   ```
   
   **Verify installation:**
   ```bash
   ffmpeg -version
   ```

---

## 📥 Installation

### Step 1: Setup Project

```bash
# Create project directory
mkdir podcast-summarizer
cd podcast-summarizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# If you encounter errors, install packages individually:
pip install streamlit
pip install openai-whisper
pip install yt-dlp
pip install requests
pip install torch
```

**Note for Windows Users:**
If you get compilation errors, install Visual C++ Build Tools from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Note for macOS M1/M2 Users:**
```bash
# Install PyTorch with Metal support
pip install torch torchvision torchaudio
```

### Step 3: Verify Installations

```bash
# Check Whisper
python -c "import whisper; print('Whisper OK')"

# Check Ollama
curl http://localhost:11434/api/tags

# Check FFmpeg
ffmpeg -version
```

---

## 🚀 Running the Application

### 1. Start Ollama Server

```bash
# Start Ollama (usually auto-starts, but verify)
ollama serve

# In another terminal, verify it's running:
ollama list
```

### 2. Launch the Application

```bash
streamlit run podcast_summarizer.py
```

The app will open in your browser at `http://localhost:8501`

---

## 💡 Usage Guide

### Method 1: Upload Audio File

1. Click on the **"📁 Upload Audio"** tab
2. Click **"Load Whisper Model"** in the sidebar (first time only)
3. Choose a Whisper model size:
   - `tiny`: Fastest, less accurate (39M params)
   - `base`: Good balance (74M params) ⭐ **Recommended**
   - `small`: Better accuracy (244M params)
   - `medium`: High accuracy (769M params)
   - `large`: Best accuracy (1550M params) - requires GPU
4. Drag & drop or browse for your audio file
5. Select your preferred summary type
6. Click **"🎯 Process Audio"**
7. Wait for transcription and summarization
8. View results in the **"📝 Results"** tab

### Method 2: Download from URL

1. Click on the **"🔗 Download from URL"** tab
2. Click **"Load Whisper Model"** in the sidebar (if not loaded)
3. Paste a podcast URL (examples below)
4. Select summary type
5. Click **"🎯 Download & Process"**
6. The app will:
   - Download the audio
   - Transcribe it
   - Generate a summary
7. View results in the **"📝 Results"** tab

**Supported URL Sources:**
- YouTube: `https://www.youtube.com/watch?v=...`
- SoundCloud: `https://soundcloud.com/...`
- Spotify Podcasts: `https://open.spotify.com/episode/...`
- Direct MP3 links: `https://example.com/podcast.mp3`
- And 1000+ other sites supported by yt-dlp

### Summary Types Explained

1. **Brief Overview** (2-3 paragraphs)
   - Quick summary of main points
   - Best for: Getting the gist quickly

2. **Detailed Summary** (comprehensive)
   - Full breakdown of all topics
   - Includes context, examples, conclusions
   - Best for: Deep understanding

3. **Key Points Only** (bullet points)
   - 8-12 main takeaways
   - Concise and actionable
   - Best for: Quick reference

4. **Q&A Format** (question-answer pairs)
   - 5-8 main questions with answers
   - Easy to scan
   - Best for: Interview-style podcasts

---

## ⚙️ Configuration Options

### Whisper Model Selection

| Model | Size | Speed | Accuracy | RAM | Best For |
|-------|------|-------|----------|-----|----------|
| tiny | 39M | Very Fast | Fair | 1GB | Testing, clear audio |
| base | 74M | Fast | Good | 1GB | General use ⭐ |
| small | 244M | Medium | Better | 2GB | Good quality |
| medium | 769M | Slow | Great | 5GB | High accuracy |
| large | 1550M | Very Slow | Best | 10GB | Professional transcription |

**Recommendations:**
- **Fast PC, clear audio**: Use `base` or `small`
- **Slow PC**: Use `tiny`
- **GPU available**: Use `medium` or `large`
- **Noisy audio, accents**: Use `large`

### Ollama Model Selection

- **llama2**: Good general purpose summarization
- **mistral**: Excellent quality summaries
- **llama2:13b**: More detailed, coherent summaries
- **codellama**: Better at technical podcasts

---

## 🎯 Real-World Examples

### Example 1: Summarizing a Tech Podcast

```
Input: 45-minute podcast about AI developments
Whisper Model: base
Ollama Model: llama2
Summary Type: Detailed Summary
Processing Time: ~5-8 minutes
Output: 
  - Full transcript: 15,000 words
  - Summary: 800 words with key insights
```

### Example 2: Quick News Podcast Summary

```
Input: 10-minute daily news podcast
Whisper Model: tiny
Ollama Model: mistral
Summary Type: Key Points Only
Processing Time: ~2-3 minutes
Output:
  - 10 bullet points covering main stories
```

### Example 3: Interview Podcast

```
Input: YouTube interview (1.5 hours)
Whisper Model: small
Ollama Model: llama2:13b
Summary Type: Q&A Format
Processing Time: ~15-20 minutes
Output:
  - Q&A pairs covering main discussion points
```

---

## 🔍 Troubleshooting

### Problem: "FFmpeg not found"

**Solution:**
```bash
# Install FFmpeg (see Prerequisites section)
# Then verify:
ffmpeg -version

# If still not working, restart terminal/IDE
```

### Problem: Whisper model download fails

**Solution:**
```bash
# Manually download models
python -c "import whisper; whisper.load_model('base')"

# Or specify cache location:
export WHISPER_CACHE_DIR=/path/to/cache
```

### Problem: "CUDA out of memory" error

**Solutions:**
1. Use a smaller Whisper model (tiny or base)
2. Process shorter audio segments
3. Use CPU instead:
```python
# In the code, modify:
device = "cpu"  # instead of "cuda"
```

### Problem: YouTube download fails

**Solutions:**
1. Update yt-dlp:
```bash
pip install --upgrade yt-dlp
```

2. Check if URL is accessible
3. Some videos may be restricted by region/copyright

### Problem: Slow transcription

**Solutions:**
1. Use smaller Whisper model (`tiny` or `base`)
2. Enable GPU acceleration (if available)
3. Process shorter audio clips
4. Upgrade RAM if possible

### Problem: Poor transcription quality

**Solutions:**
1. Use larger Whisper model (`medium` or `large`)
2. Ensure audio is clear (reduce background noise)
3. Check audio format (some formats work better)
4. Increase audio volume if too quiet

### Problem: Summary is too generic

**Solutions:**
1. Use a better Ollama model (`mistral` or `llama2:13b`)
2. Try different summary types
3. Regenerate summary with different settings
4. Ensure transcript quality is good first

### Problem: App crashes during processing

**Solutions:**
1. Check available RAM (need 2-4GB free)
2. Close other applications
3. Use smaller models
4. Process shorter audio files
5. Check logs for specific errors

---

## 📊 Performance Benchmarks

**Test System**: 16GB RAM, Intel i7, No GPU

| Audio Length | Whisper Model | Transcription Time | Summary Time |
|--------------|---------------|-------------------|--------------|
| 10 min | tiny | ~1 min | ~30 sec |
| 10 min | base | ~2 min | ~30 sec |
| 30 min | base | ~5 min | ~45 sec |
| 60 min | small | ~15 min | ~60 sec |

**With GPU (NVIDIA RTX 3080):**
- 10x faster transcription
- Summary time unchanged (Ollama uses CPU)

---

## 💡 Tips for Best Results

### Audio Quality Tips:
1. Use high-quality audio files (192kbps+)
2. Avoid heavily compressed audio
3. Clear speech works best
4. Minimize background noise

### Transcription Tips:
1. Start with `base` model, upgrade if needed
2. First run takes longer (model download)
3. Subsequent runs are faster (cached model)
4. GPU significantly speeds up transcription

### Summarization Tips:
1. Try different summary types for different content
2. Longer transcripts benefit from "Key Points Only"
3. Interview podcasts work well with "Q&A Format"
4. Educational content: use "Detailed Summary"

### Workflow Tips:
1. Test with short clips first
2. Save transcripts for future reference
3. Regenerate summaries with different types
4. Keep Ollama running in background

---

## 🔧 Advanced Configuration

### Change Whisper Language

```python
# In transcribe_audio method, add:
result = self.model.transcribe(audio_path, language='en')
# Supported: en, es, fr, de, it, pt, nl, etc.
```

### Adjust Summary Length

Edit the prompts in `create_summary_prompt()` to request shorter/longer summaries:
```python
prompt = base_prompt + "Please provide a SHORT summary (100 words) that..."
```

### Use Different Ollama API

```python
OLLAMA_API_URL = "http://your-server:11434/api"
```

### Enable GPU for Whisper

Whisper automatically uses GPU if available. To force CPU:
```python
result = self.model.transcribe(audio_path, device="cpu")
```

---

## 📚 Additional Resources

- **Whisper Documentation**: https://github.com/openai/whisper
- **Ollama Models**: https://ollama.ai/library
- **yt-dlp Supported Sites**: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md
- **FFmpeg Guide**: https://ffmpeg.org/documentation.html

---

## 🎓 Understanding the Tech Stack

### Why These Tools?

1. **Whisper**: OpenAI's state-of-the-art speech recognition
   - Trained on 680,000 hours of audio
   - Supports 99 languages
   - Highly accurate

2. **Ollama**: Local LLM inference
   - Runs on your machine (privacy!)
   - No API costs
   - Fast and reliable

3. **yt-dlp**: Universal media downloader
   - Supports 1000+ websites
   - Regularly updated
   - Extract audio only

### The Pipeline Explained:

```
┌─────────────┐
│ Audio Input │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Whisper   │ ← Speech-to-Text AI
│ Transcriber │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Transcript │ ← Text document
│    (Text)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Ollama    │ ← Large Language Model
│ Summarizer  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Summary   │ ← Final output
└─────────────┘
```

---

## ❓ FAQ

**Q: Can Ollama process audio directly?**
A: No, Ollama only works with text. That's why we need Whisper to convert audio to text first.

**Q: Is my data private?**
A: Yes! Everything runs locally on your machine. No data is sent to external servers (except when downloading from URLs).

**Q: How accurate is Whisper?**
A: Very accurate! The `base` model has ~10% word error rate on clear English audio. Larger models are even better.

**Q: How long does processing take?**
A: Typically 2-5 minutes for a 30-minute podcast with the `base` model on a modern PC.

**Q: Can I process multiple podcasts at once?**
A: Not in the current version. Process them one at a time for best results.

**Q: What if my podcast is in another language?**
A: Whisper supports 99 languages! Modify the code to specify your language.

**Q: Can I use this commercially?**
A: Check Whisper and Ollama licenses. Generally yes for most use cases.

---

## 🚀 Future Enhancements

Potential features for future versions:
- Batch processing multiple podcasts
- Speaker diarization (identify speakers)
- Timestamp extraction
- Export to PDF/Word formats
- Multi-language support UI
- Audio quality enhancement
- Chapter detection
- Sentiment analysis

---

## 📄 License

This project is open source and available for personal and educational use.

**Component Licenses:**
- Whisper: MIT License
- Ollama: MIT License
- yt-dlp: Unlicense

---

## 🤝 Contributing

Found a bug? Have a feature request? Feel free to contribute!

---

**Happy Podcast Summarizing! 🎉**

For support, check the Troubleshooting section or consult the documentation of individual tools.
