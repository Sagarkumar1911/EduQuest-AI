# 🧠 Saarthi AI 
> **"See it. Hear it. Master it."**
> A Multimodal AI Tutor bridging the gap between static textbooks and interactive mastery.

## 🌟 Overview
Saarthi AI is an inclusive, accessible education platform designed for students who struggle with traditional text-heavy learning. It combines **RAG (Retrieval Augmented Generation)** with **Computer Vision** and **Voice AI** to create a "Digital Charioteer" (Saarthi) that guides students through complex topics.

## 🚀 Key Features
* **📚 Textbook RAG:** Upload any PDF/Textbook and chat with it accurately (powered by Qdrant).
* **👁️ Visual Intelligence:** Upload diagrams/images, and the AI (Llama 4 Vision) explains them.
* **🗣️ Voice Mode:** Full Text-to-Speech & Speech-to-Text for blind/dyslexic students.
* **🎥 Smart Multimedia:** Automatically finds the best YouTube video for the topic.
* **📊 Student Dashboard:** Tracks learning progress and identifies "Weak Areas" automatically.

## 🛠️ Tech Stack
* **Backend:** FastAPI (Python)
* **AI Engine:** Groq (Llama-3.3-70b for Text, Llama-4-Maverick for Vision)
* **Vector Database:** Qdrant (Local)
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript (Vanilla)

---

## ⚙️ Setup Instructions (Run this locally)

### 1. Prerequisites
* Python 3.9+ installed.
* A [Groq Cloud API Key](https://console.groq.com).
* A [Google/YouTube Data API Key](https://console.cloud.google.com).

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Saarthi-AI.git](https://github.com/YOUR_USERNAME/Saarthi-AI.git)
cd Saarthi-AI

# Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
