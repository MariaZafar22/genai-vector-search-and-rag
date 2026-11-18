# 🔎 GenAI Vector Search and RAG

**Beginner Generative AI projects using embeddings, a vector database and an LLM.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-FF6F61)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)

## 📂 Projects

### 1️⃣ Semantic Search with a Vector Database
Stores **10 fictional university FAQs** (with metadata) in **Chroma** and searches them with natural-language questions that use **different words** from the stored text.

- 📁 Folder: `semantic-search/`
- ▶️ Run: `pip install -r requirements.txt` then `python semantic_search.py`

### 2️⃣ RAG Document Q&A System
A **Retrieval-Augmented Generation** pipeline: a fictional university handbook is split into chunks, embedded and stored in **Chroma**. Relevant chunks are retrieved for each question and sent to **Gemini** to generate an answer. Questions outside the knowledge base are **declined instead of invented**.

- 📁 Folder: `rag-document-qa/`
- ▶️ Run: `pip install -r requirements.txt` then `python rag_qa.py`
- 🔑 Needs a free [Gemini API key](https://aistudio.google.com/apikey), entered at runtime and never stored in code

---

## ⚙️ Workflow

```
question -> query embedding -> vector search (Chroma) -> relevant chunks -> LLM -> answer
```

---

## 🧱 Tech Stack

**Python** · **ChromaDB** · **Google Gemini API**

---

## 👩‍💻 Author

**Maria Zafar** · [@MariaZafar22](https://github.com/MariaZafar22)
