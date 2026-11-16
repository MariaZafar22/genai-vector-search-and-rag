# GenAI Vector Search and RAG

Beginner Generative AI projects using embeddings, a vector database and an LLM.

## Projects

### 1. Semantic Search with a Vector Database
Stores 10 fictional university FAQs (with metadata) in **Chroma** and searches them using natural-language questions that use different words from the stored text.

- Folder: `semantic-search/`
- Run: `pip install -r requirements.txt` then `python semantic_search.py`

### 2. RAG Document Q&A System
A Retrieval-Augmented Generation pipeline: a fictional university handbook is split into chunks, embedded and stored in **Chroma**. Relevant chunks are retrieved for each question and sent to **Gemini** to generate an answer. Questions outside the knowledge base are declined instead of invented.

- Folder: `rag-document-qa/`
- Run: `pip install -r requirements.txt` then `python rag_qa.py`
- Needs a free Gemini API key: https://aistudio.google.com/apikey (entered at runtime, never stored in code)

## Workflow
question -> query embedding -> vector search (Chroma) -> relevant chunks -> LLM -> answer

## Tech
Python, ChromaDB, Google Gemini API
