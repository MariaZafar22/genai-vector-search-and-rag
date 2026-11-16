"""
Simple RAG Document Q&A System
Chroma (vector database) + Gemini (LLM)

Install first:
    pip install chromadb google-genai
(Google Colab: run  !pip install chromadb google-genai  in a cell first)

Get a FREE Gemini API key: https://aistudio.google.com/apikey

Flow: question -> query embedding -> Chroma search -> top chunks -> LLM -> answer
"""

import os
from getpass import getpass

import chromadb
from google import genai

# ---------------------------------------------------------------
# 0. Setup: API key + model
# ---------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY") or getpass("Enter your Gemini API key: ")
MODEL_NAME = "gemini-2.5-flash"  # if this name gives an error, use another Gemini Flash model from AI Studio

llm = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------
# 1. Knowledge base: fictional "Horizon University" handbook
# ---------------------------------------------------------------
DOCUMENT = """
ADMISSIONS
Horizon University accepts applications from 1 June to 31 July every year. Applicants must have at least 60 percent marks in their intermediate or equivalent exam. Required documents are transcripts, a copy of the national ID card, and two passport-size photographs. Applications are submitted online through the admission portal.

FEES
The undergraduate tuition fee is 120,000 PKR per semester. Students can pay in two equal installments: the first at admission and the second before the midterm exams. A late fee of 2,000 PKR per week is charged after the due date.

SCHOLARSHIPS
Students who score a GPA of 3.5 or above receive a merit scholarship covering 50 percent of tuition. Students with a GPA of 3.8 or above receive a 75 percent scholarship. A need-based scholarship is also available; students must apply with an income certificate before 15 August.

ATTENDANCE
A minimum of 75 percent attendance is required to sit in the final exams. Students below this limit may be barred from the exam. Medical leave with a valid doctor's certificate is counted as present.

HOSTEL
On-campus hostels are available for male and female students separately. Hostel rent is 15,000 PKR per month including meals. Rooms are allotted on a first-come, first-served basis, and students must apply within one week of admission.

LIBRARY
The library is open from 8 AM to 10 PM on weekdays and from 10 AM to 4 PM on weekends. Students can borrow up to 4 books for 14 days. A fine of 50 PKR per day is charged for late returns.

EXAMS AND RESULTS
Final exam results are published on the student portal within 14 days after the last exam. Students can request rechecking within one week of the result. The passing mark in every course is 50 percent.

CAREER SERVICES
The Career Services office helps students find internships, improve resumes and practice for job interviews. A job fair is organized every March, where more than 40 companies visit the campus.
"""


# ---------------------------------------------------------------
# 2. Chunking: split by section (each section = one meaningful chunk)
# ---------------------------------------------------------------
def chunk_document(text):
    chunks = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n", 1)
        if len(lines) == 2:
            title, body = lines[0].strip(), lines[1].strip()
            chunks.append({"title": title, "text": f"{title}: {body}"})
    return chunks


chunks = chunk_document(DOCUMENT)
print(f"Document split into {len(chunks)} chunks.")

# ---------------------------------------------------------------
# 3. Embed + store chunks in Chroma (embedding is automatic)
# ---------------------------------------------------------------
chroma = chromadb.Client()
collection = chroma.get_or_create_collection(name="horizon_handbook")
collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],
    documents=[c["text"] for c in chunks],
    metadatas=[{"section": c["title"], "source": "Horizon University Handbook"} for c in chunks],
)
print(f"Stored {collection.count()} chunks in the vector database.\n")


# ---------------------------------------------------------------
# 4. RAG function: retrieve -> augment prompt -> generate
# ---------------------------------------------------------------
def ask(question, top_k=2):
    # Retrieval
    result = collection.query(query_texts=[question], n_results=top_k)
    retrieved = result["documents"][0]
    sections = [m["section"] for m in result["metadatas"][0]]

    # Augmentation
    context = "\n\n".join(retrieved)
    prompt = f"""You are a helpful assistant for Horizon University.
Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "Sorry, I don't have that information in the knowledge base."
Do not make up any information.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Generation
    response = llm.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text.strip(), sections


# ---------------------------------------------------------------
# 5. Test questions (wording differs from the document)
# ---------------------------------------------------------------
questions = [
    "How much do I have to pay each semester, and can I split it?",              # Fees
    "What marks do I need to get 75 percent of my tuition waived?",              # Scholarships
    "Where can I live on campus and what does it cost?",                         # Hostel
    "What if I was sick and missed a lot of lectures?",                          # Attendance
    "Who is the vice chancellor of Horizon University?",                         # NOT in knowledge base
]

for q in questions:
    answer, used_sections = ask(q)
    print("=" * 70)
    print(f"QUESTION : {q}")
    print(f"RETRIEVED: {', '.join(used_sections)}")
    print(f"ANSWER   : {answer}")
    print()
