"""
Semantic Search with a Vector Database (Chroma)

Flow: FAQ text -> embedding model -> vectors -> Chroma -> similarity search -> results

Install first:
    pip install chromadb
(Google Colab: run  !pip install chromadb  in a cell first)
"""

import chromadb

# 1. Create a Chroma client (in-memory, simple local experiment)
client = chromadb.Client()

# 2. Create a collection (Chroma's built-in embedding model
#    all-MiniLM-L6-v2 converts text into vectors automatically)
collection = client.get_or_create_collection(name="university_faqs")

# 3. Sample data: 10 fictional university FAQs
faqs = [
    {
        "id": "faq1",
        "text": "Admission applications open on 1 June and close on 31 July each year. Applicants must submit transcripts and a completed form online.",
        "title": "Admission Deadlines",
        "category": "Admissions",
    },
    {
        "id": "faq2",
        "text": "The undergraduate tuition fee is 120,000 PKR per semester. Fees can be paid in two installments.",
        "title": "Tuition Fee",
        "category": "Fees",
    },
    {
        "id": "faq3",
        "text": "Students with a GPA of 3.5 or above qualify for a merit scholarship that covers 50 percent of tuition.",
        "title": "Merit Scholarship",
        "category": "Scholarships",
    },
    {
        "id": "faq4",
        "text": "The library is open from 8 AM to 10 PM on weekdays and from 10 AM to 4 PM on weekends. Students must bring their ID card.",
        "title": "Library Timings",
        "category": "Campus Facilities",
    },
    {
        "id": "faq5",
        "text": "On-campus hostels are available for both male and female students. Rooms are allotted on a first-come, first-served basis.",
        "title": "Hostel Accommodation",
        "category": "Campus Facilities",
    },
    {
        "id": "faq6",
        "text": "Students can drop a course within the first two weeks of the semester without any penalty through the student portal.",
        "title": "Course Withdrawal",
        "category": "Academics",
    },
    {
        "id": "faq7",
        "text": "The Career Services office helps students find internships, review resumes and prepare for job interviews.",
        "title": "Career Services",
        "category": "Student Support",
    },
    {
        "id": "faq8",
        "text": "Final exam results are published on the student portal within 14 days after the last exam. Students can request rechecking within one week.",
        "title": "Exam Results",
        "category": "Examinations",
    },
    {
        "id": "faq9",
        "text": "The university provides free Wi-Fi across the campus. Students log in using their registration number and password.",
        "title": "Campus Wi-Fi",
        "category": "Campus Facilities",
    },
    {
        "id": "faq10",
        "text": "A minimum of 75 percent attendance is required to sit in the final exams. Students below this limit may be barred.",
        "title": "Attendance Policy",
        "category": "Academics",
    },
]

# 4. Store the records: text + embedding (created automatically) + metadata
collection.add(
    ids=[f["id"] for f in faqs],
    documents=[f["text"] for f in faqs],
    metadatas=[
        {"title": f["title"], "category": f["category"], "source": "University Handbook 2026"}
        for f in faqs
    ],
)
print(f"Stored {collection.count()} records in the vector database.\n")

# 5. Questions that do NOT use the same words as the stored text
queries = [
    "How much money do I need to pay every term?",
    "Where can I sleep if I live far from the university?",
    "What happens if I miss too many classes?",
    "Is there help for getting a job?",
]

# 6. Similarity search
for q in queries:
    results = collection.query(query_texts=[q], n_results=3)
    print("=" * 70)
    print(f"QUESTION: {q}")
    print("=" * 70)
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        text = results["documents"][0][i]
        print(f"  Rank {i + 1} | {meta['title']} ({meta['category']}) | distance: {distance:.4f}")
        print(f"     {text}")
    print()

# 7. Bonus: metadata filter (search only within one category)
print("=" * 70)
print("FILTERED SEARCH (category = Campus Facilities)")
print("=" * 70)
filtered = collection.query(
    query_texts=["Where can I study quietly?"],
    n_results=2,
    where={"category": "Campus Facilities"},
)
for i in range(len(filtered["ids"][0])):
    meta = filtered["metadatas"][0][i]
    print(f"  Rank {i + 1} | {meta['title']} | {filtered['documents'][0][i]}")
