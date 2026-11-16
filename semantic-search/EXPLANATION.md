# Semantic Search with a Vector Database

## Database selected
I used **Chroma**, a beginner-friendly open-source vector database that runs locally. It has a built-in embedding model (all-MiniLM-L6-v2), so text is automatically converted into vectors and stored along with metadata. A vector database is different from a normal database: instead of exact matches (e.g. price < 5000), it finds records that are *similar in meaning* to a query.

## Workflow
Information → embedding model → vector → Chroma → similarity search → relevant results

## Sample information stored
10 fictional university FAQs (Admission Deadlines, Tuition Fee, Merit Scholarship, Library Timings, Hostel Accommodation, Course Withdrawal, Career Services, Exam Results, Campus Wi-Fi, Attendance Policy).

Each record has:
- **Vector** (embedding of the FAQ text)
- **Document** (the FAQ text)
- **Metadata**: title, category, source

## Search queries (written with different words than the stored text)
1. "How much money do I need to pay every term?" → expected: Tuition Fee
2. "Where can I sleep if I live far from the university?" → expected: Hostel Accommodation
3. "What happens if I miss too many classes?" → expected: Attendance Policy
4. "Is there help for getting a job?" → expected: Career Services

## Returned results
(Paste your actual program output here, or attach the screenshots.)

## Observation
Even though the questions use different words from the stored FAQs, the vector search returned records with related meaning. However, similarity does not guarantee correctness, so real applications need extra techniques (like metadata filtering, re-ranking and good chunking) to improve retrieval quality.

## Source code
`semantic_search.py` (attached / repository link: ______)
