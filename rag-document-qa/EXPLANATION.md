# RAG Document Q&A System

## Knowledge source
A fictional "Horizon University Handbook" with 8 sections: Admissions, Fees, Scholarships, Attendance, Hostel, Library, Exams & Results, Career Services.

## Tools used
- **Chroma** – vector database (built-in embedding model all-MiniLM-L6-v2)
- **Gemini** – LLM that generates the final answer
- **Python** – command-line script (`rag_qa.py`)

## RAG workflow
1. **Chunking:** the handbook is split into 8 chunks, one per section, so each chunk holds one meaningful topic.
2. **Embedding + storage:** each chunk is converted into a vector and stored in Chroma with metadata (section, source).
3. **Retrieval:** the user's question is converted into a query embedding, and Chroma returns the 2 most similar chunks.
4. **Augmentation:** the retrieved chunks are inserted into the prompt as context.
5. **Generation:** the LLM is told to answer ONLY from the context and not invent information.

Flow: question → query embedding → vector search → relevant chunks → LLM → answer

## Test questions and answers
(Paste your actual program output here, or attach the screenshots.)

1. How much do I have to pay each semester, and can I split it?
2. What marks do I need to get 75 percent of my tuition waived?
3. Where can I live on campus and what does it cost?
4. What if I was sick and missed a lot of lectures?

## Question outside the knowledge base
**Question:** "Who is the vice chancellor of Horizon University?"

This information is not in the handbook. The prompt instructs the LLM to reply "Sorry, I don't have that information in the knowledge base" if the context does not contain the answer. So the system declined instead of inventing an answer (avoiding hallucination).

## Source code
`rag_qa.py` (attached / repository link: ______)
