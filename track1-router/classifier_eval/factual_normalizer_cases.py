from src.cascade import extract_factual_answer


CASES = {
    "sample-fact-1": """The user wants to know the capital of Vietnam.
I need to provide a concise answer in at most three sentences.
The question is a factual knowledge question.

Capital of Vietnam is Hanoi.

Constraint Checklist & Confidence Score:
1. Answer every part in at most three concise sentences? Yes.
2. For comparisons include relationship, key properties or mechanism, and uses? N/A (Factual question).
3. No markdown? Yes.
4. Factual knowledge question answered? Yes.

Confidence Score: 5/5

Plan: State the capital directly.""",
    "new-fact-1": """The Indian Ocean lies between Africa and Australia. This vast ocean separates the African continent to the west and the Australian continent to the east. It is a major global body of water influencing climate and trade routes.""",
    "new-fact-2": """Who developed the first successful polio vaccine?

1. Identify the core question: Who developed the first successful polio vaccine?
2. Recall relevant historical facts: The development of the polio vaccine is primarily attributed to Jonas Salk.
3. Formulate the answer concisely (max three sentences): State the developer and the significance.

Self-Correction/Refinement: Ensure the answer is direct and adheres to the constraint (at most three concise sentences).

Drafting the response.""",
    "new-fact-3": """Thinking Process:
1. Analyze the Request:
* Question: Where does the Nile River empty into the sea?
* Constraint 1: Answer every part in at most three concise sentences.
2. Determine the Answer (Factual Knowledge Retrieval):
* The Nile River is the longest river in Africa.
* It flows northward through Egypt and Sudan.
* It empties into the Mediterranean Sea.
3. Draft the Answer:
* Sentence 1: State the destination.
* Sentence 2: Provide context.
* (Check length""",
    "extra-fact-1": """The user is asking to identify the chemical element corresponding to the symbol Au.
I need to provide a concise answer in at most three sentences.
1. Identify the element.
2. Ensure the answer is concise.
The symbol Au represents the element gold.""",
    "extra-fact-2": """Antonio Vivaldi composed The Four Seasons. He was an Italian composer and virtuoso violinist. His work is a prime example of Baroque music.""",
    "extra-fact-3": """Machu Picchu is located in the Andes Mountains of Peru. It is situated in a remote and scenic area of the Sacred Valley. The site is known for its stunning Inca architecture and breathtaking panoramic views.""",
    "extra-fact-4": """The user wants to know when the Second World War ended.
I need to answer concisely in at most three sentences.
I must avoid markdown.
1. Identify the end date.
2. Formulate the answer concisely.
The Second World War officially ended in September 1945. Japan formally surrendered on August 15, 1945, leading to the cessation of hostilities. Germany surrendered in May 1945, marking the end of the conflict in Europe.""",
    "extra-fact-5": """Thinking Process:
1. Analyze the Request: The user is asking what gas plants absorb from the atmosphere during photosynthesis.
2. Constraint Check: Answer every part in at most three concise sentences. For comparisons include relationship, key properties or mechanism, and uses. No markdown.
3. Identify the Core Question: What gas do plants absorb?
4. Determine the Answer: Plants absorb carbon dioxide (CO2).
5. Draft the Answer:
* Start with the direct answer.
* Explain the role in photosynthesis.
7. Final Output Generation.""",
}

for case_id, raw in CASES.items():
    print(f"{case_id}: {extract_factual_answer(raw)!r}")
