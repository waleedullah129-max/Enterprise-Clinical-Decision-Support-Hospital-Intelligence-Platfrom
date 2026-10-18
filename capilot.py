import os
import re
from typing import List, Dict, Optional

import pandas as pd
from sqlalchemy import text

from database import engine


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

KNOWLEDGE_DIR = os.path.join(
    PROJECT_DIR,
    "knowledge_base"
)


# ============================================================
# OPTIONAL LLM
# ============================================================

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def load_knowledge_documents() -> List[Dict]:

    documents = []

    if not os.path.exists(KNOWLEDGE_DIR):

        os.makedirs(
            KNOWLEDGE_DIR,
            exist_ok=True
        )

        return documents

    for filename in os.listdir(KNOWLEDGE_DIR):

        if not filename.lower().endswith(
            (".txt", ".md")
        ):
            continue

        filepath = os.path.join(
            KNOWLEDGE_DIR,
            filename
        )

        try:

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

            if content.strip():

                documents.append({
                    "source": filename,
                    "content": content
                })

        except Exception as e:

            print(
                f"Document error ({filename}): {e}"
            )

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text_content: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> List[str]:

    text_content = re.sub(
        r"\s+",
        " ",
        text_content
    ).strip()

    if not text_content:

        return []

    chunks = []

    start = 0

    while start < len(text_content):

        end = start + chunk_size

        chunk = text_content[
            start:end
        ]

        chunks.append(
            chunk
        )

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(text_content):
            break

    return chunks


# ============================================================
# BUILD KNOWLEDGE INDEX
# ============================================================

def build_knowledge_index():

    documents = load_knowledge_documents()

    index = []

    for document in documents:

        chunks = chunk_text(
            document["content"]
        )

        for i, chunk in enumerate(chunks):

            index.append({
                "source": document["source"],
                "chunk_id": i,
                "text": chunk
            })

    return index


# ============================================================
# SIMPLE RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    top_k: int = 5
):

    index = build_knowledge_index()

    if not index:

        return []

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    )

    scored = []

    for item in index:

        text_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                item["text"].lower()
            )
        )

        score = len(
            query_words.intersection(
                text_words
            )
        )

        if score > 0:

            scored.append({
                **item,
                "score": score
            })

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored[:top_k]


# ============================================================
# PATIENT CONTEXT
# ============================================================

def get_patient_context(
    patient_id: Optional[str] = None
):

    if not patient_id:

        return "No patient selected."

    try:

        with engine.connect() as conn:

            patient = pd.read_sql(
                text("""
                    SELECT
                        "Id",
                        "BIRTHDATE"
                    FROM synthea_patients
                    WHERE "Id" = :patient_id
                """),
                conn,
                params={
                    "patient_id": str(patient_id)
                }
            )

            conditions = pd.read_sql(
                text("""
                    SELECT
                        "DESCRIPTION"
                    FROM synthea_conditions
                    WHERE "PATIENT" = :patient_id
                """),
                conn,
                params={
                    "patient_id": str(patient_id)
                }
            )

            medications = pd.read_sql(
                text("""
                    SELECT
                        "DESCRIPTION"
                    FROM synthea_medications
                    WHERE "PATIENT" = :patient_id
                """),
                conn,
                params={
                    "patient_id": str(patient_id)
                }
            )

            allergies = pd.read_sql(
                text("""
                    SELECT
                        "DESCRIPTION"
                    FROM synthea_allergies
                    WHERE "PATIENT" = :patient_id
                """),
                conn,
                params={
                    "patient_id": str(patient_id)
                }
            )

        if patient.empty:

            return (
                f"No patient found for ID: "
                f"{patient_id}"
            )

        birthdate = patient.iloc[0][
            "BIRTHDATE"
        ]

        try:

            birthdate = pd.to_datetime(
                birthdate
            )

            today = pd.Timestamp.today()

            age = (
                today.year
                - birthdate.year
            )

            if (
                today.month,
                today.day
            ) < (
                birthdate.month,
                birthdate.day
            ):

                age -= 1

        except Exception:

            age = "Unknown"

        condition_list = (
            conditions[
                "DESCRIPTION"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        medication_list = (
            medications[
                "DESCRIPTION"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        allergy_list = (
            allergies[
                "DESCRIPTION"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        return f"""
Patient ID:
{patient_id}

Age:
{age}

Conditions:
{", ".join(condition_list[:30])}

Medications:
{", ".join(medication_list[:30])}

Allergies:
{", ".join(allergy_list[:30])}
""".strip()

    except Exception as e:

        return (
            f"Unable to load patient context: {e}"
        )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    question: str,
    patient_context: str,
    retrieved_documents: List[Dict]
):

    if retrieved_documents:

        context = "\n\n".join(
            [
                (
                    f"Source: {item['source']}\n"
                    f"{item['text']}"
                )
                for item in retrieved_documents
            ]
        )

    else:

        context = (
            "No relevant knowledge-base "
            "documents were retrieved."
        )

    prompt = f"""
You are a Clinical AI Copilot for a
research and educational healthcare platform.

IMPORTANT SAFETY RULES:

1. Do not diagnose a patient.
2. Do not prescribe medication.
3. Do not provide definitive treatment decisions.
4. Clearly distinguish retrieved information
   from patient-specific observations.
5. If the provided evidence is insufficient,
   say so.
6. Encourage review by a qualified healthcare
   professional for real clinical decisions.

PATIENT CONTEXT:

{patient_context}

RETRIEVED KNOWLEDGE:

{context}

USER QUESTION:

{question}

TASK:

Provide a concise, evidence-grounded response.

Structure the response as:

1. Summary
2. Relevant patient information
3. Evidence from retrieved sources
4. Important considerations
5. Safety note

Do not invent information that is not present
in the patient context or retrieved sources.
"""

    return prompt


# ============================================================
# ANTHROPIC RESPONSE
# ============================================================

def generate_with_anthropic(
    prompt: str
):

    if Anthropic is None:

        return None

    api_key = os.getenv(
        "ANTHROPIC_API_KEY"
    )

    if not api_key:

        return None

    try:

        client = Anthropic(
            api_key=api_key
        )

        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1200,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if response.content:

            return response.content[0].text

        return None

    except Exception as e:

        print(
            "Anthropic error:",
            e
        )

        return None


# ============================================================
# FALLBACK RESPONSE
# ============================================================

def fallback_response(
    question: str,
    patient_context: str,
    retrieved_documents: List[Dict]
):

    if not retrieved_documents:

        return """
### Clinical AI Copilot

No relevant knowledge-base document was found
for this question.

Please add appropriate clinical guideline or
reference documents to the `knowledge_base`
folder and try again.

This is a research/educational prototype and
should not be used for real-world diagnosis or
treatment decisions.
""".strip()

    sources = "\n".join(
        [
            f"- {item['source']}"
            for item in retrieved_documents
        ]
    )

    evidence = "\n\n".join(
        [
            item["text"]
            for item in retrieved_documents
        ]
    )

    return f"""
### Clinical AI Copilot

**Question**

{question}

**Retrieved clinical information**

{evidence}

**Sources**

{sources}

**Safety note**

This response is based on retrieved research/demo
documents. It is not a diagnosis or treatment
recommendation. Clinical decisions should be made
by qualified healthcare professionals.
""".strip()


# ============================================================
# MAIN COPILOT FUNCTION
# ============================================================

def ask_copilot(
    question: str,
    patient_id: Optional[str] = None,
    top_k: int = 5
):

    if not question or not question.strip():

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "patient_context": None
        }

    question = question.strip()

    # --------------------------------------------------------
    # Patient context
    # --------------------------------------------------------

    patient_context = get_patient_context(
        patient_id
    )

    # --------------------------------------------------------
    # Retrieve knowledge
    # --------------------------------------------------------

    retrieved = retrieve_documents(
        question,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        patient_context,
        retrieved
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    answer = generate_with_anthropic(
        prompt
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if answer is None:

        answer = fallback_response(
            question,
            patient_context,
            retrieved
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = [
        {
            "source": item["source"],
            "score": item["score"]
        }
        for item in retrieved
    ]

    return {
        "answer": answer,
        "sources": sources,
        "patient_context": patient_context
    }
if __name__ == "__main__":
    result = ask_copilot(
        question="What clinical risks should I monitor?",
        patient_id="PUT-A-REAL-SYNTHEA-PATIENT-ID-HERE"
    )
    print(result["answer"])
    print(result["sources"])