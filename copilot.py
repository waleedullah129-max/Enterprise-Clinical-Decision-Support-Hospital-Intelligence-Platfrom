"""
copilot.py — AI Clinical Copilot
===================================
Doctor patient history summarize karwa sakta hai, ya sawal pooch sakta hai
jaise "Are there medication conflicts?" — jawab hamesha us patient ke REAL
data par based hota hai (RAG), guess nahi.

ARCHITECTURE (dashboard.py jaisa hi pattern):
    Copilot ──HTTP──> FastAPI backend (main.py) ──> PostgreSQL + ML models
    Copilot seedha database ya risk model import NAHI karta.

RAG PIPELINE:
    1. Retrieve : FastAPI se patient ka data nikal kar chhote "chunks" banao
                  (demographics, conditions, medications, risk, recommendations)
    2. Embed    : OpenAI embeddings se har chunk ko vector banao
    3. Search   : Doctor ke sawal ko bhi embed karo, cosine similarity se
                  sabse relevant chunks dhoondo
    4. Generate : Sirf un relevant chunks ko context bana kar LLM se jawab
                  banwao — is tarah LLM sirf real data par based rehta hai,
                  hallucination ka chance kam hota hai

Chalane se pehle:
1. FastAPI backend chal raha ho: uvicorn main:app --reload
2. OPENAI_API_KEY environment variable set ho
3. pip install langchain langchain-openai streamlit requests numpy python-dotenv

Chalane ka tarika:
    streamlit run copilot.py
"""

import os
from typing import Dict, List

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================
# CONFIG
# ============================================
load_dotenv()  # .env file se OPENAI_API_KEY uthata hai (agar hai to)

API_BASE_URL = "http://127.0.0.1:8000"
CHAT_MODEL = "gpt-4o-mini"       # sasta aur fast — chahen to "gpt-4o" kar sakte ho
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_CHUNKS = 6                 # doctor ke sawal ke liye kitne relevant chunks lein

SYSTEM_PROMPT = """Tum ek Clinical AI Copilot ho jo doctors ki madad karta hai.

Sakht rules:
- SIRF diye gaye "Patient Context" mein jo likha hai, usi ke basis par jawab do.
- Agar context mein jawab nahi hai, saaf kaho "Ye information available data mein nahi hai."
- Kabhi bhi guess ya assumption mat karo, khaas taur par dosage ya diagnosis ke baare mein.
- Har jawab ke end mein reminder do ke ye clinical decision support hai,
  final decision hamesha licensed clinician ka hona chahiye.
- Jawab concise aur clinically useful rakho — doctor ka time zaroori hai.
"""


# ============================================
# 1. FASTAPI SE PATIENT DATA NIKALO (RETRIEVE — step 1 of RAG)
# ============================================
def fetch_patient_documents(patient_id: str) -> List[Dict[str, str]]:
    """
    FastAPI backend se ek patient ka poora data khींchta hai aur usay
    chhote, labeled text chunks mein todta hai — RAG ke retrieval step
    ke liye "documents" yahi hain.
    """
    docs: List[Dict[str, str]] = []

    # ---- Demographics + counts (/patients/{id}) ----
    profile = _safe_get(f"/patients/{patient_id}")
    if profile:
        docs.append(
            {
                "source": "demographics",
                "text": (
                    f"Patient {patient_id} demographics: gender {profile.get('GENDER')}, "
                    f"race {profile.get('RACE')}, ethnicity {profile.get('ETHNICITY')}, "
                    f"birthdate {profile.get('BIRTHDATE')}. "
                    f"Age approx {profile.get('age')}. "
                    f"Has {profile.get('conditions_count')} recorded conditions, "
                    f"{profile.get('medications_count')} medications, "
                    f"{profile.get('allergies_count')} allergies, "
                    f"{profile.get('encounters_count')} total encounters."
                ),
            }
        )

    # ---- Risk + Recommendations (/recommendations/{id}) ----
    rec_result = _safe_get(f"/recommendations/{patient_id}")
    if rec_result:
        docs.append(
            {
                "source": "risk_assessment",
                "text": (
                    f"ML risk score: {rec_result.get('risk_score')} "
                    f"({rec_result.get('risk_label')}). This is a deterioration/high-acuity "
                    f"risk prediction based on age, conditions, medications, and encounter history."
                ),
            }
        )

        for i, rec in enumerate(rec_result.get("recommendations", [])):
            docs.append(
                {
                    "source": f"recommendation_{i}",
                    "text": (
                        f"[{rec['severity']}] {rec['type']}: {rec['message']} "
                        f"Evidence: {rec['evidence']}"
                    ),
                }
            )

    # ---- Conditions (/synthea/patients/{id}/conditions) ----
    conditions = _safe_get(f"/synthea/patients/{patient_id}/conditions") or []
    if conditions:
        condition_list = "; ".join(
            f"{c.get('DESCRIPTION')} (started {c.get('START')})" for c in conditions
        )
        docs.append(
            {"source": "conditions", "text": f"Diagnosed conditions: {condition_list}."}
        )

    # ---- Medication safety, patient-specific ----
    for endpoint, label in [
        ("duplicate-prescriptions", "duplicate prescriptions"),
        ("drug-interactions", "drug-drug interactions"),
        ("allergy-conflicts", "allergy conflicts"),
    ]:
        findings = _safe_get(
            f"/medication-safety/{endpoint}", params={"patient_id": patient_id}
        ) or []
        if findings:
            docs.append(
                {
                    "source": endpoint,
                    "text": f"Medication safety — {label}: {findings}",
                }
            )

    return docs


def _safe_get(path: str, params: dict | None = None):
    """FastAPI ko call karta hai, 404 ya error ko gracefully handle karta hai."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# ============================================
# 2 & 3. EMBED + RETRIEVE (RAG ka core)
# ============================================
def build_embeddings(docs: List[Dict[str, str]], embedder: OpenAIEmbeddings) -> np.ndarray:
    texts = [d["text"] for d in docs]
    vectors = embedder.embed_documents(texts)
    return np.array(vectors)


def retrieve_relevant_docs(
    question: str,
    docs: List[Dict[str, str]],
    doc_vectors: np.ndarray,
    embedder: OpenAIEmbeddings,
    k: int = TOP_K_CHUNKS,
) -> List[Dict[str, str]]:
    """Doctor ke sawal ko embed kar ke, cosine similarity se sabse relevant chunks dhoondta hai."""
    if len(docs) == 0:
        return []

    question_vector = np.array(embedder.embed_query(question))

    # Cosine similarity
    norms = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(question_vector)
    norms[norms == 0] = 1e-10
    similarities = doc_vectors @ question_vector / norms

    top_indices = np.argsort(similarities)[::-1][:k]
    return [docs[i] for i in top_indices]


# ============================================
# 4. GENERATE — LLM se jawab banwao (sirf retrieved context se)
# ============================================
def answer_question(
    patient_id: str,
    question: str,
    docs: List[Dict[str, str]],
    doc_vectors: np.ndarray,
    embedder: OpenAIEmbeddings,
    chat_model: ChatOpenAI,
) -> str:
    relevant_docs = retrieve_relevant_docs(question, docs, doc_vectors, embedder)
    context = "\n".join(f"- [{d['source']}] {d['text']}" for d in relevant_docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Patient Context (patient_id={patient_id}):\n{context}\n\n"
                f"Doctor's question: {question}"
            )
        ),
    ]
    response = chat_model.invoke(messages)
    return response.content


def summarize_history(
    patient_id: str, docs: List[Dict[str, str]], chat_model: ChatOpenAI
) -> str:
    """Poore patient data (sab chunks) ka concise clinical summary banata hai."""
    context = "\n".join(f"- [{d['source']}] {d['text']}" for d in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Patient Context (patient_id={patient_id}):\n{context}\n\n"
                "Task: Is patient ki medical history ka ek concise, doctor-friendly "
                "summary likho — demographics, key conditions, risk level, aur koi "
                "bhi medication safety alerts shamil karo. Bullet points mein likho."
            )
        ),
    ]
    response = chat_model.invoke(messages)
    return response.content


# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="AI Clinical Copilot", page_icon="🩺", layout="wide")
st.title("🩺 AI Clinical Copilot")
st.caption("RAG-based — jawab sirf FastAPI backend se aaye real patient data par based hain")

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "⚠️ OPENAI_API_KEY set nahi hai. Terminal mein chalao: "
        "`$env:OPENAI_API_KEY=\"sk-...\"` (PowerShell) ya ek `.env` file banao "
        "jisme `OPENAI_API_KEY=sk-...` likha ho, phir dashboard restart karo."
    )
    st.stop()

embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

patient_id = st.sidebar.text_input("Patient ID")
summarize_clicked = st.sidebar.button("📋 Summarize History")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_patient_id" not in st.session_state:
    st.session_state.loaded_patient_id = None

# Patient change hote hi chat history aur cached docs reset karo
if patient_id and patient_id != st.session_state.loaded_patient_id:
    st.session_state.chat_history = []
    st.session_state.loaded_patient_id = patient_id
    with st.spinner("Patient data fetch aur embed ho raha hai..."):
        docs = fetch_patient_documents(patient_id)
        if docs:
            st.session_state.docs = docs
            st.session_state.doc_vectors = build_embeddings(docs, embedder)
        else:
            st.session_state.docs = []
            st.session_state.doc_vectors = np.array([])

if patient_id and not st.session_state.get("docs"):
    st.warning("Ye Patient ID nahi mila ya iska koi data nahi hai.")
    st.stop()

if not patient_id:
    st.info("Sidebar mein ek Patient ID daalo shuru karne ke liye.")
    st.stop()

# ---- Summarize button ----
if summarize_clicked:
    with st.spinner("Summary ban rahi hai..."):
        summary = summarize_history(patient_id, st.session_state.docs, chat_model)
    st.markdown("### 📋 Patient History Summary")
    st.markdown(summary)
    st.divider()

# ---- Chat interface ----
st.markdown("### 💬 Doctor se sawal poocho")
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

question = st.chat_input("Jaise: 'Are there medication conflicts?' ya 'What clinical risks should I monitor?'")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Sochte hue..."):
            answer = answer_question(
                patient_id,
                question,
                st.session_state.docs,
                st.session_state.doc_vectors,
                embedder,
                chat_model,
            )
        st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))