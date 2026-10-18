import os
import sys
import pandas as pd
import streamlit as st
from sqlalchemy import text


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import engine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical AI Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}


/* Header */

.hero {
    padding: 10px 0 25px 0;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 17px;
    color: #64748b;
    margin-top: 4px;
}


/* Section */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 16px;
}


/* Patient header */

.patient-header {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-top: 10px;
    margin-bottom: 20px;
}


/* Risk */

.risk-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.risk-title {
    font-size: 17px;
    font-weight: 700;
    color: #475569;
}

.risk-score {
    font-size: 42px;
    font-weight: 850;
    margin-top: 5px;
}

.high {
    background: #fee2e2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 800;
    text-align: center;
}

.medium {
    background: #fef3c7;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 800;
    text-align: center;
}

.low {
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 800;
    text-align: center;
}


/* Info cards */

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    min-height: 140px;
}

.info-title {
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 8px;
}

.info-text {
    color: #64748b;
    font-size: 14px;
    line-height: 1.6;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}


/* Footer */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

@st.cache_data(ttl=60)
def get_stats():

    queries = {
        "patients": "SELECT COUNT(*) FROM mimic_patients",
        "admissions": "SELECT COUNT(*) FROM mimic_admissions",
        "labs": "SELECT COUNT(*) FROM mimic_labevents",
        "icu": "SELECT COUNT(*) FROM mimic_icustays"
    }

    result = {}

    with engine.connect() as conn:

        for key, query in queries.items():

            try:
                result[key] = int(
                    conn.execute(text(query)).scalar() or 0
                )

            except Exception:
                result[key] = 0

    return result


@st.cache_data(ttl=60)
def get_patients():

    query = """
        SELECT
            "Id",
            "BIRTHDATE",
            "FIRST",
            "LAST",
            "GENDER",
            "CITY",
            "STATE"
        FROM synthea_patients
        ORDER BY "LAST", "FIRST"
    """

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


@st.cache_data(ttl=60)
def get_patient(patient_id):

    query = """
        SELECT
            "Id",
            "BIRTHDATE",
            "FIRST",
            "LAST",
            "GENDER",
            "CITY",
            "STATE"
        FROM synthea_patients
        WHERE "Id" = :patient_id
    """

    with engine.connect() as conn:

        return pd.read_sql(
            text(query),
            conn,
            params={"patient_id": patient_id}
        )


@st.cache_data(ttl=60)
def get_conditions(patient_id):

    query = """
        SELECT
            "START",
            "STOP",
            "DESCRIPTION",
            "ENCOUNTER"
        FROM synthea_conditions
        WHERE "PATIENT" = :patient_id
        ORDER BY "START" DESC
    """

    with engine.connect() as conn:

        return pd.read_sql(
            text(query),
            conn,
            params={"patient_id": patient_id}
        )


@st.cache_data(ttl=60)
def get_medications(patient_id):

    query = """
        SELECT
            "START",
            "STOP",
            "DESCRIPTION",
            "REASONDESCRIPTION",
            "ENCOUNTER"
        FROM synthea_medications
        WHERE "PATIENT" = :patient_id
        ORDER BY "START" DESC
    """

    with engine.connect() as conn:

        return pd.read_sql(
            text(query),
            conn,
            params={"patient_id": patient_id}
        )


@st.cache_data(ttl=60)
def get_allergies(patient_id):

    query = """
        SELECT
            "START",
            "STOP",
            "DESCRIPTION",
            "REACTION1",
            "REACTION2",
            "SEVERITY1",
            "SEVERITY2"
        FROM synthea_allergies
        WHERE "PATIENT" = :patient_id
        ORDER BY "START" DESC
    """

    with engine.connect() as conn:

        return pd.read_sql(
            text(query),
            conn,
            params={"patient_id": patient_id}
        )


@st.cache_data(ttl=60)
def get_encounters(patient_id):

    query = """
        SELECT
            "START",
            "STOP",
            "ENCOUNTERCLASS",
            "DESCRIPTION",
            "REASONDESCRIPTION"
        FROM synthea_encounters
        WHERE "PATIENT" = :patient_id
        ORDER BY "START" DESC
    """

    with engine.connect() as conn:

        return pd.read_sql(
            text(query),
            conn,
            params={"patient_id": patient_id}
        )


@st.cache_data(ttl=60)
def get_risk(patient_id):

    query = """
        SELECT
            patient_id,
            risk_score,
            risk_level
        FROM patient_risk_scores
        WHERE patient_id = :patient_id
        LIMIT 1
    """

    try:

        with engine.connect() as conn:

            df = pd.read_sql(
                text(query),
                conn,
                params={"patient_id": str(patient_id)}
            )

        if df.empty:
            return None

        row = df.iloc[0]

        return {
            "patient_id": str(row["patient_id"]),
            "risk_score": float(row["risk_score"]),
            "risk_level": str(row["risk_level"])
        }

    except Exception as e:

        st.error(f"Risk database error: {e}")

        return None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🏥 Clinical AI Platform
</div>

<div class="hero-subtitle">
Hospital Intelligence & Clinical Decision Support System
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🏥 Clinical AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Dashboard",
        "👤 Patient Intelligence",
        "💊 Medication Safety",
        "🤖 Clinical Copilot"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### System Status")

st.sidebar.success("Database Connected")
st.sidebar.success("Risk Engine Ready")
st.sidebar.success("API Online")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Research / educational prototype.\n\n"
    "Not intended for diagnosis or treatment."
)


# ============================================================
# STATS
# ============================================================

stats = get_stats()


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "📊 Executive Dashboard":

    st.markdown(
        '<div class="section-title">Hospital Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MAIN KPI METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "TOTAL PATIENTS",
            f"{stats['patients']:,}"
        )

    with c2:

        st.metric(
            "TOTAL ADMISSIONS",
            f"{stats['admissions']:,}"
        )

    with c3:

        st.metric(
            "LAB EVENTS",
            f"{stats['labs']:,}"
        )

    with c4:

        st.metric(
            "ICU STAYS",
            f"{stats['icu']:,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CLINICAL DATA COVERAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Clinical Data Coverage</div>',
        unsafe_allow_html=True
    )

    patients = get_patients()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Synthea Patients",
            len(patients)
        )

    with c2:

        st.metric(
            "Conditions",
            "3,517"
        )

    with c3:

        st.metric(
            "Medications",
            "3,850"
        )

    with c4:

        st.metric(
            "Allergies",
            "105"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # AI PLATFORM MODULES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Platform Modules</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="info-card">

            <div class="info-title">
                🧠 Patient Risk Prediction
            </div>

            <div class="info-text">
                AI-assisted risk stratification using
                patient demographic and clinical features.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">

            <div class="info-title">
                💊 Medication Safety
            </div>

            <div class="info-text">
                Review medication history, allergies,
                and potential safety signals.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="info-card">

            <div class="info-title">
                🤖 Clinical Copilot
            </div>

            <div class="info-text">
                AI-assisted clinical information and
                patient-record summarization.
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PATIENT INTELLIGENCE
# ============================================================

elif page == "👤 Patient Intelligence":

    st.markdown(
        '<div class="section-title">Patient Intelligence</div>',
        unsafe_allow_html=True
    )

    patients = get_patients()

    if patients.empty:

        st.error("No Synthea patients found.")

    else:

        # ----------------------------------------------------
        # PATIENT SELECTOR
        # ----------------------------------------------------

        patients["display"] = (
            patients["FIRST"].fillna("").astype(str)
            + " "
            + patients["LAST"].fillna("").astype(str)
            + "  |  "
            + patients["Id"].astype(str)
        )

        selected = st.selectbox(
            "🔎 Select Patient",
            patients["display"].tolist()
        )

        patient_id = patients.loc[
            patients["display"] == selected,
            "Id"
        ].iloc[0]

        st.caption(
            f"Patient ID: `{patient_id}`"
        )

        patient = get_patient(patient_id)

        if not patient.empty:

            p = patient.iloc[0]

            # ------------------------------------------------
            # PATIENT PROFILE
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Patient Profile</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="patient-header">

                    <h2>👤 {p['FIRST']} {p['LAST']}</h2>

                    <p>
                        <b>Patient ID:</b> {p['Id']}
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Gender",
                    str(p["GENDER"])
                )

            with c2:

                st.metric(
                    "Birth Date",
                    str(p["BIRTHDATE"])
                )

            with c3:

                st.metric(
                    "City",
                    str(p["CITY"])
                )

            with c4:

                st.metric(
                    "State",
                    str(p["STATE"])
                )

            # ------------------------------------------------
            # RISK
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">🧠 AI Risk Assessment</div>',
                unsafe_allow_html=True
            )

            risk = get_risk(patient_id)

            if risk is not None:

                score = float(risk["risk_score"])
                level = str(risk["risk_level"])

                r1, r2 = st.columns([3, 1])

                with r1:

                    st.markdown(
                        f"""
                        <div class="risk-card">

                            <div class="risk-title">
                                Patient Risk Score
                            </div>

                            <div class="risk-score">
                                {score:.1f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            max(score / 100, 0.0),
                            1.0
                        )
                    )

                with r2:

                    st.markdown(
                        "<br>",
                        unsafe_allow_html=True
                    )

                    if level.lower() == "high":

                        st.markdown(
                            '<div class="high">🔴 HIGH RISK</div>',
                            unsafe_allow_html=True
                        )

                    elif level.lower() == "medium":

                        st.markdown(
                            '<div class="medium">🟡 MEDIUM RISK</div>',
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            '<div class="low">🟢 LOW RISK</div>',
                            unsafe_allow_html=True
                        )

            else:

                st.info(
                    "ℹ️ Risk score is not available for this patient yet."
                )

            # ------------------------------------------------
            # CLINICAL SUMMARY
            # ------------------------------------------------

            conditions = get_conditions(patient_id)
            medications = get_medications(patient_id)
            allergies = get_allergies(patient_id)
            encounters = get_encounters(patient_id)

            st.markdown(
                '<div class="section-title">Clinical Summary</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "🩺 Conditions",
                    len(conditions)
                )

            with c2:

                st.metric(
                    "💊 Medications",
                    len(medications)
                )

            with c3:

                st.metric(
                    "⚠️ Allergies",
                    len(allergies)
                )

            with c4:

                st.metric(
                    "🏥 Encounters",
                    len(encounters)
                )

            # ------------------------------------------------
            # CLINICAL DATA
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Clinical Information</div>',
                unsafe_allow_html=True
            )

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "🩺 Conditions",
                    "💊 Medications",
                    "⚠️ Allergies",
                    "🏥 Encounters"
                ]
            )

            with tab1:

                if not conditions.empty:

                    st.dataframe(
                        conditions,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No condition records found."
                    )

            with tab2:

                if not medications.empty:

                    st.dataframe(
                        medications,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No medication records found."
                    )

            with tab3:

                if not allergies.empty:

                    st.dataframe(
                        allergies,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.success(
                        "No allergies recorded."
                    )

            with tab4:

                if not encounters.empty:

                    st.dataframe(
                        encounters,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No encounter records found."
                    )


# ============================================================
# MEDICATION SAFETY
# ============================================================

elif page == "💊 Medication Safety":

    st.markdown(
        '<div class="section-title">💊 Medication Safety</div>',
        unsafe_allow_html=True
    )

    patients = get_patients()

    patients["display"] = (
        patients["FIRST"].fillna("").astype(str)
        + " "
        + patients["LAST"].fillna("").astype(str)
        + "  |  "
        + patients["Id"].astype(str)
    )

    selected = st.selectbox(
        "Select Patient",
        patients["display"].tolist(),
        key="medication_patient"
    )

    patient_id = patients.loc[
        patients["display"] == selected,
        "Id"
    ].iloc[0]

    medications = get_medications(patient_id)
    allergies = get_allergies(patient_id)

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Medications",
            len(medications)
        )

    with c2:

        st.metric(
            "Allergies",
            len(allergies)
        )

    st.markdown("---")

    if not allergies.empty:

        st.warning(
            "⚠️ Allergy information is available for review."
        )

    st.markdown(
        "### 💊 Medication History"
    )

    if not medications.empty:

        st.dataframe(
            medications,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No medication records found."
        )

    st.markdown(
        "### ⚠️ Allergy History"
    )

    if not allergies.empty:

        st.dataframe(
            allergies,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No allergies recorded."
        )


# ============================================================
# CLINICAL COPILOT
# ============================================================

elif page == "🤖 Clinical Copilot":

    st.markdown(
        '<div class="section-title">🤖 Clinical AI Copilot</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Research / educational prototype. "
        "AI output should not be used for diagnosis or treatment."
    )

    patients = get_patients()

    patients["display"] = (
        patients["FIRST"].fillna("").astype(str)
        + " "
        + patients["LAST"].fillna("").astype(str)
        + "  |  "
        + patients["Id"].astype(str)
    )

    selected = st.selectbox(
        "Select Patient",
        patients["display"].tolist(),
        key="copilot_patient"
    )

    patient_id = patients.loc[
        patients["display"] == selected,
        "Id"
    ].iloc[0]

    question = st.text_area(
        "Ask about this patient's recorded information",
        placeholder=(
            "Example: Summarize this patient's "
            "conditions, medications and allergies."
        )
    )

    if st.button("Generate Patient Summary"):

        conditions = get_conditions(patient_id)
        medications = get_medications(patient_id)
        allergies = get_allergies(patient_id)

        st.markdown(
            "### Patient Information"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Conditions",
                len(conditions)
            )

        with c2:

            st.metric(
                "Medications",
                len(medications)
            )

        with c3:

            st.metric(
                "Allergies",
                len(allergies)
            )

        if question:

            st.markdown(
                "### Question"
            )

            st.write(question)

            st.markdown(
                "### Available Patient Data"
            )

            if not conditions.empty:

                st.write(
                    "**Recorded conditions:**",
                    ", ".join(
                        conditions["DESCRIPTION"]
                        .dropna()
                        .astype(str)
                        .head(10)
                    )
                )

            if not medications.empty:

                st.write(
                    "**Recorded medications:**",
                    ", ".join(
                        medications["DESCRIPTION"]
                        .dropna()
                        .astype(str)
                        .head(10)
                    )
                )

            if not allergies.empty:

                st.write(
                    "**Recorded allergies:**",
                    ", ".join(
                        allergies["DESCRIPTION"]
                        .dropna()
                        .astype(str)
                        .head(10)
                    )
                )

            st.info(
                "Your RAG/LLM layer can be connected to this "
                "patient context for the final Copilot implementation."
            )

        else:

            st.warning(
                "Please enter a question first."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Clinical AI Platform • Hospital Intelligence System
        <br>
        Research & Educational Prototype •
        Not intended for diagnosis or treatment

    </div>
    """,
    unsafe_allow_html=True
)