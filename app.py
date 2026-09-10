"""
AI HR Recruitment Assistant — v3 (Advanced, Polished UI)
=========================================================
Agentic AI system that:
1. RAG            -> Retrieves relevant resume content matched against a job description
                     (skill-section retrieval + full-document TF-IDF retrieval)
2. Tool Calling    -> Calls a hybrid scoring tool + an interview-question-generator tool
3. Agent Behavior  -> Orchestrates upload -> parse -> retrieve -> score -> rank -> generate

New in v3:
- Complete visual redesign: gradient hero header, glassmorphism cards, animated
  progress bars, polished badges, custom fonts, and a refreshed color system
- Redesigned sidebar with grouped sections and icons
- Nicer empty/loading states, KPI summary row, and a cleaner results layout
- Same scoring logic and functionality as v2 — only the UI/UX layer changed

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Optional (for LLM-generated interview questions):
    export ANTHROPIC_API_KEY="your-key-here"      (Mac/Linux)
    setx ANTHROPIC_API_KEY "your-key-here"         (Windows)

Without a key, everything still works — scoring is 100% local, and
interview questions fall back to a rule-based generator.
"""

import os
import re
import io
import glob
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Optional file-format readers
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False

try:
    import docx as docx_lib
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

# ---------------------------------------------------------------------------
# Optional: Anthropic client (used if an API key is available)
# ---------------------------------------------------------------------------
USE_LLM = False
try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic()
        USE_LLM = True
except Exception:
    USE_LLM = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ===========================================================================
# FILE PARSING  (handles uploaded .txt / .pdf / .docx)
# ===========================================================================
def read_uploaded_file(uploaded_file) -> str:
    """Extract raw text from an uploaded txt/pdf/docx file."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        if not HAVE_PDF:
            st.error("pypdf not installed — add `pypdf` to requirements.txt to read PDFs.")
            return ""
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        if not HAVE_DOCX:
            st.error("python-docx not installed — add `python-docx` to requirements.txt.")
            return ""
        doc = docx_lib.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    st.warning(f"Unsupported file type: {uploaded_file.name}")
    return ""


def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_sample_resumes() -> dict:
    resumes = {}
    for filepath in sorted(glob.glob(os.path.join(DATA_DIR, "resume_*.txt"))):
        name = os.path.basename(filepath).replace(".txt", "").replace("_", " ").title()
        resumes[name] = load_text(filepath)
    return resumes


def load_sample_jd() -> str:
    return load_text(os.path.join(DATA_DIR, "job_description.txt"))


# ===========================================================================
# TOOL 1: extract_skills_section  (RAG-style targeted retrieval)
# ===========================================================================
def extract_skills_section(text: str) -> str:
    match = re.search(r"Skills:(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def parse_skill_list(skills_text: str) -> list:
    skills = re.split(r"[\n,\-•()]", skills_text)
    return [s.strip().lower() for s in skills if s.strip() and len(s.strip()) > 1]


# ===========================================================================
# TOOL 2: hybrid score_resume
#   a) skill overlap (deterministic, explainable)
#   b) TF-IDF cosine similarity across the whole document (semantic-ish RAG)
# ===========================================================================
def skill_match(jd_text: str, resume_text: str) -> dict:
    jd_skills = set(parse_skill_list(extract_skills_section(jd_text)))
    resume_skills = set(parse_skill_list(extract_skills_section(resume_text)))

    matched = set()
    for jd_skill in jd_skills:
        for r_skill in resume_skills:
            if jd_skill in r_skill or r_skill in jd_skill:
                matched.add(jd_skill)
                break

    missing = jd_skills - matched
    pct = round((len(matched) / len(jd_skills)) * 100, 1) if jd_skills else 0.0
    return {"match_percent": pct, "matched_skills": sorted(matched), "missing_skills": sorted(missing)}


def content_similarity(jd_text: str, resume_texts: list) -> list:
    """TF-IDF + cosine similarity between the JD and each full resume."""
    corpus = [jd_text] + resume_texts
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        return [round(float(s) * 100, 1) for s in sims]
    except ValueError:
        return [0.0 for _ in resume_texts]


def score_resume(jd_text: str, resume_text: str, sim_score: float, skill_weight: float) -> dict:
    sk = skill_match(jd_text, resume_text)
    content_weight = round(1 - skill_weight, 2)
    final = round(skill_weight * sk["match_percent"] + content_weight * sim_score, 1)
    sk["content_similarity"] = sim_score
    sk["final_score"] = final
    return sk


# ===========================================================================
# TOOL 3: generate_interview_questions  (LLM tool call, with offline fallback)
# ===========================================================================
def generate_interview_questions(candidate_name: str, matched_skills: list, missing_skills: list) -> list:
    if USE_LLM:
        prompt = f"""You are an HR interview panel assistant.
Candidate: {candidate_name}
Matched skills with the job: {', '.join(matched_skills) if matched_skills else 'none'}
Missing/weak skills: {', '.join(missing_skills) if missing_skills else 'none'}

Generate exactly 4 concise interview questions:
- 2 technical questions probing their matched skills
- 2 questions assessing their ability to learn the missing skills

Return ONLY a JSON array of 4 strings, nothing else."""
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            text = re.sub(r"^```json|```$", "", text).strip()
            return json.loads(text)
        except Exception as e:
            st.warning(f"LLM call failed, using offline fallback questions. ({e})")

    questions = []
    for skill in matched_skills[:2]:
        questions.append(f"Can you walk us through a project where you used {skill}?")
    for skill in missing_skills[:2]:
        questions.append(f"You haven't listed {skill} — how would you approach learning it quickly?")
    while len(questions) < 4:
        questions.append("Tell us about a challenging data problem you solved recently.")
    return questions[:4]


# ===========================================================================
# AGENT ORCHESTRATION
# ===========================================================================
def run_recruitment_agent(jd_text: str, resumes: dict, skill_weight: float) -> list:
    names = list(resumes.keys())
    texts = list(resumes.values())
    sims = content_similarity(jd_text, texts)

    results = []
    for name, resume_text, sim in zip(names, texts, sims):
        score_data = score_resume(jd_text, resume_text, sim, skill_weight)
        questions = generate_interview_questions(name, score_data["matched_skills"], score_data["missing_skills"])
        results.append({"candidate": name, **score_data, "interview_questions": questions})

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# ===========================================================================
# STREAMLIT UI
# ===========================================================================
st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL STYLE — gradient hero, glassmorphism cards, polished typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', 'Segoe UI', sans-serif !important; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a1033 0%, #0b0f1e 38%, #060810 100%);
    }

    /* ---------------- Hero header ---------------- */
    .hero-wrap {
        background: linear-gradient(120deg, #6a3df5 0%, #a24bf0 35%, #ff5ea8 70%, #ff9d6c 100%);
        border-radius: 22px;
        padding: 34px 38px;
        margin-bottom: 26px;
        box-shadow: 0 20px 45px -18px rgba(162, 75, 240, 0.55);
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(circle at 85% 20%, rgba(255,255,255,0.25), transparent 55%);
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 800; color: white; margin: 0;
        letter-spacing: -0.02em; position: relative; z-index: 1;
    }
    .hero-sub {
        color: rgba(255,255,255,0.92); font-size: 0.98rem; margin-top: 6px;
        font-weight: 500; position: relative; z-index: 1;
    }
    .hero-pills { margin-top: 16px; position: relative; z-index: 1; }
    .hero-pill {
        display:inline-block; background: rgba(255,255,255,0.18);
        backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.35);
        color: white; padding: 6px 14px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin-right: 8px;
    }

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #140f28 0%, #0c0f1c 100%);
        border-right: 1px solid #2a2545;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #e6dfff !important;
    }
    .sidebar-chip {
        display:inline-block; padding:3px 10px; border-radius:8px;
        background:#241c47; color:#c9b8ff; font-size:0.72rem; font-weight:700;
        letter-spacing:0.04em; text-transform:uppercase; margin-bottom:10px;
    }

    /* ---------------- KPI row ---------------- */
    .kpi-card {
        background: linear-gradient(145deg, #17182c 0%, #1e1f3a 100%);
        border: 1px solid #2c2b4d;
        border-radius: 16px;
        padding: 16px 18px;
        text-align: center;
    }
    .kpi-label { color: #9d97c7; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { color: #ffffff; font-size: 1.5rem; font-weight: 800; margin-top: 4px; }

    /* ---------------- Candidate cards ---------------- */
    .candidate-card {
        background: linear-gradient(150deg, rgba(31,26,58,0.9) 0%, rgba(24,22,44,0.95) 100%);
        border: 1px solid rgba(140, 120, 220, 0.25);
        border-radius: 18px;
        padding: 24px 28px 8px 28px;
        margin-bottom: 6px;
        box-shadow: 0 12px 30px -18px rgba(0,0,0,0.6);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 34px -16px rgba(120, 90, 230, 0.45);
    }
    .rank-badge {
        display:inline-block; padding:5px 14px; border-radius:999px;
        font-weight:800; font-size:0.85rem; color:white; letter-spacing:0.01em;
        box-shadow: 0 6px 16px -6px rgba(0,0,0,0.6);
    }
    .candidate-name {
        margin-top:12px; margin-bottom: 2px; font-size: 1.4rem; font-weight: 700; color: #f4f1ff;
    }

    /* ---------------- Skill badges ---------------- */
    .badge-skill {
        display:inline-block; padding:4px 12px; margin:3px 4px 3px 0;
        border-radius:999px; font-size:0.78rem; font-weight:600;
    }
    .badge-match { background: rgba(47, 212, 122, 0.14); color:#5fe895; border:1px solid rgba(47, 212, 122, 0.4); }
    .badge-missing { background: rgba(255, 94, 94, 0.12); color:#ff9494; border:1px solid rgba(255, 94, 94, 0.35); }

    .section-label {
        font-size: 0.82rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; color: #b8aeed; margin-bottom: 6px;
    }

    /* ---------------- Metrics ---------------- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 10px 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #f4f1ff; }
    div[data-testid="stMetricLabel"] { color: #9d97c7; }

    /* ---------------- Progress bar ---------------- */
    div[data-testid="stProgress"] > div > div {
        background-image: linear-gradient(90deg, #6a3df5, #ff5ea8);
    }

    /* ---------------- Buttons ---------------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(120deg, #6a3df5, #ff5ea8);
        color: white; font-weight: 700; border: none; border-radius: 12px;
        padding: 0.6rem 1.4rem; box-shadow: 0 10px 24px -10px rgba(162,75,240,0.6);
        transition: transform 0.12s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px); filter: brightness(1.08); color: white;
    }

    .stCaption, .st-emotion-cache-1629p8f { color: #a79fce !important; }

    hr, div[data-testid="stDivider"] { border-color: #2a2545 !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03); border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🧑‍💼 AI HR Recruitment Assistant</div>
    <div class="hero-sub">Agentic AI that reads job descriptions, scores every resume, and preps your interview questions — automatically.</div>
    <div class="hero-pills">
        <span class="hero-pill">🔎 RAG · skill + TF-IDF retrieval</span>
        <span class="hero-pill">🛠️ Tool calling · hybrid scoring</span>
        <span class="hero-pill">🧠 Interview Q generation</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not USE_LLM:
    st.info("🔌 Running in **offline fallback mode** (no `ANTHROPIC_API_KEY` found). "
            "Scoring is fully functional; interview questions use rule-based generation.")
else:
    st.success("🤖 Connected to Claude for AI-generated interview questions.")

# ---------------------------------------------------------------------- side
with st.sidebar:
    st.markdown('<span class="sidebar-chip">Configuration</span>', unsafe_allow_html=True)
    st.header("⚙️ Scoring Settings")
    skill_weight = st.slider(
        "Skill-match weight", 0.0, 1.0, 0.6, 0.05,
        help="Final score = (this × skill match %) + ((1 − this) × TF-IDF content similarity %)"
    )
    st.caption(f"🧮 Content-similarity weight: **{round(1 - skill_weight, 2)}**")

    st.divider()
    st.markdown('<span class="sidebar-chip">Data Sources</span>', unsafe_allow_html=True)
    st.header("📤 Upload Files")
    jd_file = st.file_uploader("Job Description (.txt / .pdf / .docx)", type=["txt", "pdf", "docx"])
    resume_files = st.file_uploader(
        "Candidate Resumes (.txt / .pdf / .docx)",
        type=["txt", "pdf", "docx"], accept_multiple_files=True
    )
    st.caption("💡 No files uploaded? The app runs on 3 bundled sample resumes so it's always demo-able.")

# ---------------------------------------------------------------- load data
if jd_file is not None:
    jd_file.seek(0)
    jd_text = read_uploaded_file(jd_file)
    jd_source = f"Uploaded: {jd_file.name}"
else:
    jd_text = load_sample_jd()
    jd_source = "Sample JD (Junior Data Analyst)"

if resume_files:
    resumes = {}
    for f in resume_files:
        f.seek(0)
        text = read_uploaded_file(f)
        if text.strip():
            resumes[os.path.splitext(f.name)[0]] = text
    resume_source = f"{len(resumes)} uploaded resume(s)"
else:
    resumes = load_sample_resumes()
    resume_source = "3 bundled sample resumes"

# ---------------------------------------------------------------- KPI row
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Job Description</div>
    <div class="kpi-value" style="font-size:1rem;">{jd_source}</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Resumes Loaded</div>
    <div class="kpi-value">{len(resumes)}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Scoring Mode</div>
    <div class="kpi-value" style="font-size:1rem;">{'LLM + Rules' if USE_LLM else 'Offline Rules'}</div></div>""", unsafe_allow_html=True)

st.write("")

with st.expander("📄 View Job Description", expanded=False):
    st.text(jd_text or "(empty)")

st.subheader("📚 Knowledge Base — Resumes")
if resumes:
    cols = st.columns(min(len(resumes), 4))
    for i, (name, text) in enumerate(resumes.items()):
        with cols[i % len(cols)]:
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            with st.expander("View resume"):
                st.text(text[:3000] or "(empty)")
else:
    st.warning("No resumes loaded yet — upload at least one, or rely on the sample set.")

st.divider()

# --------------------------------------------------------------- run agent
run_clicked = st.button("🚀 Run Recruitment Agent", type="primary", disabled=not (jd_text.strip() and resumes))

if run_clicked:
    with st.spinner("🤖 Agent is retrieving skills, scoring candidates, and generating interview questions..."):
        results = run_recruitment_agent(jd_text, resumes, skill_weight)

    st.subheader("📊 Ranked Candidates")

    # --- summary KPIs
    top = results[0]
    avg_score = round(sum(r["final_score"] for r in results) / len(results), 1)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">🏆 Top Candidate</div>
        <div class="kpi-value" style="font-size:1.1rem;">{top['candidate'].replace('_',' ').title()}</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Top Score</div>
        <div class="kpi-value">{top['final_score']}%</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Average Score</div>
        <div class="kpi-value">{avg_score}%</div></div>""", unsafe_allow_html=True)

    st.write("")

    # --- comparison chart
    chart_df = pd.DataFrame([
        {"Candidate": r["candidate"].replace("_", " ").title(),
         "Skill Match %": r["match_percent"],
         "Content Similarity %": r["content_similarity"],
         "Final Score %": r["final_score"]}
        for r in results
    ])
    fig = px.bar(
        chart_df.melt(id_vars="Candidate", var_name="Metric", value_name="Score"),
        x="Candidate", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#6a3df5", "#ff5ea8", "#2fd47a"],
        template="plotly_dark", height=380,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), legend_title_text="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d8d3f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    medal = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(results):
        rank_icon = medal[i] if i < 3 else f"#{i+1}"
        badge_color = "#2fa84f" if r["final_score"] >= 70 else "#c99a2e" if r["final_score"] >= 40 else "#c9422e"

        st.markdown(f"""
        <div class="candidate-card">
            <span class="rank-badge" style="background:{badge_color}">{rank_icon} {r['final_score']}% overall</span>
            <div class="candidate-name">{r['candidate'].replace('_',' ').title()}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Skill Match", f"{r['match_percent']}%")
        m2.metric("Content Similarity", f"{r['content_similarity']}%")
        m3.metric("Final Score", f"{r['final_score']}%")
        st.progress(min(int(r["final_score"]), 100) / 100)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">✅ Matched Skills</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="badge-skill badge-match">{s}</span>' for s in r["matched_skills"]) or "None", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-label">⚠️ Missing Skills</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="badge-skill badge-missing">{s}</span>' for s in r["missing_skills"]) or "None", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:14px;">🎤 Suggested Interview Questions</div>', unsafe_allow_html=True)
        for qi, q in enumerate(r["interview_questions"], 1):
            st.write(f"{qi}. {q}")

        st.divider()

    # --- export
    export_df = pd.DataFrame([{
        "Candidate": r["candidate"],
        "Skill Match %": r["match_percent"],
        "Content Similarity %": r["content_similarity"],
        "Final Score %": r["final_score"],
        "Matched Skills": ", ".join(r["matched_skills"]),
        "Missing Skills": ", ".join(r["missing_skills"]),
        "Interview Questions": " | ".join(r["interview_questions"]),
    } for r in results])
    st.download_button(
        "⬇️ Download Results as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="recruitment_results.csv",
        mime="text/csv",
    )

st.caption("Built for TNSDC – IBM Agentic AI Internship · Use Case 2: AI HR Recruitment Assistant (v3 — Advanced, Polished UI)")
