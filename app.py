import streamlit as st
import os
import pandas as pd
from agent import build_workflow, load_data, trend_function, comparison_function, breakdown_function, code_viewer, summarize




@st.cache_data
def get_dataframe():
    csv_path = "master_vehicle_data.csv"
    if not os.path.exists(csv_path):
        from pre_process import process_vehicle_data
        process_vehicle_data()
    return pd.read_csv(csv_path)


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Registration Analyst",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
} 
.stApp {
    background: #0d0d0d;
    color: #f0ede6;
}
.main-header {
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0ede6;
    letter-spacing: -0.03em;
    margin: 0;
}
.main-header span {
    color: #c8f135;
}
.main-header p {
    color: #666;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}
.query-box {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.step-badge {
    display: inline-block;
    background: #c8f135;
    color: #0d0d0d;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.code-container {
    background: #111;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c8f135;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #a8d8a8;
    overflow-x: auto;
    white-space: pre-wrap;
    margin: 1rem 0;
}
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.summary-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    border-left: 4px solid #c8f135;
}
.summary-card h3 {
    color: #c8f135;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.summary-card p {
    color: #ccc;
    line-height: 1.75;
    font-size: 0.97rem;
}
.sidebar-info {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #888;
}

/* Always show the sidebar collapse button visibly */
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            background-color: #a3e635 !important;
            border-radius: 50% !important;
            top: 50% !important;
}
            
.sidebar-info strong {
    color: #c8f135;
    display: block;
    margin-bottom: 0.3rem;
}
.error-box {
    background: #1a0a0a;
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 1rem;
    color: #ff8888;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
}
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.status-running { background: #1a1a00; color: #ffdd00; border: 1px solid #ffdd00; }
.status-done    { background: #0a1a0a; color: #c8f135; border: 1px solid #c8f135; }
.status-waiting { background: #1a0a1a; color: #cc88ff; border: 1px solid #cc88ff; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "stage": "input",           # input → code_review → result
        "generated_code": "",
        "intermediate_state": None,
        "summary": "",
        "analysis_result": None,
        "error_log": [],
        "query_intent": "",
        "history": [],              # list of {query, summary, chart}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚗 How to Query")
    st.markdown("""
    <div class="sidebar-info">
        <strong>Trend Analysis</strong>
        "Trend of Petrol vehicles in Odisha from 2014 to 2024"
    </div>
    <div class="sidebar-info">
        <strong>Comparison</strong>
        "Compare Petrol and Diesel in Delhi from 2016 to 2022"
    </div>
    <div class="sidebar-info">
        <strong>Breakdown</strong>
        "Show distribution of vehicle categories in Bihar for 2021"
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.session_state.history:
        st.markdown("### 📋 Past Queries")
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Q: {item['query'][:40]}..."):
                st.write(item['summary'][:200] + "...")

    st.divider()
    if st.button("🔄 Reset / New Query", use_container_width=True):
        for key in ["stage", "generated_code", "intermediate_state",
                    "summary", "analysis_result", "error_log", "query_intent"]:
            st.session_state[key] = "" if key not in ["stage"] else "input"
            if key == "stage":
                st.session_state.stage = "input"
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>Vehicle Registration <span>Analyst</span></h1>
    <p>AI-powered analysis of India's vehicle registration data — powered by LangGraph + Groq</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Query Input
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.stage == "input":

    st.markdown('<div class="step-badge">Step 1 of 3 — Ask a Question</div>', unsafe_allow_html=True)

    with st.container():
        query = st.text_area(
            "Enter your query",
            placeholder="e.g. Compare the trend of Electric vehicles across Delhi and Maharashtra from 2018 to 2024",
            height=100,
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            run_btn = st.button("▶ Analyse", use_container_width=True, type="primary")

    if run_btn and query.strip():
        with st.spinner("🔍 Extracting intent and generating analysis code..."):
            try:
                # Step 1: Load data
                initial_state = {"user_query": query.strip()}
                data_state = load_data(initial_state)
                initial_state["current_dataframe"] = get_dataframe()
                initial_state.update(data_state)

                # Step 2: Run workflow up to code generation only (skip human_approval)
                from agent import extract_query_metadata, route_analysis, trend_function, comparison_function, breakdown_function

                meta_result = extract_query_metadata(initial_state)
                initial_state.update(meta_result)

                intent = initial_state["query_intent"]
                st.session_state.query_intent = intent

                # Route and generate code
                if intent == "trend_analysis":
                    code_result = trend_function(initial_state)
                elif intent == "comparison":
                    code_result = comparison_function(initial_state)
                else:
                    code_result = breakdown_function(initial_state)

                initial_state.update(code_result)
                initial_state["human_approved_code_status"] = ""
                initial_state["human_feedback"] = []

                st.session_state.intermediate_state = initial_state
                st.session_state.generated_code = initial_state["generated_code"]
                st.session_state.stage = "code_review"
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

    elif run_btn:
        st.warning("Please enter a query first.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Code Review & Human Approval
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "code_review":

    state = st.session_state.intermediate_state
    intent_label = st.session_state.query_intent.replace("_", " ").title()

    st.markdown(f'<div class="step-badge">Step 2 of 3 — Review Generated Code</div>', unsafe_allow_html=True)

    col_info, col_badge = st.columns([3, 1])
    with col_info:
        st.markdown(f"**Query:** {state['user_query']}")
    with col_badge:
        st.markdown(f'<span class="status-pill status-waiting">Intent: {intent_label}</span>', unsafe_allow_html=True)

    st.markdown("The AI generated the following code to answer your query. Review it before executing:")

    st.markdown(
        f'<div class="code-container">{st.session_state.generated_code}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve & Run", use_container_width=True, type="primary"):
            st.session_state.intermediate_state["human_approved_code_status"] = "yes"
            st.session_state.intermediate_state["human_feedback"] = []
            st.session_state.stage = "running"
            st.rerun()

    with col2:
        with st.expander("❌ Reject & Give Feedback"):
            feedback = st.text_area("What should be fixed?", placeholder="e.g. Use ELECTRIC(BOV) instead of just ELECTRIC", key="feedback_input")
            if st.button("🔁 Regenerate with Feedback", use_container_width=True):
                with st.spinner("Regenerating code with your feedback..."):
                    try:
                        state["human_feedback"] = [feedback]
                        intent = st.session_state.query_intent
                        if intent == "trend_analysis":
                            new_code = trend_function(state)
                        elif intent == "comparison":
                            new_code = comparison_function(state)
                        else:
                            new_code = breakdown_function(state)

                        state.update(new_code)
                        st.session_state.generated_code = state["generated_code"]
                        st.session_state.intermediate_state = state
                        st.rerun()
                    except Exception as e:
                        st.error(f"Regeneration failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Execute & Show Results
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "running":

    state = st.session_state.intermediate_state

    st.markdown('<div class="step-badge">Step 3 of 3 — Results</div>', unsafe_allow_html=True)
    st.markdown(f"**Query:** {state['user_query']}")

    with st.spinner("⚙️ Executing analysis and generating summary..."):
        try:
            # Execute code
            exec_result = code_viewer(state)
            state.update(exec_result)

            if state["error_log"]:
                st.markdown(
                    f'<div class="error-box">❌ Execution Error:<br>{state["error_log"][0]}</div>',
                    unsafe_allow_html=True
                )
                if st.button("← Go back and fix"):
                    st.session_state.stage = "code_review"
                    st.rerun()
            else:
                # Generate summary
                summary_result = summarize(state)
                state.update(summary_result)

                st.session_state.summary = state["summary"]
                st.session_state.analysis_result = state["analysis_result"]
                st.session_state.stage = "done"

                # Save to history
                st.session_state.history.append({
                    "query": state["user_query"],
                    "summary": state["summary"],
                })
                st.rerun()

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Done: Show Chart + Summary
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "done":

    state = st.session_state.intermediate_state

    st.markdown('<div class="step-badge">✓ Analysis Complete</div>', unsafe_allow_html=True)
    st.markdown(f"**Query:** {state['user_query']}")
    st.markdown(f'<span class="status-pill status-done">Done</span>', unsafe_allow_html=True)

    # Chart
    if os.path.exists("chart.png"):
        st.markdown("### 📊 Chart")
        st.image("chart.png", use_container_width=True)
    else:
        # Try legacy chart names
        for name in ["comparison_chart.png", "breakdown_chart.png", "trend_chart.png"]:
            if os.path.exists(name):
                st.image(name, use_container_width=True)
                break

    # Summary
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>📝 AI Summary</h3>
            <p>{st.session_state.summary}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Raw result expander
    if st.session_state.analysis_result is not None:
        with st.expander("🔢 Raw Analysis Data"):
            result = st.session_state.analysis_result
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
            elif isinstance(result, dict):
                st.json(result)
            else:
                st.write(result)

    # Generated code expander
    with st.expander("🧾 View Executed Code"):
        st.code(st.session_state.generated_code, language="python")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ask Another Question", use_container_width=True, type="primary"):
            st.session_state.stage = "input"
            st.rerun()
    with col2:
        if st.button("🔁 Re-run Same Query", use_container_width=True):
            st.session_state.stage = "code_review"
            st.rerun()
