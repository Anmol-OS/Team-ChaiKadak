import streamlit as st
import os
import tempfile
from pathlib import Path
from graph import create_forensic_graph
from nodes import ForensicState
import json

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Forensic Image Analyzer | Team ChaiKadak",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-color: #2E4057;
        --secondary-color: #048A81;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E4057 0%, #048A81 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        position: relative;
    }

    .team-badge {
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 0.8rem;
        opacity: 0.8;
        letter-spacing: 1px;
    }

    /* Perfectly aligned Flexbox metrics */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 150px;
    }

    .metric-value { 
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #2E4057; 
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    .conclusion-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px 10px 0 0;
        margin-top: 1rem;
    }
    
    /* FIX: Proper text wrapping for the report content */
    .report-content {
        background: #1E1E1E; /* Matches your dark terminal style */
        padding: 1.5rem;
        border: 1px solid #333;
        border-radius: 0 0 10px 10px;
        color: #FFFFFF;
        line-height: 1.6;
        white-space: normal;      /* Standard wrapping */
        word-wrap: break-word;    /* Break long words */
        overflow-wrap: break-word;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .methodology-box {
        background-color: #f9f9f9;
        border-left: 5px solid var(--secondary-color);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.markdown("""
<div class="main-header">
    <div class="team-badge">BY TEAM CHAIKADAK</div>
    <h1>🔍 Forensic Image Analyzer</h1>
    <p>AI Orchestration for Authentic Image Verification</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 System Status")
    st.success("Core Engine: Active")
    st.info("Analysis Tier: Paid (Llama 4 Scout)")
    st.markdown("---")
    if st.button("🗑️ Reset Workspace"):
        st.session_state.results = None
        st.rerun()
    st.caption("Developed by Team ChaiKadak")

# ============================================================
# ANALYSIS CONSOLE
# ============================================================
st.markdown("### 🔬 Input Selection")
col_up, col_preview = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'])

with col_preview:
    if uploaded_file:
        st.image(uploaded_file, caption="Target Image", width=250)
        analyze_button = st.button("🚀 Execute Analysis Pipeline", use_container_width=True, type="primary")
    else:
        analyze_button = False

# ============================================================
# EXECUTION LOGIC
# ============================================================
if analyze_button and uploaded_file:
    with st.spinner("🕵️ Orchestrating forensic agents..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            graph = create_forensic_graph()
            initial_state = ForensicState(
                image_path=tmp_path,
                messages=[],
                visual_description="",
                metadata={},
                ela={},
                osint={},
                conclusions={},
                skipped_exams={},
                planned_run=[],
                confidence_label="",
                confidence_score=0.0,
                confidence_reasoning="",
                final_report=""
            )
            
            st.session_state.results = graph.invoke(initial_state)
            st.rerun() 
            
        except Exception as e:
            st.error(f"Analysis Pipeline Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ============================================================
# RESULTS TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Verification Results", "🔍 Audit Logs", "ℹ️ Methodology & Calculation"])

with tab1:
    if st.session_state.results:
        res = st.session_state.results
        score = int(res.get('confidence_score', 0) * 100)
        label = res.get("confidence_label", "Unknown")
        count = len(res.get('conclusions', {}))

        # Metrics Row
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-value">{score}%</div>
                <div class="metric-label">Calculated Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.5rem;">{label}</div>
                <div class="metric-label">Security Tier</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{count}</div>
                <div class="metric-label">Completed Nodes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="conclusion-header"><h3>📋 Final Expert Conclusion</h3></div>', unsafe_allow_html=True)
        # Using a div with the report-content class ensures text wraps properly
        st.markdown(f'<div class="report-content">{res.get("final_report", "No report generated.")}</div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting input. Please upload an image and execute the pipeline.")

with tab2:
    if st.session_state.results:
        res = st.session_state.results
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Conclusion Details")
            for node, conclusion in res.get('conclusions', {}).items():
                # HEADER UPDATE: Renamed ELA
                header_name = "Error Level Analysis (ELA)" if node == "ela" else node.replace('_', ' ').title()
                with st.expander(f"Result: {header_name}", expanded=True):
                    st.write(conclusion)
        with c2:
            st.subheader("⏭️ Process Optimization")
            for node, reason in res.get('skipped_exams', {}).items():
                header_name = "Error Level Analysis (ELA)" if node == "ela" else node.replace('_', ' ').title()
                with st.expander(f"Skipped: {header_name}"):
                    st.write(reason)
        st.divider()
        st.subheader("🛠️ Technical Data Export")
        st.json({"metadata": res.get("metadata"), "Error Level Analysis (ELA)": res.get("ela"), "osint": res.get("osint")})

with tab3:
    st.markdown("## 🧠 How the Analysis is Calculated")
    
    st.markdown("### 1. Orchestration & Planning")
    st.write("The system utilizes a StateGraph to manage a multi-step forensic audit. An initial 'Planner' agent evaluates the extracted metadata to determine which specific forensic nodes—such as Error Level Analysis (ELA) or OSINT—are necessary.")

    st.markdown("### 2. Forensic Node Methodology")
    st.markdown(f"""
    <div class="methodology-box">
    <b>Metadata Audit:</b> Extracts embedded EXIF and GPS tags to find software modification signatures or device inconsistencies.<br><br>
    <b>Error Level Analysis (ELA):</b> Identifies image manipulation by detecting differences in JPEG compression levels throughout the image. Localized inconsistencies often point to added or retouched elements.<br><br>
    <b>Visual Environment:</b> Uses vision-language models to verify if the visual scene matches the provided metadata context.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 3. Confidence Scoring Formula")
    st.write("The Confidence Score is a weighted metric derived from the strength of findings and the coverage of the audit:")
    st.latex(r"Final\ Score = Raw\ LLM\ Score \times \left( \frac{Executed\ Nodes}{Total\ Planned\ Nodes} \right)")

    st.markdown("---")
    st.markdown("### 🍵 About Team ChaiKadak")
    st.write("Team ChaiKadak develops robust AI solutions focused on transparency and forensic integrity.")

st.markdown("---")
st.caption("Forensic Image Analyzer v2.0 | Team ChaiKadak")