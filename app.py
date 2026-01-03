import streamlit as st
import os
import tempfile
from pathlib import Path
from graph import create_forensic_graph
from nodes import ForensicState
import json

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
    
    .report-content {
        background: #1E1E1E; 
        padding: 1.5rem;
        border: 1px solid #333;
        border-radius: 0 0 10px 10px;
        color: #FFFFFF;
        line-height: 1.6;
        white-space: normal;      
        word-wrap: break-word;    
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

if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800;900&family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');

    .main-header {
        position: relative;
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        padding: 3.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        overflow: hidden;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-overlay {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(4px);
        z-index: 1;
    }

    .header-content {
        position: relative;
        z-index: 2;
    }

    .main-header h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        font-size: 4rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: -2px;
        line-height: 1;
        background: linear-gradient(to bottom, #ffffff 60%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.4));
    }

    .main-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
        max-width: 700px;
        margin: 15px auto 0;
        letter-spacing: 0.5px;
    }
    
    .header-scan-line {
        position: absolute;
        width: 100%;
        height: 3px;
        background: rgba(255, 255, 255, 0.2);
        top: 0;
        left: 0;
        animation: scan 5s linear infinite;
        z-index: 2;
    }

    @keyframes scan {
        0% { top: 0%; opacity: 0; }
        50% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
</style>

<div class="main-header">
    <div class="header-overlay"></div>
    <div class="header-scan-line"></div>
    <div class="header-content">
        <h1>Forensic Image Analyzer</h1>
        <p>Advanced AI Orchestration for Digital Authenticity & Tamper Detection</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <style>
            .sidebar-team-header {
                text-align: center;
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.1rem;
                font-weight: 700;
                color: #A9A9A9;
                letter-spacing: 1px;
                text-transform: uppercase;
                margin: 0;
                padding: 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-team-header">Team ChaiKadak</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 System Status")
    st.success("Core Engine: Active")
    st.info("⚠️ This analysis is performed by AI models and should be used at your own discretion")
    
    st.markdown("---")
    if st.button("🗑️ Reset Workspace"):
        st.session_state.results = None
        st.rerun()

        
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

if analyze_button and uploaded_file:
    with st.spinner("🕵️ Orchestrating forensic agent..."):
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

tab1, tab2, tab3 = st.tabs(["📊 Verification Results", "🔍 Audit Logs", "ℹ️ Methodology & Calculation"])

with tab1:
    if st.session_state.results:
        res = st.session_state.results
        score = int(res.get('confidence_score', 0) * 100)
        label = res.get("confidence_label", "Unknown")
        count = len(res.get('conclusions', {}))


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
        st.markdown(f"""
        <div style="background: black; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <pre style="white-space: pre-wrap; font-family: 'Segoe UI', sans-serif; color: #FFFFFF; line-height: 1.6; margin: 0;">{res.get('final_report', 'No report generated.')}</pre>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting input. Please upload an image and execute the pipeline.")

with tab2:
    if st.session_state.results:
        res = st.session_state.results
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Conclusion Details")
            for node, conclusion in res.get('conclusions', {}).items():
                header_name = "Error Level Analysis (ELA)" if node == "ela" else node.replace('_', ' ').title()
                with st.expander(f"{header_name}", expanded=True):
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
    st.write("The system utilizes a **StateGraph** to manage a multi-step forensic audit. An initial **'Planner'** agent evaluates the extracted metadata to determine which specific forensic nodes—such as Error Level Analysis (ELA) or OSINT—are necessary.")

    st.markdown("""
    * **Dynamic Investigation Scoping**: The Planner analyzes metadata to populate a `planned_run` queue, identifying only the examinations required to reach a high-confidence conclusion while skipping redundant steps.
    * **State-Persistent Orchestration**: A central `ForensicState` object tracks all visual descriptions, technical metadata, and individual node conclusions to maintain data integrity throughout the audit.
    * **Deterministic Routing**: The orchestrator follows 'Hard Invariants' where the routing space strictly shrinks after each node is executed, preventing infinite loops and ensuring a final report is always generated.
    * **Adaptive Fail-Safes**: If the Planner cannot reach a deterministic decision, the system defaults to a baseline verification (e.g., Visual Environment) to ensure forensic coverage is never compromised.
    """)
    
    st.markdown("### 2. Forensic Node Methodology")
    st.markdown("""
<style>
    .methodology-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }

    .method-card {
        background: #ffffff;
        border: 1px solid #e0e6ed;
        border-left: 5px solid #048A81;
        padding: 1.5rem;
        border-radius: 8px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .method-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }

    .method-title {
        color: #2E4057;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .method-text {
        color: #4A5568;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }

    .method-icon {
        font-size: 1.3rem;
    }
</style>

<div class="methodology-grid">
    <div class="method-card">
        <div class="method-title"><span class="method-icon">🏷️</span> Metadata Audit</div>
        <p class="method-text">Extracts embedded <b>EXIF and GPS tags</b> to identify software modification signatures or device inconsistencies. It cross-references hardware profiles to detect spoofed origins.</p>
    </div>
    <div class="method-card" style="border-left-color: #667eea;">
        <div class="method-title"><span class="method-icon">📉</span> Error Level Analysis</div>
        <p class="method-text">Identifies manipulation by detecting <b>JPEG compression variances</b>. Localized inconsistencies in the error level often pinpoint added, removed, or retouched visual elements.</p>
    </div>
    <div class="method-card" style="border-left-color: #fdbb2d;">
        <div class="method-title"><span class="method-icon">👁️</span> Visual Environment</div>
        <p class="method-text">Leverages <b>Vision-Language Models</b> to verify semantic consistency. It ensures the visual scene (lighting, weather, objects) logically matches the recorded metadata context.</p>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 3. Confidence Scoring Formula")
    st.write("The Confidence Score is a weighted metric derived from the strength of findings and the coverage of the audit:")
    st.latex(r"Final\ Score = Raw\ LLM\ Score \times \left( \frac{Executed\ Nodes}{Total\ Planned\ Nodes} \right)")

    st.markdown("---")
    st.markdown("### 🍵 About Team ChaiKadak")
    st.write("""- **Anmol Bhatnagar** Enrollment: 992401040039  
- **Dhruv Arora** Enrollment: 992401040023  """)

st.markdown("---")
st.caption("Forensic Image Analyzer v2.0 | Team ChaiKadak")