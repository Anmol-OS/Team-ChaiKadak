import streamlit as st
import os
import tempfile
from pathlib import Path
from graph import create_forensic_graph
from nodes import ForensicState
import json

# Page configuration
st.set_page_config(
    page_title="Forensic Image Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E4057;
        --secondary-color: #048A81;
        --accent-color: #54C6EB;
        --danger-color: #F24236;
        --success-color: #06D6A0;
        --warning-color: #FFD23F;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2E4057 0%, #048A81 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #048A81;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E4057;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    .badge-high {
        background-color: #06D6A0;
        color: white;
    }
    
    .badge-medium {
        background-color: #FFD23F;
        color: #2E4057;
    }
    
    .badge-low {
        background-color: #F24236;
        color: white;
    }
    
    .badge-executed {
        background-color: #048A81;
        color: white;
    }
    
    .badge-skipped {
        background-color: #e0e0e0;
        color: #666;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #048A81;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    /* Progress indicator */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .progress-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    
    /* Results section */
    .conclusion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .conclusion-box h3 {
        margin-top: 0;
        font-size: 1.5rem;
    }
    
    /* Expandable sections */
    .stExpander {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #048A81 0%, #2E4057 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Image preview */
    .image-preview {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Forensic Image Analyzer</h1>
    <p>Advanced AI-powered image authenticity verification system</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 About This Tool")
    st.info("""
    This forensic analyzer uses multiple examination techniques to assess image authenticity:
    
    - **Metadata Analysis**: Extracts EXIF data
    - **Visual Analysis**: AI-powered scene understanding
    - **ELA**: Error Level Analysis for tampering detection
    - **OSINT**: Reverse image search
    """)
    
    st.markdown("### ⚙️ System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Status", "🟢 Active")
    with col2:
        st.metric("Version", "2.0")
    
    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. Upload an image
    2. Automated planning determines necessary tests
    3. Forensic examinations are executed
    4. Comprehensive report is generated
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["🔬 Analysis", "📊 Results", "ℹ️ Information"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Image for Analysis")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image for forensic analysis"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown("### Analysis Options")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h4 style="color: white; margin-top: 0;">🎯 Adaptive Analysis</h4>
            <p style="color: rgba(255,255,255,0.95); margin-bottom: 0;">The system automatically determines which examinations are necessary based on available data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file:
            analyze_button = st.button("🚀 Start Analysis", use_container_width=True, type="primary")
        else:
            st.warning("⚠️ Please upload an image first")
            analyze_button = False

with tab2:
    # Force refresh when results are available
    if 'results' not in st.session_state or st.session_state.results is None:
        st.info("👈 Upload an image and run analysis to see results here")
    else:
        results = st.session_state.results
        
        # Debug: Show that we have results
        st.markdown("## 📈 Analysis Results")
        
        # Confidence Score Display
        col1, col2, col3 = st.columns(3)
        
        confidence_score = int(results.get('confidence_score', 0) * 100)
        confidence_label = results.get('confidence_label', 'Unknown')
        executed_count = len(results.get('conclusions', {}))
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value">{confidence_score}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            badge_class = 'badge-high' if 'High' in confidence_label else ('badge-medium' if 'Medium' in confidence_label else 'badge-low')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence Level</div>
                <div class="metric-value" style="font-size: 1.5rem;">
                    <span class="status-badge {badge_class}">{confidence_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tests Executed</div>
                <div class="metric-value">{executed_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Final Report
        st.markdown("---")
        final_report_text = results.get('final_report', 'No report available')
        st.markdown("""
        <div class="conclusion-box">
            <h3>📋 Final Expert Conclusion</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display report text in readable format
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <pre style="white-space: pre-wrap; font-family: 'Segoe UI', sans-serif; color: #2E4057; line-height: 1.6; margin: 0;">{final_report_text}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Results
        st.markdown("---")
        st.markdown("## 🔍 Detailed Examination Results")
        
        conclusions = results.get('conclusions', {})
        skipped = results.get('skipped_exams', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Executed Examinations")
            if conclusions:
                for exam_name, conclusion in conclusions.items():
                    with st.expander(f"📌 {exam_name.replace('_', ' ').title()}", expanded=False):
                        st.markdown(f'<span class="status-badge badge-executed">Executed</span>', unsafe_allow_html=True)
                        st.write(conclusion)
            else:
                st.info("No examinations were executed")
        
        with col2:
            st.markdown("### ⏭️ Skipped Examinations")
            if skipped:
                for exam_name, reason in skipped.items():
                    with st.expander(f"📌 {exam_name.replace('_', ' ').title()}", expanded=False):
                        st.markdown(f'<span class="status-badge badge-skipped">Skipped</span>', unsafe_allow_html=True)
                        st.write(reason)
            else:
                st.success("All planned examinations were executed")
        
        # Technical Data
        st.markdown("---")
        st.markdown("## 🔧 Technical Data")
        
        with st.expander("📊 Metadata", expanded=False):
            metadata = results.get('metadata', {})
            if metadata:
                st.json(metadata)
            else:
                st.info("No metadata available")
        
        with st.expander("📈 ELA Analysis", expanded=False):
            ela = results.get('ela', {})
            if ela:
                st.json(ela)
            else:
                st.info("ELA analysis not performed")
        
        with st.expander("🌐 OSINT Results", expanded=False):
            osint = results.get('osint', {})
            if osint:
                st.json(osint)
            else:
                st.info("OSINT search not performed")
        
        # Download Report
        st.markdown("---")
        report_text = results.get('final_report', '')
        st.download_button(
            label="📥 Download Full Report",
            data=report_text,
            file_name="forensic_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

with tab3:
    st.markdown("## ℹ️ Examination Methods")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Metadata Analysis
        Extracts and analyzes EXIF data including:
        - Device information
        - GPS coordinates
        - Timestamps
        - Software information
        
        ### 🖼️ Visual Environment
        AI-powered visual analysis to identify:
        - Scene composition
        - Objects and elements
        - Potential inconsistencies
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Error Level Analysis (ELA)
        Detects compression artifacts that may indicate:
        - Image manipulation
        - Copy-paste edits
        - Retouching areas
        
        ### 🌐 OSINT Search
        Reverse image search to find:
        - Original sources
        - Similar images
        - Publication history
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 Confidence Scoring")
    st.markdown("""
    The confidence score is calculated based on:
    - **Strength of findings**: How conclusive each examination is
    - **Coverage**: Number of examinations successfully executed
    - **Consistency**: Agreement between different examination methods
    
    **High Confidence (70-100%)**: Strong evidence from multiple sources  
    **Medium Confidence (40-69%)**: Some evidence but with limitations  
    **Low Confidence (0-39%)**: Limited or inconclusive evidence
    """)

# Analysis Logic
if analyze_button:
    with st.spinner("🔄 Running forensic analysis..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Progress tracking
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### 🔄 Analysis Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📋 Initializing forensic graph...")
                progress_bar.progress(10)
                
                # Create and run forensic graph
                graph = create_forensic_graph()
                
                status_text.text("🔍 Extracting metadata...")
                progress_bar.progress(30)
                
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
                
                status_text.text("⚙️ Planning examinations...")
                progress_bar.progress(50)
                
                # Execute graph
                result = graph.invoke(initial_state)
                
                status_text.text("🎯 Calculating confidence...")
                progress_bar.progress(80)
                
                status_text.text("📝 Generating report...")
                progress_bar.progress(95)
                
                # Store results
                st.session_state.results = result
                st.session_state.analysis_complete = True
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
            
            st.success("✅ Analysis completed successfully!")
            st.balloons()  # Add celebration effect
            
            # Show results immediately in expanders
            st.markdown("---")
            st.markdown("## 📊 Quick Results Preview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence_score = int(result.get('confidence_score', 0) * 100)
                st.metric("Confidence Score", f"{confidence_score}%")
            
            with col2:
                confidence_label = result.get('confidence_label', 'Unknown')
                st.metric("Confidence Level", confidence_label)
            
            with col3:
                executed_count = len(result.get('conclusions', {}))
                st.metric("Tests Executed", executed_count)
            
            with st.expander("📋 View Final Report", expanded=True):
                st.markdown(result.get('final_report', 'No report available'))
            
            st.info("👉 **Switch to the 'Results' tab** for detailed examination findings and technical data")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔒 Forensic Image Analyzer v2.0 | Built with AI-powered analysis</p>
    <p style="font-size: 0.9rem;">⚠️ This tool provides forensic analysis for investigative purposes. Results should be verified by qualified experts.</p>
</div>
""", unsafe_allow_html=True)