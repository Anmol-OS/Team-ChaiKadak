import os
import streamlit as st
from PIL import Image
from graph import create_forensic_graph

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Forensic Image Authenticity Auditor",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Forensic Image Authenticity Auditor")

st.markdown(
    """
This application performs an **adaptive forensic examination** of an image.

### Core principles
- Original image bytes are preserved (EXIF-safe)
- Missing metadata is **not** treated as suspicious
- Each forensic test runs **at most once**
- Conclusions are drawn **only from available evidence**

⚠️ This tool provides expert-style forensic assessment, not legal proof.
"""
)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image for forensic analysis",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to begin the forensic audit.")
    st.stop()

# ============================================================
# DISPLAY IMAGE (UI ONLY)
# ============================================================

image = Image.open(uploaded_file)
st.image(image, caption="Submitted Evidence", use_container_width=True)

# ============================================================
# PRESERVE ORIGINAL BYTES (CRITICAL)
# ============================================================

original_ext = os.path.splitext(uploaded_file.name)[1].lower()
if original_ext not in [".jpg", ".jpeg", ".png"]:
    original_ext = ".jpg"

temp_path = f"evidence_temp{original_ext}"

with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# ============================================================
# RUN FORENSIC AUDIT
# ============================================================

if st.button("🔍 Run Forensic Audit", type="primary"):
    with st.spinner("Conducting forensic examination..."):
        graph = create_forensic_graph()

        # ----------------------------------------------------
        # INITIAL STATE (ALL REQUIRED KEYS)
        # ----------------------------------------------------

        initial_state = {
            "image_path": temp_path,
            "messages": [],

            "visual_description": "",
            "metadata": {},
            "ela": {},
            "osint": {},

            "conclusions": {},
            "skipped_exams": {},
            "planned_run": [],

            "confidence_label": "",
            "confidence_score": 0.0,
            "confidence_reasoning": "",
            "final_report": ""
        }

        result = graph.invoke(initial_state)

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    st.markdown("## 🧾 Executive Summary")

    label = result.get("confidence_label", "Inconclusive")
    score = int(result.get("confidence_score", 0.0) * 100)

    if label == "Inconclusive":
        st.warning(
            "**Assessment Inconclusive** — insufficient or conflicting forensic evidence."
        )
    else:
        st.success(
            f"**Confidence Level:** {label}  \n"
            f"**Confidence Score:** {score}%"
        )

    st.markdown("**Rationale:**")
    st.markdown(result.get("confidence_reasoning", "No reasoning provided."))

    # ========================================================
    # EXAMINATION CONCLUSIONS
    # ========================================================

    st.markdown("---")
    st.markdown("## 🔬 Examination Conclusions")

    conclusions = result.get("conclusions", {})
    if conclusions:
        for exam, conclusion in conclusions.items():
            st.markdown(f"### {exam.replace('_', ' ').title()}")
            st.markdown(conclusion)
    else:
        st.info("No examinations produced conclusions.")

    # ========================================================
    # SKIPPED EXAMS
    # ========================================================

    skipped = result.get("skipped_exams", {})
    if skipped:
        st.markdown("---")
        st.markdown("## ⏭️ Skipped Examinations")

        for exam, reason in skipped.items():
            st.markdown(
                f"- **{exam.replace('_', ' ').title()}** — {reason}"
            )

    # ========================================================
    # FULL REPORT
    # ========================================================

    st.markdown("---")
    st.markdown("## 📄 Full Forensic Report")
    st.markdown(result.get("final_report", "No report generated."))

    # ========================================================
    # CLEANUP
    # ========================================================

    if os.path.exists(temp_path):
        os.remove(temp_path)
