# 🕵️‍♂️ Forensic Image Analyzer | CodeFusion Hackathon

**Developed by Team ChaiKadak**  
_For the CodeFusion Hackathon (hosted by Google Developers Group on Campus JIIT-128)_

---

## 🍵 Team ChaiKadak

- **Anmol Bhatnagar**  
  Enrollment: 992401040039  
- **Dhruv Arora**  
  Enrollment: 992401040023  

---

## 🚀 Project Overview

In a landscape of increasingly sophisticated digital misinformation, traditional metadata checks alone are no longer sufficient to verify image authenticity.

The **Forensic Image Analyzer** is an **agentic forensic system** designed to verify the authenticity of digital images. Built using **LangGraph**, the system orchestrates specialized forensic agents through a **dynamic state machine**, producing:

- A comprehensive forensic audit trail  
- A logically reasoned expert conclusion  
- A mathematically backed **confidence score** for every investigation  

---

## 🧠 Adaptive Intelligence: The Planner Agent

The core innovation of this project lies in its **Adaptive Orchestration**.

Instead of running every forensic test on every image—which is computationally expensive and often redundant—the system employs an intelligent **Planner Node** that acts as the **Forensic Lead**.

### 🔍 Dynamic Tool Selection

The Planner Agent analyzes early findings (especially metadata conclusions) and dynamically decides:

#### ⚡ Efficiency
- If metadata reveals a definitive indicator of manipulation (e.g., editing software signatures), the system may **skip expensive OSINT searches** to reduce computation and latency.

#### 🧩 Contextual Necessity
- The agent evaluates whether a specific test (such as ELA) adds **marginal value**.
- If the visual environment is already inconsistent, the system prioritizes **evidence gathering** over redundant analysis.

#### 🚀 Performance Optimization
- By pruning unnecessary investigation paths, the system:
  - Reduces API latency
  - Saves token usage
  - Maintains forensic integrity

---

## 🛠 Forensic Examination Modules

The investigation pipeline consists of specialized **Nodes** that can be dynamically triggered or bypassed based on the Planner’s logic:

### 📄 Metadata Audit
- Extracts embedded **EXIF**, **GPS**, and **hardware tags**
- Identifies:
  - Device mismatches
  - Software editing traces (e.g., Adobe Photoshop, Lightroom)

### 🧪 Error Level Analysis (ELA)
- Re-saves the image at a known compression rate
- Generates an ELA map to detect:
  - Localized compression anomalies
  - Possible retouching or object insertion

### 👁️ Visual Environment Analysis
- Uses a **Vision-Language Model**
- Detects logical inconsistencies between:
  - Lighting
  - Shadows
  - Objects
  - Metadata context

### 🌐 OSINT Integration
- Integrates **Google Lens via SerpAPI**
- Performs reverse image searches to identify:
  - Prior publication
  - Known sources
  - Reused or misattributed imagery

---

## 🔢 Confidence Scoring & Logic

The system does not output a simple *“real or fake”* verdict.

Instead, it produces a **Confidence Score** using a transparent, two-step normalization process:

### 1️⃣ Raw Evaluation
- An expert agent scores the findings from all executed forensic nodes.

### 2️⃣ Coverage Penalty
- To prevent partial investigations from appearing overly confident, the final score is adjusted based on investigation coverage:

\[
\textbf{Final Score} = \textbf{Raw LLM Score} \times \left( \frac{\textbf{Executed Nodes}}{\textbf{Total Planned Nodes}} \right)
\]

This ensures:
- Transparency
- Fair confidence estimation
- Clear auditability

---

## 💻 Tech Stack

- **Orchestration:** LangGraph (StateGraph)  
- **Intelligence:** ChatGroq (Llama 4 Scout) for rapid, expert-level reasoning  
- **Vision Engine:** OpenCV (CV2) & NumPy for statistical Error Level Analysis  
- **Frontend:** Streamlit with custom Flexbox CSS  
  - Responsive dashboard  
  - Perfectly aligned metrics  
  - Wrapped expert reporting for maximum readability  

---

## 🏆 Hackathon

**CodeFusion Hackathon**  
Hosted by **GDG on Campus JIIT-128**  

**Team:** ChaiKadak 🍵  
