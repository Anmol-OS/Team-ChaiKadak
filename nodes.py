import os
import json
import cv2
import numpy as np
import exifread
import requests
import torch
import re
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from PIL import Image


from transformers import BlipProcessor, BlipForConditionalGeneration

from sklearn.decomposition import PCA

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

load_dotenv()

class ForensicState(TypedDict):
    image_path: str
    image_url : str
    messages: List[BaseMessage]
    visual_description: str
    metadata: Dict[str, Any]
    ela: Dict[str, Any]
    osint: Dict[str, Any]
    noise_analysis: Dict[str, Any]
    strings_audit: Dict[str, Any]
    pca_analysis: Dict[str, Any]
    luminance_check: Dict[str, Any]
    clone_detection: Dict[str, Any]
    conclusions: Dict[str, str]
    skipped_exams: Dict[str, str]
    planned_run: List[str]
    confidence_label: str
    confidence_score: float
    confidence_reasoning: str
    final_report: str


def safe_parse_json(raw: str) -> Dict[str, Any] | None:
    """Robustly extracts and parses JSON from a string."""
    if not raw or not raw.strip():
        return None
    start_index = raw.find('{')
    end_index = raw.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        raw = raw[start_index : end_index + 1]
    else:
        raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def upload_to_tmpfiles(image_path: str, state : ForensicState) -> str | None:
    """Uploads an image to TmpFiles.org to get a public URL."""
    url = "https://tmpfiles.org/api/v1/upload"
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
        if response.status_code == 200:
            data = response.json()
            raw_url = data.get('data', {}).get('url')
            if raw_url:
                state["image_url"] = raw_url
                return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        return None
    except Exception:
        return None


@tool
def extract_metadata(image_path: str) -> str:
    """Extracts available EXIF and GPS metadata from an image."""
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        def gps_to_decimal(values, ref):
            d = values[0].num / values[0].den
            m = values[1].num / values[1].den
            s = values[2].num / values[2].den
            val = d + m / 60 + s / 3600
            return -val if ref in ["S", "W"] else val

        lat = lon = None
        if "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
            lat = gps_to_decimal(tags["GPS GPSLatitude"].values, tags["GPS GPSLatitudeRef"].printable)
        if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
            lon = gps_to_decimal(tags["GPS GPSLongitude"].values, tags["GPS GPSLongitudeRef"].printable)

        gps = {"lat": lat, "lon": lon} if lat is not None and lon is not None else None

        return json.dumps({
            "device": (f"{tags.get('Image Make','')} {tags.get('Image Model','')}").strip() or None,
            "software": str(tags.get("Image Software")) if "Image Software" in tags else None,
            "timestamp": str(tags.get("EXIF DateTimeOriginal")) if "EXIF DateTimeOriginal" in tags else None,
            "gps": gps
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_ela_analysis(image_path: str) -> str:
    """Performs Error Level Analysis (ELA) and returns deviation statistics."""
    pid = os.getpid()
    tmp = f"ela_{pid}.jpg"
    try:
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Invalid image")

        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recompressed = cv2.imread(tmp)

        diff = cv2.absdiff(img, recompressed)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))

        return json.dumps({
            "std_deviation": std,
            "risk": "HIGH" if std > 10 else "LOW"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if os.path.exists(tmp): os.remove(tmp)


@tool
def run_noise_analysis(image_path: str) -> str:
    """Performs Noise Residue Analysis."""
    try:
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Invalid image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        noise_map = cv2.absdiff(gray, denoised)
        
        noise_std = float(np.std(noise_map))
        inference = "Normal"
        if noise_std < 1.0: inference = "Suspiciously Smooth (Possible Synthetic)"
        elif noise_std > 12.0: inference = "High Noise (High ISO or Added Grain)"

        return json.dumps({
            "noise_mean": float(np.mean(noise_map)),
            "noise_std_deviation": noise_std,
            "automated_inference": inference
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def extract_strings_tool(image_path: str) -> str:
    """Finds printable text hidden in the binary file."""
    try:
        with open(image_path, "rb") as f:
            content = f.read()
            matches = re.findall(b"[ -~]{4,}", content)
            found_strings = [m.decode("utf-8", errors="ignore") for m in matches]
            
        full_text = "\n".join([s for s in found_strings if len(s) > 4])
        if len(full_text) > 2000: full_text = full_text[:2000] + "\n...[Truncated]"

        return json.dumps({"raw_strings": full_text})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_pca_analysis(image_path: str) -> str:
    """Performs PCA-based anomaly detection."""
    try:
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Invalid image")
        
        h, w = img.shape[:2]
        if w > 1000:
            scale = 1000 / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
        flat_img = img.reshape(-1, 3).astype(np.float32) / 255.0
        pca = PCA(n_components=1) 
        projected = pca.fit_transform(flat_img)
        reconstructed = pca.inverse_transform(projected)
        
        diff = flat_img - reconstructed
        error_map = np.sqrt(np.sum(diff**2, axis=1)).reshape(img.shape[:2])
        
        return json.dumps({
            "mean_reconstruction_error": float(np.mean(error_map)),
            "max_reconstruction_error": float(np.max(error_map)),
            "explained_variance": float(pca.explained_variance_ratio_[0])
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_luminance_analysis(image_path: str) -> str:
    """Analyzes Luminance Gradient."""
    try:
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Invalid image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        hist_angles = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)[0]
        dominant_idx = np.argmax(hist_angles)
        
        return json.dumps({
            "average_gradient_magnitude": float(np.mean(magnitude)),
            "sharpness_variance": float(np.std(magnitude)),
            "dominant_light_angle_approx": f"{dominant_idx * 45}° - {(dominant_idx * 45)+45}°",
            "lighting_consistency": "Directional" if np.max(hist_angles) > 2 * np.mean(hist_angles) else "Diffuse/Flat"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_clone_detection(image_path: str) -> str:
    """Detects Copy-Move Forgery using ORB."""
    try:
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Invalid image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=2000)
        kp, des = orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 2:
             return json.dumps({"status": "clean", "details": "Not enough features."})

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=2)

        suspicious_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                pt1 = np.array(kp[m.queryIdx].pt)
                pt2 = np.array(kp[m.trainIdx].pt)
                if np.linalg.norm(pt1 - pt2) > 50:
                    suspicious_matches.append(float(np.linalg.norm(pt1 - pt2)))

        count = len(suspicious_matches)
        return json.dumps({
            "suspicious_match_count": count,
            "inference": "High Probability of Cloning" if count > 15 else "Clean"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def google_lens_search(image_path: str) -> str:
    """Performs reverse image search."""
    key = os.getenv("SERPAPI_KEY")
    if not key: return json.dumps({"skipped": "SERPAPI_KEY missing"})
    public_url = upload_to_tmpfiles(image_path)
    if not public_url: return json.dumps({"skipped": "Upload failed"})

    try:
        params = {"engine": "google_lens", "type": "exact_matches", "api_key": key, "url": public_url}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        data = r.json()
        return json.dumps({
            "visual_matches": data.get("visual_matches", [])[:5],
            "pages": data.get("pages_with_matching_images", [])[:5],
        })
    except Exception as e:
        return json.dumps({"error": str(e)})



class ForensicNodes:
    def __init__(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )
        self.vis_model_id = "Salesforce/blip-image-captioning-base"
        try:
            self.vis_processor = BlipProcessor.from_pretrained(self.vis_model_id)
            self.vis_model = BlipForConditionalGeneration.from_pretrained(self.vis_model_id)
            print("✅ Vision Model Loaded.")
        except:
            self.vis_model = None

    def _get_previous_findings(self, state: ForensicState) -> str:
        if not state.get("conclusions"): return ""
        history = "PREVIOUS EXAM FINDINGS (For Context Only):\n"
        for exam, result in state["conclusions"].items():
            history += f"[{exam.upper()}]: {result}\n"
        return history

    def metadata_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        raw = extract_metadata.invoke({"image_path": state["image_path"]})
        state["metadata"] = safe_parse_json(raw) or {}
        history = self._get_previous_findings(state)

        prompt = f"""
        You are a forensic expert.
        {history}
        CURRENT DATA (Metadata): {json.dumps(state["metadata"], indent=2)}
        
        INSTRUCTION: 
        Analyze the metadata. Use previous findings only to verify consistency.
        OUTPUT REQUIREMENT: 
        Provide ONLY the final forensic conclusion about the metadata. 
        Do not list cross-checks. Do not mention "Task 1" or "Task 2".
        If there are no inconsistencies, state the finding simply.
        """
        state["conclusions"]["metadata"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        return state

    def planner_node(self, state: ForensicState):
        state.setdefault("planned_run", [])
        state.setdefault("skipped_exams", {})
        state.setdefault("conclusions", {})
        
        prompt = f"""
        Metadata conclusion: {state["conclusions"].get("metadata", "")}
        Decide exams to run. Return JSON ONLY: {{ "run": ["ela", "osint", ...], "skip": {{}} }}
        Allowed: visual_environment, ela, osint, noise_analysis, strings_audit, pca_analysis, luminance_check, clone_detection.
        """
        raw = self.llm.invoke([HumanMessage(content=prompt)]).content
        decision = safe_parse_json(raw) or {}
        
        run = decision.get("run", [])
        if not run: run = ["visual_environment", "ela", "noise_analysis", "osint"]
        state["planned_run"] = list(set(run))
        return state

    def visual_environment_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        if not self.vis_model: return state

        try:
            image = Image.open(state["image_path"]).convert("RGB")
            inputs = self.vis_processor(image, return_tensors="pt")
            out = self.vis_model.generate(**inputs, max_new_tokens=50)
            state["visual_description"] = self.vis_processor.decode(out[0], skip_special_tokens=True).strip()

            history = self._get_previous_findings(state)
            blip_caption = state["visual_description"]
            metadata_str = json.dumps(state.get("metadata", {}))
            final_description = f"Initial BLIP Caption: {blip_caption}"
            try:
                if "image_url" not in state:
                    public_url = upload_to_tmpfiles(state["image_path"], state)
                else:
                    public_url = state["image_url"]
                
                prompt = f"""
                    You are a Visual Forensic Analyst.
                    
                    INPUTS:
                    1. IMAGE: (See attached URL)
                    2. LOCAL BLIP CAPTION: "{blip_caption}"
                    3. METADATA: {metadata_str}
                    4. {history}

                    TASK:
                    1. Verify the BLIP caption.
                    2. Analyze lighting, shadows, and reflections for consistency.
                    3. Check if the scene matches the metadata (time/location).
                    
                    OUTPUT FORMAT:
                    1. **Scene Description**: Detailed visual breakdown.
                    2. **Visual Consistency**: Analysis of physics/lighting.
                    3. **Final Verdict**: Authentic or Suspicious?
                    """
                    
                message = HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": public_url}},
                        {"type": "text", "text": prompt},
                    ]
                )

                response = self.llm.invoke([message])
                final_description += f"\n[Deep Forensic Analysis]:\n{response.content}"
                    
                state["visual_description"] = final_description
                state["conclusions"]["visual_environment"] = response.content

            except Exception as vision_err:
                print(f"Vision API failed: {vision_err}. Falling back to Text-Only analysis.")
                
                prompt = f"""
                You are a Visual Forensic Analyst.
                
                Vision API Failed. RELYING ON LOCAL CAPTION: "{blip_caption}"
                METADATA: {json.dumps(state.get("metadata", {}))}
                
                TASK:
                Does this caption logically align with the metadata?
                
                OUTPUT:
                1. **Analysis**: Based on caption "{blip_caption}".
                2. **Final Verdict**: Inconclusive (Vision API Error) or Consistent/Inconsistent.
                """
                fallback_response = self.llm.invoke([HumanMessage(content=prompt)]).content
                state["conclusions"]["visual_environment"] = f"⚠️ [Vision API Failed - Using Fallback]\n{fallback_response}"
        
        except Exception as e:
            error_msg = f"Critical Vision Failure: {str(e)}"
            state["skipped_exams"]["visual_environment"] = error_msg
            state["conclusions"]["visual_environment"] = f"❌ Analysis Failed: {error_msg}"

        return state

    def ela_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = run_ela_analysis.invoke({"image_path": state["image_path"]})
            state["ela"] = safe_parse_json(raw) or {}
            history = self._get_previous_findings(state)
            
            prompt = f"""
            You are a Digital Artifact Analyst.
            {history}
            CURRENT DATA (ELA): {json.dumps(state['ela'], indent=2)}
            
            INSTRUCTION:
            Interpret the ELA deviation.
            Silently cross-check with history.
            OUTPUT: Provide ONLY the final conclusion. No verbose reasoning steps.
            """
            state["conclusions"]["ela"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["ela"] = str(e)
        return state

    def noise_analysis_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = run_noise_analysis.invoke({"image_path": state["image_path"]})
            state["noise_analysis"] = safe_parse_json(raw) or {}
            history = self._get_previous_findings(state)
            
            prompt = f"""
            You are a Noise Analyst.
            {history}
            CURRENT DATA (Noise): {json.dumps(state['noise_analysis'], indent=2)}
            
            INSTRUCTION:
            Analyze noise levels.
            OUTPUT: Write ONLY the final forensic conclusion regarding image naturalness.
            """
            state["conclusions"]["noise_analysis"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["noise_analysis"] = str(e)
        return state

    def strings_audit_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = extract_strings_tool.invoke({"image_path": state["image_path"]})
            data = safe_parse_json(raw) or {}
            state["strings_audit"] = data
            history = self._get_previous_findings(state)
            
            prompt = f"""
            You are a Data Analyst.
            {history}
            CURRENT DATA (Strings): "{data.get("raw_strings", "")}"

            INSTRUCTION:
            Identify hidden tags. 
            OUTPUT: Write ONLY the findings (e.g., "Found Photoshop tag" or "No suspicious strings").
            """
            state["conclusions"]["strings_audit"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["strings_audit"] = str(e)
        return state

    def pca_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = run_pca_analysis.invoke({"image_path": state["image_path"]})
            data = safe_parse_json(raw) or {}
            state["pca_analysis"] = data
            history = self._get_previous_findings(state)
            
            prompt = f"""
            You are a Geometry Analyst.
            {history}
            CURRENT DATA (PCA): {json.dumps(data, indent=2)}

            INSTRUCTION:
            Analyze reconstruction error.
            OUTPUT: Write ONLY the final conclusion about color consistency.
            """
            state["conclusions"]["pca_analysis"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["pca_analysis"] = str(e)
        return state

    def luminance_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = run_luminance_analysis.invoke({"image_path": state["image_path"]})
            data = safe_parse_json(raw) or {}
            state["luminance_check"] = data
            history = self._get_previous_findings(state)
            
            prompt = f"""
            You are a Lighting Analyst.
            {history}
            CURRENT DATA (Luminance): {json.dumps(data, indent=2)}
            
            INSTRUCTION:
            Check lighting consistency.
            OUTPUT: Write ONLY the final observation.
            """
            state["conclusions"]["luminance_check"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["luminance_check"] = str(e)
        return state

    def clone_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = run_clone_detection.invoke({"image_path": state["image_path"]})
            data = safe_parse_json(raw) or {}
            state["clone_detection"] = data
            history = self._get_previous_findings(state)
            
            prompt = f"""
            You are a Pattern Analyst.
            {history}
            CURRENT DATA (Clone Detection): {json.dumps(data, indent=2)}
            
            INSTRUCTION:
            Check for copy-move forgery.
            OUTPUT: Write ONLY the final conclusion regarding duplication.
            """
            state["conclusions"]["clone_detection"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["clone_detection"] = str(e)
        return state

    def osint_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = google_lens_search.invoke({"image_path": state["image_path"]}, state=state)
            state["osint"] = safe_parse_json(raw) or {}
            history = self._get_previous_findings(state)

            prompt = f"""
            You are a Digital Investigator.
            {history}
            CURRENT DATA (Reverse Search): {json.dumps(state["osint"], indent=2)}

            INSTRUCTION:
            Check for online duplicates.
            OUTPUT: Write ONLY the final conclusion regarding online presence.
            """
            state["conclusions"]["osint"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["osint"] = str(e)
        return state

    def confidence_node(self, state: ForensicState):
        conclusions = state.get("conclusions", {})
        if not conclusions:
            state["confidence_label"] = "Inconclusive"
            state["confidence_score"] = 0.0
            return state

        prompt = f"""
        Conclusions: {json.dumps(conclusions, indent=2)}
        TASK: Determine confidence score (0-100) and label.
        OUTPUT JSON ONLY: {{ "confidence_label": "High", "confidence_score": 95, "reasoning": "..." }}
        """
        raw = self.llm.invoke([HumanMessage(content=prompt)]).content
        result = safe_parse_json(raw) or {}

        state["confidence_label"] = result.get("confidence_label", "Low")
        state["confidence_score"] = float(result.get("confidence_score", 0)) / 100.0
        state["confidence_reasoning"] = result.get("reasoning", "")
        return state

    def report_node(self, state: ForensicState):
        conclusions = state.get("conclusions", {})
        prompt = f"Summarize into a final forensic report (4-5 sentences):\n{json.dumps(conclusions, indent=2)}"
        state["final_report"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        return state