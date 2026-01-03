import os
import json
import cv2
import numpy as np
import exifread
import requests
import torch
import re
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

load_dotenv()


def safe_parse_json(raw: str) -> Dict[str, Any] | None:
    """
    Robustly extracts and parses JSON from a string, handling 
    markdown code blocks and conversational filler.
    """
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


def upload_to_tmpfiles(image_path: str) -> str | None:
    """
    Uploads an image to TmpFiles.org (temporary hosting) to get a public URL.
    Returns the direct download URL required for APIs.
    """
    url = "https://tmpfiles.org/api/v1/upload"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            raw_url = data.get('data', {}).get('url')
            
            if raw_url:
                return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        return None
    except Exception as e:
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
            lat = gps_to_decimal(
                tags["GPS GPSLatitude"].values,
                tags["GPS GPSLatitudeRef"].printable
            )

        if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
            lon = gps_to_decimal(
                tags["GPS GPSLongitude"].values,
                tags["GPS GPSLongitudeRef"].printable
            )

        gps = None
        if lat is not None and lon is not None:
            gps = {"lat": lat, "lon": lon}

        return json.dumps({
            "device": (
                f"{tags.get('Image Make','')} {tags.get('Image Model','')}"
                .strip() or None
            ),
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
        if img is None:
            raise ValueError("Invalid image")

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
        if os.path.exists(tmp):
            os.remove(tmp)


@tool
def google_lens_search(image_path: str) -> str:
    """Performs reverse image search using SerpAPI (Google Lens)."""
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return json.dumps({"skipped": "SERPAPI_KEY missing"})

    public_url = upload_to_tmpfiles(image_path)
    
    if not public_url:
        return json.dumps({
            "skipped": "Temporary image hosting failed", 
            "details": "Could not upload to TmpFiles. Check internet connection."
        })

    try:
        params = {
            "engine": "google_lens",
            "api_key": key,
            "url": public_url
        }
        
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)

        if r.status_code != 200:
            return json.dumps({
                "error": f"SerpApi Failed: {r.status_code}", 
                "details": r.text[:200]
            })

        data = r.json()
        return json.dumps({
            "visual_matches": data.get("visual_matches", [])[:5],
            "pages": data.get("pages_with_matching_images", [])[:5],
            "knowledge_graph": data.get("knowledge_graph", [])
        })

    except Exception as e:
        return json.dumps({"error": f"Search execution failed: {str(e)}"})


class ForensicState(TypedDict):
    image_path: str
    messages: List[BaseMessage]
    visual_description: str
    metadata: Dict[str, Any]
    ela: Dict[str, Any]
    osint: Dict[str, Any]
    conclusions: Dict[str, str]
    skipped_exams: Dict[str, str]
    planned_run: List[str]
    confidence_label: str
    confidence_score: float
    confidence_reasoning: str
    final_report: str


class ForensicNodes:
    def __init__(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )

        self.vis_model_id = "Salesforce/blip-image-captioning-base"
        
        print(f"⏳ Loading Vision Model: {self.vis_model_id}...")
        
        try:
            self.vis_processor = BlipProcessor.from_pretrained(self.vis_model_id)
            self.vis_model = BlipForConditionalGeneration.from_pretrained(self.vis_model_id)
            print("✅ Vision Model Loaded Successfully.")
        except Exception as e:
            print(f"❌ Failed to load Vision Model: {e}")
            self.vis_model = None
            self.vis_processor = None

    def metadata_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        state.setdefault("skipped_exams", {})

        raw = extract_metadata.invoke({"image_path": state["image_path"]})
        state["metadata"] = safe_parse_json(raw) or {}

        prompt = f"""
You are a forensic image examiner.
Metadata: {json.dumps(state["metadata"], indent=2)}
Rules: Analyze only present fields. Missing data is NOT suspicious.
Write a concise conclusion.
"""
        state["conclusions"]["metadata"] = self.llm.invoke(
            [HumanMessage(content=prompt)]
        ).content
        return state

    def planner_node(self, state: ForensicState):
        state.setdefault("planned_run", [])
        state.setdefault("skipped_exams", {})
        state.setdefault("conclusions", {})

        metadata_conclusion = state["conclusions"].get("metadata", "")
        prompt = f"""
    You are a forensic examination controller.
    Metadata conclusion: {metadata_conclusion}
    Decide NECESSARY exams (visual_environment, ela, osint).
    Return JSON ONLY: {{ "run": [...], "skip": {{...}} }}
    """
        raw = self.llm.invoke([HumanMessage(content=prompt)]).content
        decision = safe_parse_json(raw)

        if not decision or not isinstance(decision, dict):
            state["planned_run"] = ["visual_environment", "ela", "osint"]
            return state

        run = decision.get("run", [])
        skip = decision.get("skip", {})
        if not isinstance(run, list): run = []
        if not isinstance(skip, dict): skip = {}

        allowed = {"visual_environment", "ela", "osint"}
        run = [x for x in run if x in allowed]
        
        clean_skip = {k: str(v) for k, v in skip.items() if k in allowed}
        
        if not run: run = ["visual_environment", "osint"]
        
        state["planned_run"] = list(set(run))
        state["skipped_exams"].update(clean_skip)
        return state

    def visual_environment_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        state.setdefault("skipped_exams", {})

        if not self.vis_model:
            state["skipped_exams"]["visual_environment"] = "Vision Model failed to load (Hardware/VRAM issue)."
            return state

        try:
            image = Image.open(state["image_path"]).convert("RGB")

            inputs = self.vis_processor(image, return_tensors="pt")

            out = self.vis_model.generate(**inputs, max_new_tokens=50)

            description = self.vis_processor.decode(out[0], skip_special_tokens=True)
            
            state["visual_description"] = description.strip()
            prompt = f"""
Visual description (Generated by Salesforce/blip-image-captioning-base):
{state["visual_description"]}

Metadata:
{json.dumps(state["metadata"], indent=2)}

Does the visual content logically match the metadata? (e.g. time of day, location type).
"""
            state["conclusions"]["visual_environment"] = self.llm.invoke(
                [HumanMessage(content=prompt)]
            ).content

        except Exception as e:
            state["skipped_exams"]["visual_environment"] = f"BLIP generation failed: {e}"

        return state

    def ela_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        try:
            raw = run_ela_analysis.invoke({"image_path": state["image_path"]})
            state["ela"] = safe_parse_json(raw) or {}
            prompt = f"ELA Results: {json.dumps(state['ela'], indent=2)}\nInterpret for tampering."
            state["conclusions"]["ela"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            state["skipped_exams"]["ela"] = str(e)
        return state

    def osint_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        ground_truth = state.get("visual_description", "No visual description available.")

        try:
            raw = google_lens_search.invoke({"image_path": state["image_path"]})
            state["osint"] = safe_parse_json(raw) or {}

            prompt = f"""
    You are a skeptical forensic investigator analyzing Reverse Image Search results.
    
    1. THE EVIDENCE (Uploaded Image Content):
    "{ground_truth}"

    2. THE SEARCH RESULTS:
    {json.dumps(state["osint"], indent=2)}

    CRITICAL INSTRUCTION:
    - Google Lens returns visual matches (style/color) that are NOT the same image.
    - Check 'pages_with_matching_images' for exact duplicates.
    - Compare visual_matches strictly against 'THE EVIDENCE'. If subjects differ, dismiss them.
    
    Write a conclusion distinguishing exact matches from false positives.
    """
            state["conclusions"]["osint"] = self.llm.invoke(
                [HumanMessage(content=prompt)]
            ).content
        except Exception as e:
            state["skipped_exams"]["osint"] = str(e)
        return state

    def confidence_node(self, state: ForensicState):
        conclusions = state.get("conclusions", {})
        executed = list(conclusions.keys())

        if not executed:
            state["confidence_label"] = "Inconclusive"
            state["confidence_score"] = 0.0
            return state
        
        prompt = f"""
    Exams run: {executed}
    Conclusions: {json.dumps(conclusions, indent=2)}
    
    TASK:
    Evaluate the consistency of the findings to determine a confidence score (0-100) and security label.
    
    OUTPUT FORMAT (STRICT JSON ONLY, NO TEXT BEFORE OR AFTER):
    {{
        "confidence_label": "High",  
        "confidence_score": 95,
        "reasoning": "Metadata matches visual content and OSINT confirms location."
    }}
    """
        raw = self.llm.invoke([HumanMessage(content=prompt)]).content
        result = safe_parse_json(raw)

        if result:
            state["confidence_label"] = result.get("confidence_label", "Low")
            try:
                score = float(result.get("confidence_score", 0))
                state["confidence_score"] = score / 100.0 if score > 1.0 else score
            except (ValueError, TypeError):
                 state["confidence_score"] = 0.0
            
            state["confidence_reasoning"] = result.get("reasoning", "")
        else:
            state["confidence_label"] = "Error"
            state["confidence_score"] = 0.0
            state["confidence_reasoning"] = "System failed to parse confidence score from LLM."
        return state

    def report_node(self, state: ForensicState):
        conclusions = state.get("conclusions", {})
        prompt = f"Summarize findings into final report (4-5 sentences):\n{json.dumps(conclusions, indent=2)}"
        state["final_report"] = self.llm.invoke([HumanMessage(content=prompt)]).content
        return state