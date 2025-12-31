import os
import json
import cv2
import numpy as np
import exifread
import requests
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from PIL import Image

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

load_dotenv()


def safe_parse_json(raw: str) -> Dict[str, Any] | None:
    if not raw or not raw.strip():
        return None

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


@tool
def extract_metadata(image_path: str) -> str:
    """
    Extracts available EXIF and GPS metadata from an image.

    Returns JSON string with:
    - device
    - software
    - timestamp
    - gps (lat, lon) if available
    """
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
    """
    Performs Error Level Analysis (ELA) and returns deviation statistics.
    """
    tmp = f"ela_{os.getpid()}.jpg"
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
    """
    Performs reverse image search using SerpAPI (Google Lens).
    """
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return json.dumps({"skipped": "SERPAPI_KEY missing"})

    try:
        with open(image_path, "rb") as f:
            r = requests.post(
                "https://serpapi.com/search",
                params={"engine": "google_lens", "api_key": key},
                files={"encoded_image": f},
                timeout=30
            )

        data = r.json()
        return json.dumps({
            "visual_matches": data.get("visual_matches", [])[:5],
            "pages": data.get("pages_with_matching_images", [])[:5]
        })

    except Exception as e:
        return json.dumps({"error": str(e)})



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

        vision_pipe = pipeline(
            task="image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device_map="auto"
        )

        self.vis_model = vision_pipe

    def metadata_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        state.setdefault("skipped_exams", {})

        raw = extract_metadata.invoke({"image_path": state["image_path"]})
        state["metadata"] = safe_parse_json(raw) or {}

        prompt = f"""
You are a forensic image examiner.

Metadata:
{json.dumps(state["metadata"], indent=2)}

Rules:
- Analyze only present fields
- Missing data is NOT suspicious

Write a concise conclusion.
"""
        state["conclusions"]["metadata"] = self.llm.invoke(
            [HumanMessage(content=prompt)]
        ).content

        return state

    def planner_node(self, state: ForensicState):
        """
        Decide which forensic examinations should be run.

        Planner rules:
        - Planner NEVER claims execution
        - Planner only decides necessity
        - Output must be audit-safe and deterministic
        """

        state.setdefault("planned_run", [])
        state.setdefault("skipped_exams", {})
        state.setdefault("conclusions", {})

        metadata_conclusion = state["conclusions"].get("metadata", "")

        prompt = f"""
    You are a forensic examination controller.

    Available examinations:
    - visual_environment
    - ela
    - osint

    Metadata conclusion:
    {metadata_conclusion}

    Rules:
    - Decide which examinations are NECESSARY
    - You may skip an exam ONLY if it adds no value
    - DO NOT claim any exam was already performed
    - DO NOT speculate
    - Skip reasons MUST start with: "Not required because ..."

    Return JSON ONLY in this format:
    {{
    "run": ["visual_environment", "ela"],
    "skip": {{
        "osint": "Not required because metadata and visual evidence are sufficient"
    }}
    }}
    """

        raw = self.llm.invoke([HumanMessage(content=prompt)],response_format={"type": "json_object"}).content
        decision = safe_parse_json(raw)

        if not decision or not isinstance(decision, dict):
            state["planned_run"] = ["visual_environment", "ela"]
            state["skipped_exams"]["osint"] = (
                "Not required because automated OSINT was unavailable"
            )
            return state


        run = decision.get("run", [])
        skip = decision.get("skip", {})

        if not isinstance(run, list):
            run = []

        if not isinstance(skip, dict):
            skip = {}

        allowed_exams = {"visual_environment", "ela", "osint"}
        run = [exam for exam in run if exam in allowed_exams]

        seen = set()
        run = [x for x in run if not (x in seen or seen.add(x))]

        clean_skip = {}

        for exam, reason in skip.items():
            if exam not in allowed_exams:
                continue

            if not isinstance(reason, str) or not reason.strip():
                clean_skip[exam] = "Not required because available evidence is sufficient"
                continue

            lowered = reason.lower()

            if any(word in lowered for word in ["already", "performed", "executed", "done"]):
                clean_skip[exam] = "Not required because available evidence is sufficient"
            elif not lowered.startswith("not required because"):
                clean_skip[exam] = "Not required because available evidence is sufficient"
            else:
                clean_skip[exam] = reason.strip()
        if not run:
            run = ["visual_environment"]
            state["skipped_exams"].setdefault(
                "planner",
                "Not required because metadata alone was insufficient"
            )

        state["planned_run"] = run
        state["skipped_exams"].update(clean_skip)

        return state


    def visual_environment_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        state.setdefault("skipped_exams", {})

        try:
            image = Image.open(state["image_path"]).convert("RGB")
            result = self.vis_model(image)[0]
            state["visual_description"] = result.get("generated_text", "").strip()

            prompt = f"""
Visual description:
{state["visual_description"]}

Metadata:
{json.dumps(state["metadata"], indent=2)}

Highlight only concrete inconsistencies.
"""
            state["conclusions"]["visual_environment"] = self.llm.invoke(
                [HumanMessage(content=prompt)]
            ).content

        except Exception as e:
            state["skipped_exams"]["visual_environment"] = f"Vision model failed: {e}"

        finally:
            if "visual_environment" in state["planned_run"]:
                state["planned_run"].remove("visual_environment")

        return state


    def ela_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        state.setdefault("skipped_exams", {})

        try:
            raw = run_ela_analysis.invoke({"image_path": state["image_path"]})
            state["ela"] = safe_parse_json(raw) or {}

            prompt = f"""
You are a forensic image examiner.

ELA (Error Level Analysis) measures compression inconsistencies.
It does NOT prove authenticity or manipulation by itself.

ELA results:
{json.dumps(state["ela"], indent=2)}

Rules:
- Do NOT explain statistics
- Do NOT discuss unrelated meanings of ELA
- Interpret ONLY in the context of image tampering
- LOW risk means no strong ELA-based evidence of manipulation

Write a 2–3 sentence forensic conclusion.


"""
            state["conclusions"]["ela"] = self.llm.invoke(
                [HumanMessage(content=prompt)]
            ).content

        except Exception as e:
            state["skipped_exams"]["ela"] = str(e)

        finally:
            if "ela" in state["planned_run"]:
                state["planned_run"].remove("ela")

        return state


    def osint_node(self, state: ForensicState):
        state.setdefault("conclusions", {})
        state.setdefault("skipped_exams", {})

        try:
            raw = google_lens_search.invoke({"image_path": state["image_path"]})
            state["osint"] = safe_parse_json(raw) or {}

            prompt = f"""
OSINT data:
{json.dumps(state["osint"], indent=2)}
"""
            state["conclusions"]["osint"] = self.llm.invoke(
                [HumanMessage(content=prompt)]
            ).content

        except Exception as e:
            state["skipped_exams"]["osint"] = str(e)

        finally:
            if "osint" in state["planned_run"]:
                state["planned_run"].remove("osint")

        return state


    def confidence_node(self, state: ForensicState):
        """
        Compute forensic confidence based ONLY on executed examinations.

        Rules:
        - At least one exam must have run
        - Skipped exams do NOT invalidate confidence
        - Confidence score reflects both:
            (a) strength of conclusions
            (b) coverage (how many exams ran)
        """

        conclusions = state.get("conclusions", {})
        skipped = state.get("skipped_exams", {})


        executed_exams = [
            exam for exam in ["metadata", "visual_environment", "ela", "osint"]
            if exam in conclusions
        ]

        if len(executed_exams) == 0:
            state["confidence_label"] = "Inconclusive"
            state["confidence_score"] = 0.0
            state["confidence_reasoning"] = (
                "No forensic examinations were successfully executed."
            )
            return state

        prompt = f"""
    You are a forensic examiner.

    Executed examinations:
    {executed_exams}

    Their conclusions:
    {json.dumps({k: conclusions[k] for k in executed_exams}, indent=2)}

    Rules:
    - Evaluate confidence ONLY from the executed examinations
    - Skipped exams do NOT imply failure
    - Fewer exams means lower coverage, not invalidity
    - Assign confidence_label from:
    High Confidence | Medium Confidence | Low Confidence
    - Assign confidence_score from 0–100
    - Explicitly state that the score is based ONLY on executed exams

    Return JSON only:
    {{
    "confidence_label": "...",
    "confidence_score": 0-100,
    "reasoning": "brief explanation"
    }}
    """

        raw = self.llm.invoke([HumanMessage(content=prompt)]).content
        result = safe_parse_json(raw)

        if not result:
            state["confidence_label"] = "Low Confidence"
            state["confidence_score"] = 0.3
            state["confidence_reasoning"] = (
                "Confidence derived from limited executed examinations only."
            )
            return state


        raw_score = float(result.get("confidence_score", 0))
        coverage_ratio = len(executed_exams) / 4.0  # metadata + 3 exams

        adjusted_score = raw_score * coverage_ratio

        state["confidence_label"] = result.get("confidence_label", "Low Confidence")
        state["confidence_score"] = round(adjusted_score / 100.0, 2)

        state["confidence_reasoning"] = (
            result.get("reasoning", "").strip()
            + f"\n\nNote: Confidence is based only on the following executed examinations: "
            + ", ".join(executed_exams)
            + "."
        )

        return state


    def report_node(self, state: ForensicState):
        """
        Generate a final forensic report with a single synthesized conclusion.

        Rules:
        - Synthesize ONLY from existing conclusions
        - Do NOT introduce new facts
        - Explicitly state scope limitations
        """

        conclusions = state.get("conclusions", {})
        skipped = state.get("skipped_exams", {})

        if not conclusions:
            state["final_report"] = (
                "FORENSIC IMAGE AUTHENTICITY REPORT\n\n"
                "No forensic examinations were successfully executed. "
                "No conclusion can be drawn."
            )
            return state

        prompt = f"""
    You are a forensic image examiner writing the FINAL CONCLUSION section
    of a forensic report.

    Individual examination conclusions:
    {json.dumps(conclusions, indent=2)}

    Skipped examinations:
    {json.dumps(skipped, indent=2)}

    Rules:
    - Write ONE coherent expert conclusion in plain English
    - Base the conclusion ONLY on the provided examination conclusions
    - Do NOT introduce new evidence or speculation
    - Acknowledge limitations due to skipped examinations
    - Do NOT list exams separately
    - Do NOT repeat text verbatim
    - Length: 4–6 sentences

    Write only the final conclusion text.
    """

        final_conclusion = self.llm.invoke(
            [HumanMessage(content=prompt)]
        ).content.strip()

        state["final_report"] = f"""
    FORENSIC IMAGE AUTHENTICITY REPORT

    Confidence Level: {state.get("confidence_label")}
    Confidence Score: {int(state.get("confidence_score", 0) * 100)}%

    Final Expert Conclusion:
    {final_conclusion}
    """

        return state

