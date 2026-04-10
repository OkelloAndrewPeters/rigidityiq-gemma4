# ── app.py ──────────────────────────────────────────────────────────────────
# Entry point for RigidityIQ.
# This file builds the Gradio UI, wires together all backend services,
# and defines the assessment workflow that a Community Health Worker (CHW)
# interacts with during a patient consultation.
# ────────────────────────────────────────────────────────────────────────────

import os
# Force all HuggingFace and transformer libraries to operate fully offline.
# These must be set BEFORE any library imports — once a library loads,
# it may have already attempted a network call. Setting them here at the
# top of the entry point guarantees no internet is required during clinical use.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" #Required on Windows where symlinks are restricted

#importing all necessary libraries to use
import gradio as gr
import json
from engine import assess_with_reasoning
from database import init_db, save_assessment, get_patient_history, get_all_patients

from knowledge_base import build_knowledge_base


# ── Service Initialisation ───────────────────────────────────────────────────
# Both services are initialised once at startup, before the UI loads.
# This ensures the vector store and database are ready before the first
# patient assessment is requested — avoiding cold-start delays mid-consultation.

build_knowledge_base()   #Load MDS-UPDRS clinical guidelines into the local ChromaDB vector store
init_db()                #Create SQLite tables if they don't exist yet (safe to call repeatedly)


# ── Grade Display Mapping ─────────────────────────────────────────────────────
# Maps integer rigidity grades (0–3) from the MDS-UPDRS Part III Item 3.3 scale
# to colour-coded display labels. Visual grading helps CHWs with limited clinical
# training immediately understand severity without needing to interpret numbers.
GRADE_COLORS = {
    0: "🟢 Grade 0 — No Rigidity",
    1: "🟡 Grade 1 — Slight",
    2: "🟠 Grade 2 — Mild",
    3: "🔴 Grade 3 — Moderate",
    4: "🚨 Grade 4 — Severe"
}

def format_report(result):
    # Transforms the structured JSON output from Gemma 4 into a human-readable
    # clinical report. The goal is a format a CHW can read aloud, file, or
    # photograph — not a developer-facing JSON dump.
    if not result:
        return "Assessment failed. Please try again."
    
    grade = result.get("rigidity_grade", "?")
    grade_display = GRADE_COLORS.get(grade, f"Grade {grade}")

    # Referral recommendation is a binary decision to keep it unambiguous for CHWs.
    # A CHW should never have to interpret nuance in an emergency referral decision
    referral = "✅ YES — Refer to specialist" if result.get("referral_recommended") else "❌ No referral needed"
    
    report = f"""
{'='*50}
RIGIDITY ASSESSMENT REPORT
{'='*50}

BODY SITE: {result.get('body_site', 'N/A')}
RIGIDITY GRADE: {grade_display}
CONFIDENCE: {result.get('confidence', 'N/A')}

CLINICAL REASONING:
{result.get('clinical_reasoning', 'N/A')}

KEY SYMPTOMS OBSERVED:
{chr(10).join(f"  • {s}" for s in result.get('key_symptoms', []))}

PROGRESSION NOTE:
{result.get('progression', 'N/A')}

{'='*50}
REFERRAL: {referral}
URGENCY: {result.get('urgency', 'N/A')}
FOLLOW UP: {result.get('follow_up_timeframe', 'N/A')}

NOTES FOR HEALTH WORKER:
{result.get('health_worker_notes', 'N/A')}
{'='*50}
"""
    return report


def run_assessment(patient_code, age, body_site, previous_grade,
                   walking, arm, posture, observations):
    # Core assessment function — called when the CHW clicks "Generate Assessment".
    # Orchestrates the full pipeline: input validation → history retrieval →
    # Gemma 4 inference → database save → UI update.
    # Keeping this orchestration in app.py (rather than engine.py) maintains a
    # clean separation between UI logic and inference logic.
    if not patient_code.strip():
        return "Please enter a patient code.", "", []
    
    # Retrieve this patient's previous assessments from the local SQLite database.
    # Passed into the engine so Gemma 4 can comment on disease progression,
    # not just the current visit in isolation.
    history = get_patient_history(patient_code)

    # Run the two-pass Gemma 4 inference pipeline.
    # Returns a validated structured result dict, or an error string on failure.
    result, error = assess_with_reasoning(
        age, body_site, walking, arm,
        posture, observations, previous_grade, history
    )

    if error:
        return error, "", []

    # Persist the validated assessment to SQLite immediately after inference.
    # Saving before formatting ensures data is never lost due to a display error.
    save_assessment(patient_code, result)

    #Format the structured result into the CHW-facing clinical report.
    report = format_report(result)

    # Refresh the history table to include the assessment just saved,
    # giving the CHW immediate visual confirmation the record was stored.
    updated_history = get_patient_history(patient_code)
    history_display = [
        [h["date"], h["body_site"], f"Grade {h['grade']}", h["urgency"]]
        for h in updated_history
    ]

    # Return three outputs: the formatted report, the raw reasoning trace,
    # and the updated history table — each bound to a separate UI component below.
    return report, result.get("reasoning_trace", ""), history_display


# ── Custom CSS ───────────────────────────────────────────────────────────────
# Visual styling is intentionally clinical and minimal — not decorative.
# High contrast, monospace report output, and dark reasoning trace are designed
# to feel like a medical tool, not a consumer app. This matters for CHW trust.
custom_css = """
/* Main Container */
.gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}

/* Headers */
h1 {
    color: #1e293b !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 0px !important;
}

h2 {
    color: #64748b !important;
    font-weight: 400 !important;
    font-size: 1.1rem !important;
    margin-top: 0px !important;
}

/* Tab Styling */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
}

button.selected {
    border-bottom: 3px solid #2563eb !important;
    color: #2563eb !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* Cards (Columns) */
.gap.p-4 {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1) !important;
    padding: 24px !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
}

/* Input Fields */
input, select, textarea {
    border-radius: 8px !important;
    border-color: #cbd5e1 !important;
}

/* Textbox styling for the Report */
#clinical-report-box textarea {
    background-color: #f1f5f9 !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

/* Reasoning Trace styling */
#reasoning-box textarea {
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 13px !important;
}

/* Labels */
label span {
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}
"""

# ── Gradio UI Layout ─────────────────────────────────────────────────────────
# The UI is split into two tabs: New Assessment and Patient History.
# This mirrors the two core workflows of a CHW: conducting a new visit
# and reviewing a returning patient's trajectory over time.
with gr.Blocks(
    #title="RigidityIQ — Parkinson's Assessment",
    #theme=gr.themes.Soft()
    css=custom_css, 
    title="RigidityIQ — Clinical Decision Support",
    theme=gr.themes.Soft(primary_hue="blue", spacing_size="sm", radius_size="md")

) as app:

    ## Display app header
    gr.Markdown("""
    # 🧠 RigidityIQ
    ## Parkinson's Rigidity Assessment — Offline Clinical Decision Support
    *Powered by Gemma 4 via Ollama | No internet required | Patient data stays on device*
    ---
    """)

    with gr.Tabs():
        # ── Tab 1: New Assessment ────────────────────────────────────────────
        # Left column collects structured clinical observations.
        # Right column displays the assessment report and reasoning trace.
        # Two-column layout allows the CHW to reference inputs while reading output.
        with gr.Tab("New Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Patient")
                    patient_code = gr.Textbox(label="Patient Code", placeholder="e.g. PD-001")
                    age = gr.Slider(minimum=40, maximum=90, value=65, step=1, label="Age")
                    body_site = gr.Dropdown(
                        choices=["Neck","Right Upper Extremity","Left Upper Extremity",
                                 "Right Lower Extremity","Left Lower Extremity"],
                        label="Body Site",
                        value="Right Upper Extremity"
                    )

                    # Previous grade is injected into context to enable progression tracking.
                    # Knowing a patient was Grade 1 last visit changes the clinical meaning
                    # of a Grade 2 result today — it signals deterioration, not just severity
                    previous_grade = gr.Dropdown(
                            choices=[
                                "No previous assessment",
                                "Grade 0 — No rigidity",
                                "Grade 1 — Slight",
                                "Grade 2 — Mild",
                                "Grade 3 — Moderate",
                                "Grade 4 — Severe"
                            ],
                            label="Previous Grade",
                            value="No previous assessment"
                        )

                    #Movement observation input fields
                    gr.Markdown("### Movement Observations")

                    # These four observation fields map directly to MDS-UPDRS motor examination
                    # criteria. Dropdown options are worded to match what a CHW can observe
                    # without clinical training — no medical jargon required.
                    walking = gr.Dropdown(
                        choices=["Normal","Slightly slow","Slow and shuffling","Very slow"],
                        label="Walking Speed",
                        value="Slow and shuffling"
                    )
                    arm = gr.Dropdown(
                        choices=["Normal swing","Slightly reduced","Markedly reduced","Absent"],
                        label="Arm Swing",
                        value="Markedly reduced"
                    )
                    posture = gr.Dropdown(
                        choices=["Upright","Slightly stooped","Stooped","Severely stooped"],
                        label="Posture",
                        value="Slightly stooped"
                    )

                    # Free-text field for observations that don't fit structured dropdowns —
                    # tremor, facial masking, medication timing, falls history.
                    # This feeds directly into Gemma 4's context and often contains
                    # the most diagnostically relevant information.
                    observations = gr.Textbox(
                        label="Additional Observations",
                        placeholder="Facial expression, voice, tremor, medication timing, balance...",
                        lines=4
                    )
                    submit = gr.Button("🔍 Generate Assessment", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Assessment Report")
                    report_output = gr.Textbox(label="Clinical Report", lines=25)
                    gr.Markdown("### Model Reasoning")

                    # The reasoning trace exposes Gemma 4's internal deliberation to the CHW.
                    # This serves two purposes: transparency (the CHW can see why a grade
                    # was assigned) and education (repeated exposure teaches clinical reasoning).
                    reasoning_output = gr.Textbox(label="Gemma 4 Reasoning Trace", lines=8)

            # Bind the submit button to the full assessment pipeline.
            # Outputs map to: report text, reasoning trace, and history table.
            submit.click(
                fn=run_assessment,
                inputs=[patient_code, age, body_site, previous_grade, walking, arm, posture, observations],
                outputs=[report_output, reasoning_output, gr.Dataframe(
                    headers=["Date", "Body Site", "Grade", "Urgency"],
                    label="Assessment History"
                )]
            )

        # ── Tab 2: Patient History ───────────────────────────────────────────
        # Allows CHWs to look up any patient's full assessment history by code.
        # Critical for longitudinal care — a CHW seeing a patient for the third
        # time needs to see the trajectory, not just the last result.
        with gr.Tab("Patient History"):
            with gr.Row():
                history_code = gr.Textbox(label="Enter Patient Code", placeholder="e.g. PD-001")
                load_btn = gr.Button("Load History", variant="secondary")

            #Display historical assessments in a table
            history_table = gr.Dataframe(headers=["Date", "Body Site", "Grade", "Urgency"], label="Assessment History")

            #Load and format patient history for display
            def load_history(code):
                # Fetches and formats all stored assessments for a given patient code.
                # Returns up to 10 most recent records, ordered newest first.
                records = get_patient_history(code)
                return [
                    [r["date"], r["body_site"], f"Grade {r['grade']}", r["urgency"]]
                    for r in records
                ]

            load_btn.click(fn=load_history, inputs=history_code, outputs=history_table)

    gr.Markdown("""
    ---
    **Clinical grounding:** Assessment criteria based on MDS-UPDRS Part III Item 3.3 
    rigidity scale. Grade boundaries and clinical descriptions are informed by 
    peer-reviewed Parkinson's disease research.*
    """)

app.launch()
