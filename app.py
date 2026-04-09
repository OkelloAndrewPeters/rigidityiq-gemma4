import gradio as gr
import json
from engine import assess_with_reasoning
from database import init_db, save_assessment, get_patient_history, get_all_patients

from knowledge_base import build_knowledge_base

# Build knowledge base on first run
build_knowledge_base()

# Initialise database on startup
init_db()

GRADE_COLORS = {
    0: "🟢 Grade 0 — No Rigidity",
    1: "🟡 Grade 1 — Slight",
    2: "🟠 Grade 2 — Mild",
    3: "🔴 Grade 3 — Moderate to Severe"
}

def format_report(result):
    if not result:
        return "Assessment failed. Please try again."
    
    grade = result.get("rigidity_grade", "?")
    grade_display = GRADE_COLORS.get(grade, f"Grade {grade}")
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
    
    if not patient_code.strip():
        return "Please enter a patient code.", "", []

    # Get patient history from database
    history = get_patient_history(patient_code)

    # Run assessment
    result, error = assess_with_reasoning(
        age, body_site, walking, arm,
        posture, observations, previous_grade, history
    )

    if error:
        return error, "", []

    # Save to database
    save_assessment(patient_code, result)

    # Format report
    report = format_report(result)

    # Update history display
    updated_history = get_patient_history(patient_code)
    history_display = [
        [h["date"], h["body_site"], f"Grade {h['grade']}", h["urgency"]]
        for h in updated_history
    ]

    return report, result.get("reasoning_trace", ""), history_display


with gr.Blocks(
    title="RigidityIQ — Parkinson's Assessment",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
    # 🧠 RigidityIQ
    ## Parkinson's Rigidity Assessment — Offline Clinical Decision Support
    *Powered by Gemma 4 via Ollama | No internet required | Patient data stays on device*
    ---
    """)

    with gr.Tabs():

        # ── Tab 1: Assessment ──────────────────────────────
        with gr.Tab("New Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Patient")
                    patient_code = gr.Textbox(
                        label="Patient Code",
                        placeholder="e.g. PD-001"
                    )
                    age = gr.Slider(
                        minimum=40, maximum=90,
                        value=65, step=1,
                        label="Age"
                    )
                    body_site = gr.Dropdown(
                        choices=[
                            "Neck",
                            "Right Upper Extremity",
                            "Left Upper Extremity",
                            "Right Lower Extremity",
                            "Left Lower Extremity"
                        ],
                        label="Body Site",
                        value="Right Upper Extremity"
                    )
                    previous_grade = gr.Dropdown(
                        choices=[
                            "No previous assessment",
                            "Grade 0 — No rigidity",
                            "Grade 1 — Slight",
                            "Grade 2 — Mild",
                            "Grade 3 — Moderate to Severe"
                        ],
                        label="Previous Grade",
                        value="No previous assessment"
                    )

                    gr.Markdown("### Movement Observations")
                    walking = gr.Dropdown(
                        choices=[
                            "Normal",
                            "Slightly slow",
                            "Slow and shuffling",
                            "Very slow"
                        ],
                        label="Walking Speed",
                        value="Slow and shuffling"
                    )
                    arm = gr.Dropdown(
                        choices=[
                            "Normal swing",
                            "Slightly reduced",
                            "Markedly reduced",
                            "Absent"
                        ],
                        label="Arm Swing",
                        value="Markedly reduced"
                    )
                    posture = gr.Dropdown(
                        choices=[
                            "Upright",
                            "Slightly stooped",
                            "Stooped",
                            "Severely stooped"
                        ],
                        label="Posture",
                        value="Slightly stooped"
                    )
                    observations = gr.Textbox(
                        label="Additional Observations",
                        placeholder="Facial expression, voice, tremor, medication timing, balance...",
                        lines=4
                    )
                    submit = gr.Button(
                        "🔍 Generate Assessment",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Assessment Report")
                    report_output = gr.Textbox(
                        label="Clinical Report",
                        lines=25
                    )
                    gr.Markdown("### Model Reasoning")
                    reasoning_output = gr.Textbox(
                        label="Gemma 4 Reasoning Trace",
                        lines=8
                    )

            submit.click(
                fn=run_assessment,
                inputs=[
                    patient_code, age, body_site, previous_grade,
                    walking, arm, posture, observations
                ],
                outputs=[report_output, reasoning_output, gr.Dataframe(
                    headers=["Date", "Body Site", "Grade", "Urgency"],
                    label="Assessment History"
                )]
            )

        # ── Tab 2: Patient History ─────────────────────────
        with gr.Tab("Patient History"):
            with gr.Row():
                history_code = gr.Textbox(
                    label="Enter Patient Code",
                    placeholder="e.g. PD-001"
                )
                load_btn = gr.Button("Load History", variant="secondary")

            history_table = gr.Dataframe(
                headers=["Date", "Body Site", "Grade", "Urgency"],
                label="Assessment History"
            )

            def load_history(code):
                records = get_patient_history(code)
                return [
                    [r["date"], r["body_site"],
                     f"Grade {r['grade']}", r["urgency"]]
                    for r in records
                ]

            load_btn.click(
                fn=load_history,
                inputs=history_code,
                outputs=history_table
            )

    gr.Markdown("""
    ---
    **Clinical grounding:** MDS-UPDRS Item 3.3 | Validated on WearGait-PD dataset
    ($n=147$, $\\kappa=0.24$, subject-independent evaluation)

    *RigidityIQ is a clinical decision support aid. It does not replace 
    professional medical judgment.*
    """)

app.launch()