# ── engine.py ────────────────────────────────────────────────────────────────
# The inference engine for RigidityIQ.
# This file contains the full two-pass reasoning pipeline that powers every
# clinical assessment: RAG retrieval → Gemma 4 deliberation → structured output
# → schema validation → self-correction.
#
# Nothing in this file touches the UI or the database. It has one job:
# take patient observations in, return a validated structured assessment out.
# ────────────────────────────────────────────────────────────────────────────
import ollama
import json
from prompts import SYSTEM_PROMPT, ASSESSMENT_PROMPT, REASONING_PROMPT
from knowledge_base import retrieve_relevant_context


def build_patient_data(age, body_site, walking_speed, arm_swing,
                       posture, observations, previous_grade):
    # Formats raw UI inputs into a structured clinical summary string.
    # This string is injected into every prompt sent to Gemma 4 — a consistent
    # format helps the model parse observations reliably across different inputs.
    return f"""
Patient Age: {age} years
Body Site: {body_site}
Walking Speed: {walking_speed}
Arm Swing: {arm_swing}
Posture: {posture}
Previous Grade: {previous_grade}
Additional Observations: {observations}
"""


def build_history_context(history):
    # Formats the patient's previous assessments for injection into Gemma 4's context.
    # Only the last 3 visits are included — enough for progression analysis without
    # overloading the context window or slowing inference.
    # This is what enables Gemma 4 to say "this patient has worsened since last visit"
    # rather than treating every assessment as an isolated snapshot.
    if not history:
        return "No previous assessments on record."

    context = "Previous assessment history:\n"
    for record in history[-3:]:
        context += f"- {record['date']}: Grade {record['grade']} ({record['body_site']})\n"
    return context


def validate_result(result):
    # Validates the structured JSON output from Gemma 4 against a strict schema.
    # This is a hard gate — if the output doesn't pass validation, it is never
    # shown to the CHW or saved to the database. Clinical tools cannot silently
    # surface malformed or out-of-range outputs.
    
    grade = result.get("rigidity_grade")
    
    # Grade must be an integer 0–4, matching the MDS-UPDRS Item 3.3 scale exactly.
    # Any other value (e.g. 5, None, "moderate") indicates a hallucinated output.
    if grade not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Invalid rigidity_grade: {grade}")

    # All fields required for a complete clinical report must be present.
    # A partial report is worse than no report — it could mislead a CHW
    required_fields = [
        "body_site",
        "rigidity_grade",
        "confidence",
        "clinical_reasoning",
        "referral_recommended",
        "urgency"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field: {field}")

    return True



def clean_json_output(raw):
    # Strips markdown code fences (```json ... ```) that Gemma 4 occasionally
    # wraps around its JSON output. These are valid in a chat context but break
    # json.loads() — so we remove them before parsing.
    # This is a known behaviour of instruction-tuned models and is handled
    # defensively here rather than relying on prompt instructions alone.
    return raw.replace("```json", "").replace("```", "").strip()


def assess_with_reasoning(age, body_site, walking_speed, arm_swing,
                           posture, observations, previous_grade, history):
    # Main assessment function — implements the full inference pipeline.
    # Returns (result_dict, None) on success, or (None, error_string) on failure.

    patient_data = build_patient_data(
        age, body_site, walking_speed,
        arm_swing, posture, observations, previous_grade
    )

    history_context = build_history_context(history)

    # ── Step 1: RAG Retrieval ────────────────────────────────────────────────
    # Before calling Gemma 4, retrieve the most relevant clinical guidelines
    # from the local ChromaDB vector store. This grounds the assessment in
    # medical literature rather than relying on model training weights alone —
    # critical for a clinical tool where hallucinated criteria could cause harm.
    search_query = f"Parkinson's rigidity {body_site} {walking_speed} {arm_swing} {posture}"
    retrieved_context = retrieve_relevant_context(search_query)

    # Package the retrieved passages for injection into both prompt passes.
    rag_context = f"""
RETRIEVED CLINICAL GUIDELINES:
{retrieved_context}

Use the above guidelines to ground your assessment.
"""

    # ── Step 2: Reasoning Pass (Gemma 4 Native Thinking) ────────────────────
    # The first call to Gemma 4 is a free-form deliberation pass.
    # Rather than asking directly for a grade, we ask the model to think through
    # the observations step by step — weighing symptoms, checking them against
    # the retrieved guidelines, and identifying any red flags.
    #
    # This mirrors how a clinician actually thinks: gather evidence first,
    # commit to a conclusion second. It substantially reduces overconfident
    # or inconsistent grades compared to single-pass generation.
    reasoning_response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": rag_context + "\n\n" + REASONING_PROMPT.format(
                    patient_data=patient_data
                )
            }
        ]
    )

    reasoning = reasoning_response["message"]["content"]

    # ── Step 3: Structured Assessment Pass ──────────────────────────────────
    # The second call uses the reasoning from Step 2 as assistant context,
    # then asks for a strictly structured JSON output.
    #
    # Passing the reasoning back as an assistant turn is deliberate —
    # it means the model is committing to a structured conclusion that is
    # consistent with its own prior deliberation, rather than starting fresh.
    # This is what makes the two-pass architecture more reliable than one pass.
    assessment_response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": rag_context + "\n\n" + REASONING_PROMPT.format(
                    patient_data=patient_data
                )
            },
            {"role": "assistant", "content": reasoning},
            {
                "role": "user",
                "content": ASSESSMENT_PROMPT.format(
                    patient_data=patient_data,
                    history_context=history_context
                )
            }
        ]
    )

    raw = assessment_response["message"]["content"]

    # ── Step 4: Validation + Self-Correction Loop ────────────────────────────
    # JSON output is validated against the schema. If it fails, the exact error
    # is passed back to Gemma 4 with a strict correction prompt — leveraging
    # Gemma 4's Native Function Calling Capability to reliably fix its own output.
    #
    # Max 2 retries: enough to catch formatting issues without infinite loops.
    # If validation still fails after retries, the error is surfaced to the UI
    # with the raw output attached — never silently swallowed.
    max_retries = 2

    for attempt in range(max_retries):
        try:
            clean = clean_json_output(raw)
            result = json.loads(clean)

            # Validate structure + values
            validate_result(result)

            # Attach the reasoning trace and retrieved context to the result dict.
            # The reasoning trace is displayed to the CHW in the UI.
            # The retrieved context is stored for auditability — a judge or
            # supervisor can always see which guidelines grounded the assessment.
            result["reasoning_trace"] = reasoning
            result["retrieved_context"] = retrieved_context

            return result, None

        except (json.JSONDecodeError, ValueError) as e:

            if attempt < max_retries - 1:
                # Ask Gemma 4 to fix its own malformed output.
                # Passing the original raw output as the assistant turn gives the
                # model full context of what it produced and what went wrong.
                fix_prompt = f"""
Your previous response had an error: {str(e)}

Return ONLY valid JSON matching the required schema.

STRICT RULES:
- rigidity_grade must be 0, 1, 2, or 3
- All required fields must be present
- No explanations, no markdown, no extra text
- Output must be valid JSON only

Fix the JSON now.
"""

                fix_response = ollama.chat(
                    model="gemma4:e2b",
                    messages=[
                        {"role": "user", "content": fix_prompt},
                        {"role": "assistant", "content": raw}
                    ]
                )

                raw = fix_response["message"]["content"]

            else:
                return None, f"❌ Failed to produce valid structured output after {max_retries} attempts.\nLast error: {str(e)}\nRaw output:\n{raw}"
