import ollama
import json
from prompts import SYSTEM_PROMPT, ASSESSMENT_PROMPT, REASONING_PROMPT
from knowledge_base import retrieve_relevant_context


def assess_with_reasoning(age, body_site, walking_speed, arm_swing,
                           posture, observations, previous_grade, history):

    patient_data = build_patient_data(
        age, body_site, walking_speed,
        arm_swing, posture, observations, previous_grade
    )
    history_context = build_history_context(history)

    # ── RAG: Retrieve relevant clinical guidelines ──
    search_query = f"Parkinson's rigidity {body_site} {walking_speed} {arm_swing} {posture}"
    retrieved_context = retrieve_relevant_context(search_query)

    rag_context = f"""
RETRIEVED CLINICAL GUIDELINES:
{retrieved_context}

Use the above guidelines to ground your assessment.
"""

    # Step 1 — Reasoning with retrieved context
    reasoning_response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rag_context + "\n\n" + REASONING_PROMPT.format(
                patient_data=patient_data
            )}
        ]
    )
    reasoning = reasoning_response["message"]["content"]

    # Step 2 — Structured output
    assessment_response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rag_context + "\n\n" + REASONING_PROMPT.format(
                patient_data=patient_data
            )},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": ASSESSMENT_PROMPT.format(
                patient_data=patient_data,
                history_context=history_context
            )}
        ]
    )

    raw = assessment_response["message"]["content"]

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        result["reasoning_trace"] = reasoning
        result["retrieved_context"] = retrieved_context
        return result, None
    except json.JSONDecodeError:
        return None, f"Could not parse model output:\n{raw}"