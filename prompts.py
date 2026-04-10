# ── prompts.py ───────────────────────────────────────────────────────────────
# Prompt templates for RigidityIQ's two-pass clinical reasoning pipeline.
#
# Prompt engineering is where the clinical safety of this system is defined.
# These three prompts work together in a deliberate sequence — system context
# first, free reasoning second, structured output third — to produce assessments
# that are conservative, evidence-grounded, and consistently formatted.
#
# Changing these prompts changes clinical behaviour. Any modifications should
# be tested across a range of patient scenarios before deployment.
# ────────────────────────────────────────────────────────────────────────────


# ── SYSTEM_PROMPT ─────────────────────────────────────────────────────────────
# Establishes Gemma 4's role, clinical grounding, and behavioural constraints
# for the entire conversation. This is sent as the system message on every
# inference call — it is the persistent clinical identity of the model.
#
# Key design decisions:
# - "low-resource settings where neurologists are unavailable" — anchors the
#   model's output register. Responses should be actionable for a CHW, not
#   written for a specialist audience.
# - Explicit grade definitions — even though the same definitions exist in the
#   RAG knowledge base, repeating them here in the system prompt ensures they
#   are always in context, regardless of what the retrieval step returns.
# - "Conservative and evidence-based" and "Explicit about uncertainty" —
#   deliberate constraints that counteract the tendency of LLMs to express
#   false confidence. In a clinical tool, an uncertain output that says
#   "I cannot determine the grade from these observations" is safer than
#   a confident but wrong grade.
SYSTEM_PROMPT = """You are a specialist clinical decision support system for 
Parkinson's disease assessment, designed to assist community health workers 
in low-resource settings where neurologists are unavailable.

You are grounded in the MDS-UPDRS Part III Item 3.3 rigidity scale:
- Grade 0: No increase in muscle tone
- Grade 1: Slight increase, only with activation maneuver  
- Grade 2: Mild increase detected without activation maneuver
- Grade 3: Moderate increase, full range of motion still possible
- Grade 4: Severe increase, full range of motion not achievable

Your assessments must be:
- Conservative and evidence-based
- Expressed in plain language a non-specialist can act on
- Explicit about uncertainty when observations are ambiguous
- Structured and consistent across all assessments"""


# ── REASONING_PROMPT ──────────────────────────────────────────────────────────
# The first of the two inference passes — the deliberation step.
#
# This prompt intentionally does NOT ask for a grade. It asks the model to
# think through the evidence before committing to anything. This mirrors
# clinical reasoning: a neurologist reviews observations and weighs alternatives
# before writing a conclusion, they don't generate a grade on first glance.
#
# The four questions are structured to cover the full diagnostic reasoning chain:
# 1. Evidence identification — what is actually present in the observations
# 2. Grade mapping — how the evidence maps to MDS-UPDRS criteria
# 3. Red flag detection — urgency signals that override grade-based referral logic
# 4. Counterfactual thinking — what would change the assessment, surfacing uncertainty
#
# The output of this prompt becomes the assistant turn in the second inference
# call — Gemma 4 is then committing to a structured output that is consistent
# with its own prior reasoning, rather than generating a grade from scratch.
# This is the core mechanism that makes the two-pass architecture more reliable.
REASONING_PROMPT = """Before making your final assessment, think through 
these observations carefully:

{patient_data}

Consider:
1. Which symptoms most strongly indicate rigidity?
2. What grade do the observations most closely match?
3. Are there any red flags requiring urgent referral?
4. What would change your assessment?

Think step by step."""


# ── ASSESSMENT_PROMPT ─────────────────────────────────────────────────────────
# The second inference pass — structured output generation.
#
# This prompt is sent after the reasoning trace has been produced and injected
# back as context. At this point Gemma 4 has already deliberated — this prompt
# asks it to formalise that reasoning into a strict JSON schema.
#
# Key design decisions:
# - history_context injection — the patient's last 3 assessments are included
#   here (not in the reasoning prompt) so the progression commentary appears
#   in the structured output field, not in the free reasoning trace.
# - "Return ONLY the JSON" — firm instruction to suppress any conversational
#   wrapper text. Combined with clean_json_output() in engine.py, this handles
#   the most common formatting failures from instruction-tuned models.
# - Every field in the schema has a clinical purpose:
#     body_site / rigidity_grade / grade_label — the core clinical finding
#     confidence — surfaces model uncertainty to the CHW
#     clinical_reasoning — the human-readable justification
#     key_symptoms — the observable evidence that drove the grade
#     progression — longitudinal commentary using the injected history
#     referral_recommended — binary action signal, unambiguous for CHWs
#     urgency — tiered escalation: Routine / Soon / Urgent
#     health_worker_notes — plain-language next steps, no medical jargon
#     follow_up_timeframe — concrete scheduling guidance
ASSESSMENT_PROMPT = """Based on the following patient observations, provide 
a structured rigidity assessment.

{patient_data}

{history_context}

Respond in valid JSON with exactly this structure:
{{
    "body_site": "string",
    "rigidity_grade": 0,
    "grade_label": "string",
    "confidence": "High/Medium/Low",
    "clinical_reasoning": "string",
    "key_symptoms": ["symptom1", "symptom2", "symptom3"],
    "progression": "string",
    "referral_recommended": true,
    "urgency": "Routine/Soon/Urgent",
    "health_worker_notes": "string",
    "follow_up_timeframe": "string"
}}

Return ONLY the JSON. No explanation before or after."""
