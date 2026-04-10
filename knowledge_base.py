# ── knowledge_base.py ────────────────────────────────────────────────────────
# Builds and manages the local clinical knowledge base for RigidityIQ.
#
# This is the RAG (Retrieval-Augmented Generation) layer of the system.
# Rather than relying solely on Gemma 4's training weights for clinical knowledge,
# every assessment retrieves the most relevant passages from this vector store
# and injects them into the prompt. This grounds outputs in peer-reviewed criteria
# and makes the system's clinical reasoning auditable and updateable.
#
# The entire knowledge base runs locally — no external API, no internet dependency.
# ────────────────────────────────────────────────────────────────────────────
import os
# Block all network calls before loading any HuggingFace library.
# The embedding model (all-MiniLM-L6-v2) must load from the local cache only —
# any attempted download during a clinical encounter would cause a failure
# in a field setting with no connectivity.
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # block all network calls
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import chromadb
from chromadb.utils import embedding_functions

# all-MiniLM-L6-v2 is a lightweight sentence-transformer model (~80MB) optimised
# for semantic similarity tasks. It converts clinical text queries into vector
# embeddings, enabling the system to retrieve guideline passages by meaning
# rather than keyword matching — so a query about "shuffling gait" correctly
# retrieves passages about Grade 4 rigidity even if the words don't overlap exactly.
#
# cache_folder="./models" ensures the model loads from local disk after the
# one-time setup run by download_models.py. This path is explicit to avoid
# platform-specific default cache locations that may require symlinks (Windows).
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    cache_folder="./models"
)

# ChromaDB is a lightweight local vector database — no server, no cloud, no config.
# PersistentClient writes the vector store to disk at ./clinical_kb so it survives
# between application restarts without needing to re-embed the documents each time.
client = chromadb.PersistentClient(path="./clinical_kb")
collection = client.get_or_create_collection(
    name="parkinson_guidelines",
    embedding_function=ef
)

def build_knowledge_base():
    # Loads the clinical knowledge base into ChromaDB on first run.
    # On subsequent runs, if the collection already contains documents,
    # this function exits immediately — avoiding redundant re-embedding
    # which would slow startup and waste compute on constrained hardware.
    #
    # To update the knowledge base (e.g. with new clinical guidelines),
    # delete the ./clinical_kb directory and restart the application.
    
    documents = [
        # ── MDS-UPDRS Grade Definitions ─────────────────────────────────────
        # These are the authoritative grade boundary definitions from the
        # Movement Disorder Society's Unified Parkinson's Disease Rating Scale
        # (MDS-UPDRS) Part III, Item 3.3 — the international standard for
        # clinical rigidity assessment. Every grade assignment Gemma 4 makes
        # is grounded against these definitions via RAG retrieval.
        "MDS-UPDRS Item 3.3 Grade 0: No increase in muscle tone detected during passive movement of major joints at rest.",
        "MDS-UPDRS Item 3.3 Grade 1: Slight increase in muscle tone, manifested as a catch and release or minimal resistance at the end of range of motion during passive movement.",
        "MDS-UPDRS Item 3.3 Grade 2: Mild increase in muscle tone, manifested as a catch followed by minimal resistance throughout the remainder of passive range of motion.",
        "MDS-UPDRS Item 3.3 Grade 3: Moderate increase in muscle tone through most of the range of motion, but affected parts still movable.",
        "MDS-UPDRS Item 3.3 Grade 4: Severe increase in muscle tone. Passive movement is difficult and the full range of motion cannot be achieved.",

        # ── Clinical Observations by Grade ───────────────────────────────────
        # Maps observable behaviours — gait, arm swing, posture — to grade ranges.
        # These are what a CHW can actually see without physical examination,
        # and form the basis of the structured observation inputs in the UI.
        "Patients with Grade 1 Parkinson's rigidity may show slight resistance only when the examiner activates the limb. Walking speed may be near normal. Arm swing may be slightly reduced on the affected side.",
        "Patients with Grade 2 Parkinson's rigidity typically show reduced arm swing, slightly stooped posture, and mildly slow gait. Resistance is detectable without activation maneuver.",
        "Patients with Grade 3 Parkinson's rigidity show markedly reduced or absent arm swing, stooped posture, shuffling gait, and significant resistance throughout range of motion.",
        "Patients with Grade 4 Parkinson's rigidity show severe motor impairment. Passive movement of the affected limb is very difficult. Gait is severely disrupted and falls risk is high. Urgent specialist referral is required.",

        # ── Referral Guidelines ──────────────────────────────────────────────
        # Evidence-based thresholds for specialist referral, adapted for
        # community health worker settings where neurologists are unavailable.
        # These are retrieved and injected into every assessment to ensure
        # Gemma 4's referral recommendations are grounded in clinical protocol,
        # not generated from training data alone.
        "Grade 1 rigidity with no functional impairment may be monitored by a community health worker with 3-month follow-up.",
        "Patients with new onset rigidity Grade 2 or above should be referred to a neurologist or movement disorder specialist within 4 weeks.",
        "Patients with Grade 3 rigidity, falls history, or rapid progression should be referred urgently within 1 week.",
        "Patients with Grade 4 rigidity require immediate urgent referral. This grade indicates severe motor impairment and high falls risk. Same-day or next-day escalation is warranted.",

        # ── Differential Diagnosis ───────────────────────────────────────────
        # Rigidity must be correctly distinguished from other movement conditions.
        # These passages help Gemma 4 reason about differential diagnosis —
        # a capability critical for avoiding misclassification in the field.
        "Rigidity must be distinguished from spasticity. Rigidity in Parkinson's disease is present throughout passive range of motion (lead-pipe) or with superimposed tremor (cogwheel). Spasticity is velocity-dependent.",
        "Cogwheel rigidity, a ratchet-like resistance during passive movement, is characteristic of Parkinson's disease and indicates the presence of both rigidity and tremor.",
        "Rigidity in Parkinson's disease typically begins asymmetrically, affecting one side before the other, which distinguishes it from other parkinsonian syndromes.",

        # ── Medication Context ───────────────────────────────────────────────
        # Levodopa and other Parkinson's medications significantly affect rigidity
        # severity — the same patient can present very differently depending on
        # whether they are in an 'on' or 'off' medication state.
        # Including this context helps Gemma 4 flag when medication timing
        # may be confounding the assessment, rather than missing it entirely.
        "Parkinson's rigidity severity can fluctuate significantly with medication timing. Assessments should note whether the patient is in an 'on' state (medication effective) or 'off' state (medication wearing off).",
        "Levodopa is the most effective medication for Parkinson's rigidity. Patients on stable levodopa therapy may show significantly reduced rigidity compared to unmedicated baseline.",

        # ── Community Health Worker Guidance ─────────────────────────────────
        # Practical field protocols for CHWs conducting assessments without
        # specialist supervision. These are retrieved when the query involves
        # gait or functional observations, grounding the assessment notes
        # Gemma 4 provides in real-world CHW practice.
        "Community health workers should observe gait, arm swing, facial expression, and posture during a 10-metre walk test to inform rigidity assessment.",
        "In low-resource settings, the timed up and go test (TUG) is a practical screening tool. A TUG time greater than 12 seconds indicates significant motor impairment.",
        "Health workers should ask about medication timing, falls in the past month, and whether symptoms have worsened since last assessment.",
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Only embed and insert documents if the collection is empty.
    # Embedding is compute-intensive on constrained hardware — skipping this
    # on subsequent startups keeps the application launch time fast.
    if collection.count() == 0:
        collection.add(documents=documents, ids=ids)
        print(f"Knowledge base built with {len(documents)} documents")
    else:
        print(f"Knowledge base already exists with {collection.count()} documents")


def retrieve_relevant_context(query, n_results=4):
    # Retrieves the top-n most semantically relevant clinical guideline passages
    # for a given patient observation query.
    #
    # n_results=4 is a deliberate balance: enough context to ground the assessment
    # across grade definitions, referral thresholds, and observation criteria,
    # without overloading Gemma 4's context window or introducing irrelevant noise.
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results["documents"]:
        passages = results["documents"][0]
        return "\n\n".join(f"• {p}" for p in passages)
    
    return "No specific guidelines retrieved."
