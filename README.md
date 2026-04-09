# 🧠 RigidityIQ
### Offline Parkinson's Rigidity Assessment for the Last Mile

> Bringing neurological clinical decision support to community health workers
> in low-resource settings — powered entirely by Gemma 4 running locally via Ollama.

**Track:** Health & Sciences
**Model:** Gemma 4 E2B via Ollama
**Stack:** Python · Gradio · ChromaDB · SQLite · sentence-transformers

---

## The Problem

Uganda has approximately 6–7 trained neurologists for a population of over
47 million, roughly 0.03 neurologists per 100,000 people, compared to 2.96
in high-income countries. Patients with Parkinson's disease go unassessed for
years, not because the clinical knowledge doesn't exist, but because the tools
to apply it never reach them.

RigidityIQ puts neurological clinical decision support directly in the hands
of Community Health Workers (CHWs) — running fully offline on a basic laptop,
with no internet required during clinical use.

---

## Demo

> 📹 [Watch the demo video](#) ← replace with your YouTube link

![RigidityIQ Demo](assets/demo.gif) ← replace with your gif if you have one

---

## How It Works

A CHW enters structured clinical observations — walking speed, arm swing,
posture, and free-text notes. RigidityIQ returns a full clinical assessment
graded on the **MDS-UPDRS Part III Item 3.3** rigidity scale (Grades 0–4),
the same scale used by neurologists, along with a referral recommendation,
urgency tier, and plain-language notes.

### Inference Pipeline
Clinical Observations (Gradio UI) --> Semantic Search — all-MiniLM-L6-v2 --> ChromaDB Vector Store — MDS-UPDRS Guidelines (local) --> Gemma 4 E2B — Reasoning Pass (Native Thinking) --> Gemma 4 E2B — Structured JSON Pass (Native Function Calling) --> Schema Validation + Self-Correction Loop --> Gradio UI   SQLite DB + Report     (local)

100% AIR-GAPPED. ZERO CLOUD DEPENDENCY.
### Why Gemma 4

Gemma 4 met three non-negotiable requirements for this deployment context:

- **Native Thinking mode** — drives a deliberate reasoning pass before
  committing to a grade, reducing overconfident or inconsistent outputs
- **Native Function Calling Capability** — makes strict JSON schema
  enforcement reliable without an external parsing library
- **Multimodal Readiness** — the same model family handles text today
  and will handle video gait analysis tomorrow, without an architectural
  rewrite

---

## Project Structure
rigidityiq/
│
├── app.py              # Gradio UI + assessment orchestration
├── engine.py           # Two-pass Gemma 4 inference pipeline
├── knowledge_base.py   # ChromaDB RAG vector store
├── database.py         # SQLite patient history
├── prompts.py          # Clinical prompt templates
└── download_models.py  # One-time offline setup script

---

## Quickstart

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed on your machine

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rigidityiq.git
cd rigidityiq
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Gemma 4 via Ollama

```bash
ollama pull gemma4:e2b
```

> This is a one-time download (~7.2GB). Run it at a location with a
> stable internet connection. After this step, the model runs fully offline.

### 4. Download the embedding model

```bash
python download_models.py
```

> Downloads all-MiniLM-L6-v2 (~80MB) to `./models`. One-time only.

### 5. Run the app

```bash
python app.py
```

Open your browser at `http://localhost:7860`

> After setup, steps 3 and 4 are never needed again. The application
> runs 100% offline from this point forward.

---

## Offline Deployment (Field Use)

RigidityIQ uses a **Hub-and-Spoke deployment model**. Complete the setup
steps above once at a location with internet access, then:

1. Copy the entire project folder to a USB drive
2. Transfer to the target device at the clinic or field site
3. Run `python app.py` — no internet required, ever again

Patient data is stored locally in `rigidityiq_patients.db` and never
leaves the device.

---

## Clinical Grounding

Assessment criteria are based on the **MDS-UPDRS Part III Item 3.3**
rigidity scale — the international standard for clinical Parkinson's
rigidity assessment:

| Grade | Definition |
|-------|-----------|
| 0 | No increase in muscle tone |
| 1 | Slight increase, only with activation maneuver |
| 2 | Mild increase detected without activation maneuver |
| 3 | Moderate increase, full range of motion still possible |
| 4 | Severe increase, full range of motion not achievable |

Grade boundaries, referral thresholds, and clinical descriptions are
stored in a local ChromaDB vector store and retrieved via semantic
search on every assessment — grounding every output in peer-reviewed
literature rather than model weights alone.

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Ollama (local) over cloud API | Zero connectivity dependency during clinical use |
| Gemma 4 E2B quantization | Runs on 16GB consumer hardware — matches field device profile |
| Two-pass inference | Reasoning first, structured output second — reduces grade inconsistency |
| SQLite over server DB | Zero configuration, zero network, runs on any hardware |
| RAG over fine-tuning | Clinical guidelines stay auditable and updateable without retraining |
| all-MiniLM-L6-v2 | Lightweight (~80MB), strong semantic similarity, offline-capable |

---

## Limitations & Honest Notes

- Inference takes approximately **124 seconds** on a 16GB laptop — this
  is a deliberate feature of the Deep Clinical Reasoning mode, not a bug.
  It mirrors the time a clinician takes to review notes before committing
  to a grade.
- RigidityIQ does **not** replace a neurologist. It is a decision support
  tool — the CHW's observations and judgment remain part of the loop.
- Assessment quality depends on the accuracy of the CHW's observations.
  A validation study on CHW-entered inputs is identified as the next
  research step.

---

## Roadmap

The next phase leverages Gemma 4's **Multimodal Readiness** directly.
A CHW will record a 10-second video of the patient walking — Gemma 4
will analyze frames to detect arm swing reduction, shuffling gait, and
postural stooping automatically.

The RAG pipeline, SQLite history, offline deployment model, and
self-correction loop are already in place and transfer directly to
video input. The input layer is what changes.

---

## License
MIT License — see [LICENSE](LICENSE)


*Clinical grounding based on MDS-UPDRS Part III Item 3.3 rigidity scale.
Grade boundaries and clinical descriptions informed by peer-reviewed
Parkinson's disease research.*
