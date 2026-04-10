# ── database.py ──────────────────────────────────────────────────────────────
# Local patient data persistence layer for RigidityIQ.
#
# All patient records are stored in a local SQLite database — a single file
# on the device. No data ever leaves the device, no server is required, and
# the database survives application restarts. This is the longitudinal memory
# of the system: without it, every assessment would be a one-off snapshot
# with no ability to track disease progression over time.
# ────────────────────────────────────────────────────────────────────────────
import sqlite3
from datetime import datetime
from pathlib import Path

# Single file database — stored in the application's working directory.
# SQLite was chosen over a full database server (PostgreSQL, MySQL) deliberately:
# it requires zero configuration, zero network, and runs on any hardware.
# For a deployment target of district health facilities in low-resource settings,
# operational simplicity is a clinical safety requirement, not just a convenience.
DB_PATH = "rigidityiq_patients.db"


def init_db():
    # Initialises the database schema on first run.
    # CREATE TABLE IF NOT EXISTS makes this safe to call on every startup —
    # it creates tables only if they don't already exist, so existing patient
    # data is never overwritten or lost between application restarts.


    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # The patients table stores basic demographic metadata per patient code.
    # Patient codes are CHW-assigned identifiers (e.g. PD-001) that replace
    # personally identifiable information — names, addresses, national IDs
    # are never stored, protecting patient privacy on a shared device.
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT NOT NULL,
            age INTEGER,
            created_at TEXT
        )
    """)

    # The assessments table is the core clinical record.
    # Every field maps to a specific output from the Gemma 4 assessment pipeline:
    # grade and grade_label capture severity, confidence captures model certainty,
    # urgency and referral_recommended drive the CHW's next action,
    # and full_report stores the complete JSON result for full auditability.
    # referral_recommended is stored as INTEGER (0/1) — SQLite has no boolean type.
    c.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT NOT NULL,
            date TEXT NOT NULL,
            body_site TEXT,
            rigidity_grade INTEGER,
            grade_label TEXT,
            confidence TEXT,
            urgency TEXT,
            referral_recommended INTEGER,
            full_report TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_assessment(patient_code, result):
    # Persists a validated assessment result to the database immediately after
    # inference. This is called in app.py before any UI formatting — ensuring
    # the record is saved even if a display error occurs downstream.
    #
    # Only validated results reach this function — engine.py's validation loop
    # guarantees the result dict has all required fields before returning it.
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO assessments 
        (patient_code, date, body_site, rigidity_grade, grade_label,
         confidence, urgency, referral_recommended, full_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_code,
        datetime.now().strftime("%Y-%m-%d %H:%M"), # Human-readable timestamp for CHW display
        result.get("body_site"),
        result.get("rigidity_grade"),
        result.get("grade_label"),
        result.get("confidence"),
        result.get("urgency"),
        1 if result.get("referral_recommended") else 0, # Boolean → INTEGER for SQLite
        str(result)  # Full JSON result stored as string for complete audit trail
    ))
    conn.commit()
    conn.close()


def get_patient_history(patient_code):
    # Retrieves the 10 most recent assessments for a given patient code,
    # ordered newest first. Used in two places:
    #
    # 1. In engine.py — the last 3 records are injected into Gemma 4's context
    #    so the model can comment on disease progression across visits.
    # 2. In app.py — the full 10-record list is displayed in the UI history table
    #    so the CHW can review the patient's trajectory at a glance.
    #
    # LIMIT 10 keeps the query fast on constrained hardware and the UI readable.
    # The engine.py history context further slices to [-3:] from this result.
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT date, body_site, rigidity_grade, urgency
        FROM assessments
        WHERE patient_code = ?
        ORDER BY date DESC
        LIMIT 10
    """, (patient_code,))
    rows = c.fetchall()
    conn.close()

    # Return as a list of dicts rather than raw tuples — makes field access
    # explicit and readable throughout the rest of the codebase.
    return [
        {"date": r[0], "body_site": r[1],
         "grade": r[2], "urgency": r[3]}
        for r in rows
    ]


def get_all_patients():
    # Returns a deduplicated, alphabetically sorted list of all patient codes
    # that have at least one assessment on record.
    # Currently used for administrative overview — could power a future
    # dashboard showing caseload per CHW or facility-level reporting.
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT patient_code FROM assessments ORDER BY patient_code")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]
