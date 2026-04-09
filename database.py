import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "rigidityiq_patients.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT NOT NULL,
            age INTEGER,
            created_at TEXT
        )
    """)
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO assessments 
        (patient_code, date, body_site, rigidity_grade, grade_label,
         confidence, urgency, referral_recommended, full_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_code,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        result.get("body_site"),
        result.get("rigidity_grade"),
        result.get("grade_label"),
        result.get("confidence"),
        result.get("urgency"),
        1 if result.get("referral_recommended") else 0,
        str(result)
    ))
    conn.commit()
    conn.close()


def get_patient_history(patient_code):
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
    return [
        {"date": r[0], "body_site": r[1],
         "grade": r[2], "urgency": r[3]}
        for r in rows
    ]


def get_all_patients():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT patient_code FROM assessments ORDER BY patient_code")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]