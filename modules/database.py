
import sqlite3
from pathlib import Path
import pandas as pd


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DB_PATH = Path("data/student_risk.db")


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    return sqlite3.connect(DB_PATH)


# =========================================================
# INITIALIZE DATABASE TABLES
# =========================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    # STUDENTS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT,
            department TEXT,
            semester INTEGER,
            attendance REAL,
            marks REAL,
            assignments REAL,
            risk_level TEXT
        )
    """)

    # INTERVENTIONS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            intervention_type TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(student_id)
        )
    """)

    # USERS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            role TEXT
        )
    """)

    conn.commit()
    conn.close()


# =========================================================
# SAVE STUDENTS
# =========================================================

def save_students(students):

    initialize_database()

    conn = get_connection()

    students.to_sql(
        "students",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()


# =========================================================
# LOAD STUDENTS
# =========================================================

def load_students():

    initialize_database()

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM students",
        conn
    )

    conn.close()

    return df


# =========================================================
# ADD INTERVENTION
# =========================================================

def add_intervention(
    student_id,
    intervention_type,
    notes
):

    initialize_database()

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interventions
        (student_id, intervention_type, notes)
        VALUES (?, ?, ?)
    """, (
        student_id,
        intervention_type,
        notes
    ))

    conn.commit()
    conn.close()


# =========================================================
# LOAD INTERVENTIONS
# =========================================================

def load_interventions():

    initialize_database()

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM interventions ORDER BY created_at DESC",
        conn
    )

    conn.close()

    return df


# =========================================================
# DATABASE STATISTICS
# =========================================================

def get_database_stats():

    initialize_database()

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM students")
    student_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM interventions")
    intervention_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]

    conn.close()

    return {
        "students": student_count,
        "interventions": intervention_count,
        "users": user_count
    }

