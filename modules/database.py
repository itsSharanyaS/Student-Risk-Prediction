import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

DB_PATH = Path("data/student_risk.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                student_name TEXT,
                department TEXT,
                semester INTEGER,
                attendance REAL,
                marks REAL,
                assignments REAL,
                risk_level TEXT,
                risk_score REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                intervention_type TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(student_id) REFERENCES students(student_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL
            )
        """)


def _first_value(row, *names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default



def save_students(students: pd.DataFrame):
    """Synchronize the current dataframe into the students table without dropping the table schema."""
    initialize_database()
    records = []
    for _, row in students.iterrows():
        student_id = str(_first_value(row, "Student ID", "student_id", "USN", default="")).strip()
        if not student_id:
            continue
        records.append((
            student_id,
            str(_first_value(row, "Student", "Student Name", "student_name", default="")),
            str(_first_value(row, "Department", "department", default="CSE")),
            int(_first_value(row, "Semester", "semester", default=0) or 0),
            float(_first_value(row, "Attendance", "attendance", default=0) or 0),
            float(_first_value(row, "Marks", "marks", default=0) or 0),
            float(_first_value(row, "Assignments", "assignments", default=0) or 0),
            str(_first_value(row, "Risk", "Risk Level", "risk_level", default="Unknown")),
            float(_first_value(row, "Risk Score", "risk_score", default=0) or 0),
        ))
    with get_connection() as conn:
        conn.execute("DELETE FROM students")
        conn.executemany("""\n            INSERT INTO students
            (student_id, student_name, department, semester, attendance, marks, assignments, risk_level, risk_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
    return len(records)



def load_students():
    initialize_database()
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM students ORDER BY student_id", conn)



def add_intervention(student_id, intervention_type, notes):
    initialize_database()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO interventions (student_id, intervention_type, notes)
            VALUES (?, ?, ?)
        """, (str(student_id), intervention_type, notes))



def load_interventions():
    initialize_database()
    with get_connection() as conn:
        return pd.read_sql_query("""\n            SELECT i.intervention_id, i.student_id, s.student_name, i.intervention_type,
                   i.notes, i.created_at
            FROM interventions i LEFT JOIN students s ON s.student_id = i.student_id
            ORDER BY i.created_at DESC
        """, conn)



def get_database_stats():
    initialize_database()
    with get_connection() as conn:
        student_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        intervention_count = conn.execute("SELECT COUNT(*) FROM interventions").fetchone()[0]
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return {"students": student_count, "interventions": intervention_count, "users": user_count}



def render_database_dashboard(students):
    """Admin dashboard for syncing and inspecting the SQLite database."""
    st.title("🗄️ Database Integration & Management")
    st.caption("SQLite storage • Student records • Intervention history")
    initialize_database()

    stats = get_database_stats()
    a, b, c = st.columns(3)
    a.metric("Database Students", stats["students"])
    b.metric("Intervention Records", stats["interventions"])
    c.metric("Registered Database Users", stats["users"])

    st.subheader("CSV → SQLite Synchronization")
    st.write("Copy the current student dataset into the SQLite database. Existing student rows are replaced by the current dataset during synchronization; intervention history is retained.")
    if st.button("🔄 Sync Current Student Dataset to Database", use_container_width=True):
        count = save_students(students)
        st.success(f"Successfully synchronized {count} student records into SQLite.")
        st.rerun()

    db_students = load_students()
    st.subheader("Stored Student Records")
    if db_students.empty:
        st.info("The database has no student records yet. Click the synchronization button above to import the current dataset.")
    else:
        st.dataframe(db_students, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Database Student Records (CSV)", db_students.to_csv(index=False).encode("utf-8"), "database_students.csv", "text/csv")

    st.divider()
    st.subheader("Add Intervention Record")
    if db_students.empty:
        st.warning("Sync student data before adding intervention records.")
    else:
        with st.form("database_intervention_form"):
            chosen_id = st.selectbox("Student ID", db_students["student_id"].tolist())
            intervention_type = st.selectbox("Intervention Type", ["Mentoring", "Counselling", "Remedial Class", "Attendance Follow-up", "Parent Meeting", "Other"])
            notes = st.text_area("Notes", placeholder="Describe the support provided...")
            submitted = st.form_submit_button("Save Intervention")
        if submitted:
            add_intervention(chosen_id, intervention_type, notes)
            st.success("Intervention record saved to the database.")
            st.rerun()

    st.subheader("Intervention History")
    interventions = load_interventions()
    if interventions.empty:
        st.info("No intervention records have been stored yet.")
    else:
        st.dataframe(interventions, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Intervention History", interventions.to_csv(index=False).encode("utf-8"), "database_interventions.csv", "text/csv")

    st.divider()
    st.caption(f"Database file: {DB_PATH.as_posix()}")
    st.warning("On Streamlit Cloud, local SQLite files may not persist across app restarts or deployments. This database is suitable for the project demo; persistent hosted storage can be configured for production.")
