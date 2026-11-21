
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime


# =========================================================
# STUDENT RECOVERY SUCCESS TRACKER
# =========================================================

DATA_FILE = Path("data/recovery_history.csv")

METRICS = ["Attendance", "Marks", "Assignments"]


# =========================================================
# DATA STORAGE
# =========================================================

def load_recovery_history():

    if DATA_FILE.exists():
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame(columns=[
        "Student ID", "Student", "Attendance",
        "Marks", "Assignments", "Review Date"
    ])


def save_recovery_history(df):

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)


# =========================================================
# PERFORMANCE ANALYSIS
# =========================================================

def compare_student_progress(history, student_id):

    records = history[
        history["Student ID"].astype(str) == str(student_id)
    ].copy()

    if records.empty or len(records) < 2:
        return None

    records["Review Date"] = pd.to_datetime(
        records["Review Date"], errors="coerce"
    )

    records = records.sort_values("Review Date")

    first = records.iloc[0]
    latest = records.iloc[-1]

    changes = {}

    for metric in METRICS:
        changes[metric] = round(
            float(latest[metric]) - float(first[metric]), 2
        )

    avg_change = sum(changes.values()) / len(changes)

    if avg_change >= 5:
        status = "Improving"

    elif avg_change <= -5:
        status = "Declining"

    else:
        status = "Stable"

    return {
        "First Review": first,
        "Latest Review": latest,
        "Changes": changes,
        "Average Improvement": round(avg_change, 2),
        "Status": status,
        "Review Count": len(records)
    }


# =========================================================
# MAIN UI
# =========================================================

def render_recovery_success_tracker(students):

    st.title("📈 Student Recovery Success Tracker")

    st.caption(
        "Monitor student improvement after academic interventions "
        "using historical performance reviews."
    )

    if students is None or students.empty:
        st.warning("No student data available.")
        return

    history = load_recovery_history()

    # -----------------------------------------------------
    # STUDENT SELECTION
    # -----------------------------------------------------

    st.subheader("👤 Select Student")

    selected_student = st.selectbox(
        "Choose a student",
        students["Student"].tolist(),
        key="success_tracker_student"
    )

    student = students[
        students["Student"] == selected_student
    ].iloc[0]

    student_id = str(student["Student ID"])

    # -----------------------------------------------------
    # RECORD NEW REVIEW
    # -----------------------------------------------------

    st.divider()

    st.subheader("📝 Record Recovery Review")

    st.caption(
        "Enter the student's latest performance after intervention. "
        "Each review is saved as a historical record."
    )

    with st.form("recovery_review_form"):

        col1, col2, col3 = st.columns(3)

        attendance = col1.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(student["Attendance"]),
            step=1.0
        )

        marks = col2.number_input(
            "Marks (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(student["Marks"]),
            step=1.0
        )

        assignments = col3.number_input(
            "Assignments (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(student["Assignments"]),
            step=1.0
        )

        review_date = st.date_input(
            "Review Date",
            value=datetime.today().date()
        )

        submitted = st.form_submit_button(
            "💾 Save Recovery Review",
            use_container_width=True
        )

    if submitted:

        new_record = pd.DataFrame([{
            "Student ID": student_id,
            "Student": selected_student,
            "Attendance": attendance,
            "Marks": marks,
            "Assignments": assignments,
            "Review Date": review_date.strftime("%Y-%m-%d")
        }])

        history = load_recovery_history()

        history = pd.concat(
            [history, new_record],
            ignore_index=True
        )

        save_recovery_history(history)

        st.success("Recovery review saved successfully!")

        st.rerun()

    # -----------------------------------------------------
    # STUDENT HISTORY
    # -----------------------------------------------------

    history = load_recovery_history()

    student_history = history[
        history["Student ID"].astype(str) == student_id
    ].copy()

    if student_history.empty:

        st.info(
            "No recovery history recorded yet. "
            "Save the student's first review above to begin tracking."
        )

        st.markdown("### 📌 Current Academic Baseline")

        col1, col2, col3 = st.columns(3)

        col1.metric("Attendance", f"{student['Attendance']}%")
        col2.metric("Marks", f"{student['Marks']}%")
        col3.metric("Assignments", f"{student['Assignments']}%")

        return

    student_history["Review Date"] = pd.to_datetime(
        student_history["Review Date"], errors="coerce"
    )

    student_history = student_history.sort_values("Review Date")

    # -----------------------------------------------------
    # PROGRESS ANALYSIS
    # -----------------------------------------------------

    analysis = compare_student_progress(history, student_id)

    st.divider()

    st.subheader("📊 Recovery Performance Overview")

    if analysis is None:

        st.info(
            "Record at least two recovery reviews to see improvement analysis."
        )

        st.dataframe(
            student_history,
            use_container_width=True,
            hide_index=True
        )

        return

    first = analysis["First Review"]
    latest = analysis["Latest Review"]
    changes = analysis["Changes"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Recovery Status",
        analysis["Status"]
    )

    col2.metric(
        "Average Improvement",
        f"{analysis['Average Improvement']:+.1f}"
    )

    col3.metric(
        "Review Records",
        analysis["Review Count"]
    )

    col4.metric(
        "Marks Improvement",
        f"{changes['Marks']:+.1f}%"
    )

    # -----------------------------------------------------
    # STATUS MESSAGE
    # -----------------------------------------------------

    if analysis["Status"] == "Improving":

        st.success(
            "🎉 Student performance is improving. "
            "Continue the current recovery strategy and monitor progress."
        )

    elif analysis["Status"] == "Declining":

        st.error(
            "⚠️ Student performance is declining. "
            "Review the current intervention and consider additional support."
        )

    else:

        st.warning(
            "📊 Student performance is relatively stable. "
            "Continue monitoring and identify areas for further improvement."
        )

    # -----------------------------------------------------
    # BEFORE VS AFTER
    # -----------------------------------------------------

    st.subheader("🔄 Initial Review vs Latest Review")

    comparison = pd.DataFrame({
        "Metric": METRICS,
        "Initial Review": [
            first[m] for m in METRICS
        ],
        "Latest Review": [
            latest[m] for m in METRICS
        ]
    })

    comparison_chart = comparison.melt(
        id_vars="Metric",
        var_name="Review",
        value_name="Score"
    )

    fig = px.bar(
        comparison_chart,
        x="Metric",
        y="Score",
        color="Review",
        barmode="group",
        title="Academic Performance Comparison",
        text_auto=True
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # INDIVIDUAL IMPROVEMENT
    # -----------------------------------------------------

    st.subheader("📈 Indicator-wise Improvement")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Attendance Change",
        f"{changes['Attendance']:+.1f}%",
        delta=f"{changes['Attendance']:+.1f}"
    )

    col2.metric(
        "Marks Change",
        f"{changes['Marks']:+.1f}%",
        delta=f"{changes['Marks']:+.1f}"
    )

    col3.metric(
        "Assignment Change",
        f"{changes['Assignments']:+.1f}%",
        delta=f"{changes['Assignments']:+.1f}"
    )

    # -----------------------------------------------------
    # PERFORMANCE TREND
    # -----------------------------------------------------

    st.subheader("📅 Recovery Trend")

    trend_chart = student_history.melt(
        id_vars=["Review Date"],
        value_vars=METRICS,
        var_name="Indicator",
        value_name="Score"
    )

    trend_fig = px.line(
        trend_chart,
        x="Review Date",
        y="Score",
        color="Indicator",
        markers=True,
        title="Student Performance Across Reviews"
    )

    trend_fig.update_layout(
        yaxis_range=[0, 100],
        height=400
    )

    st.plotly_chart(
        trend_fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # REVIEW HISTORY
    # -----------------------------------------------------

    st.subheader("📋 Recovery Review History")

    st.dataframe(
        student_history[
            ["Review Date", "Attendance", "Marks", "Assignments"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------------------------

    st.divider()

    report = f"""
STUDENT RECOVERY SUCCESS REPORT
==============================

Student: {selected_student}
Student ID: {student_id}

Recovery Status: {analysis['Status']}
Average Improvement: {analysis['Average Improvement']}

INITIAL PERFORMANCE
-----------------
Attendance: {first['Attendance']}%
Marks: {first['Marks']}%
Assignments: {first['Assignments']}%

LATEST PERFORMANCE
-----------------
Attendance: {latest['Attendance']}%
Marks: {latest['Marks']}%
Assignments: {latest['Assignments']}%

INDICATOR IMPROVEMENT
----------------------
Attendance: {changes['Attendance']:+.2f}
Marks: {changes['Marks']:+.2f}
Assignments: {changes['Assignments']:+.2f}

Total Review Records: {analysis['Review Count']}
"""

    st.download_button(
        "📥 Download Recovery Report",
        data=report,
        file_name=f"{student_id}_recovery_success_report.txt",
        mime="text/plain"
    )
