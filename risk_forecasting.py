
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# =========================================================
# 1. RISK CLASSIFICATION
# =========================================================

def forecast_risk(attendance, marks, assignments):

    def get_risk(a, m, ass):
        if a < 50 or m < 35:
            return "Critical"
        elif a < 60 or m < 50 or ass < 50:
            return "High"
        elif a < 80 or m < 75 or ass < 75:
            return "Moderate"
        return "Low"

    current_risk = get_risk(attendance, marks, assignments)

    # Simulated academic decline scenario
    future_attendance = max(0, attendance - 10)
    future_marks = max(0, marks - 8)
    future_assignments = max(0, assignments - 10)

    predicted_risk = get_risk(
        future_attendance,
        future_marks,
        future_assignments
    )

    return {
        "current_risk": current_risk,
        "predicted_risk": predicted_risk,
        "future_attendance": future_attendance,
        "future_marks": future_marks,
        "future_assignments": future_assignments
    }


# =========================================================
# 2. COLUMN DETECTION
# =========================================================

def get_column(df, possible_names):

    normalized_columns = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in df.columns
    }

    for name in possible_names:
        normalized_name = (
            name.strip().lower().replace("_", " ").replace("-", " ")
        )

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


# =========================================================
# 3. RISK FORECASTING PAGE
# =========================================================

def render_risk_forecasting(students, apply_chart_theme, risk_colors):

    st.title("Academic Risk Forecasting")

    st.caption(
        "Explore simulated academic performance changes and identify "
        "students who may require early intervention."
    )

    st.info(
        "This tool creates a simulated academic decline scenario using "
        "attendance, marks and assignment performance. It is not a real "
        "machine-learning prediction of future performance."
    )

    # -----------------------------------------------------
    # Empty DataFrame protection
    # -----------------------------------------------------

    if students is None or students.empty:
        st.warning("No student records are currently available.")
        st.info("Add student records in Student Management to use forecasting.")
        return

    # -----------------------------------------------------
    # Detect column names
    # -----------------------------------------------------

    id_col = get_column(
        students,
        ["Student ID", "student_id", "USN", "StudentID"]
    )

    name_col = get_column(
        students,
        ["Student", "Name", "Student Name"]
    )

    attendance_col = get_column(
        students,
        ["Attendance", "attendance"]
    )

    marks_col = get_column(
        students,
        ["Marks", "mark", "Score", "scores"]
    )

    assignments_col = get_column(
        students,
        ["Assignments", "Assignment", "assignment"]
    )

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    missing_columns = []

    if id_col is None:
        missing_columns.append("Student ID")

    if attendance_col is None:
        missing_columns.append("Attendance")

    if marks_col is None:
        missing_columns.append("Marks")

    if assignments_col is None:
        missing_columns.append("Assignments")

    if missing_columns:

        st.error(
            "Required student data columns are missing: "
            + ", ".join(missing_columns)
        )

        st.write("Available columns:", list(students.columns))
        return

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------

    students = students.copy()

    students[id_col] = students[id_col].astype(str)

    # Remove records with missing academic data
    students = students.dropna(
        subset=[
            id_col,
            attendance_col,
            marks_col,
            assignments_col
        ]
    )

    if students.empty:
        st.warning("No complete student records available for forecasting.")
        return

    # -----------------------------------------------------
    # Student selection
    # -----------------------------------------------------

    student_ids = students[id_col].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids,
        key="forecast_student"
    )

    student = students[
        students[id_col].astype(str) == str(selected_id)
    ].iloc[0]

    # -----------------------------------------------------
    # Student information
    # -----------------------------------------------------

    if name_col is not None:

        student_name = student[name_col]

        st.markdown(
            f"### Student: {student_name}"
        )

    st.caption(f"Student ID: {selected_id}")

    # -----------------------------------------------------
    # Academic values
    # -----------------------------------------------------

    attendance = float(student[attendance_col])
    marks = float(student[marks_col])
    assignments = float(student[assignments_col])

    result = forecast_risk(
        attendance,
        marks,
        assignments
    )

    # -----------------------------------------------------
    # KPI metrics
    # -----------------------------------------------------

    st.subheader("Forecast Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Risk",
        result["current_risk"]
    )

    col2.metric(
        "Forecast Risk",
        result["predicted_risk"]
    )

    col3.metric(
        "Current Attendance",
        f"{attendance:.1f}%"
    )

    col4.metric(
        "Forecast Attendance",
        f'{result["future_attendance"]:.1f}%'
    )

    # -----------------------------------------------------
    # Current vs forecast comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame({
        "Metric": [
            "Attendance",
            "Marks",
            "Assignments"
        ],
        "Current": [
            attendance,
            marks,
            assignments
        ],
        "Forecast": [
            result["future_attendance"],
            result["future_marks"],
            result["future_assignments"]
        ]
    })

    st.subheader("Current vs Forecast Performance")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Current Performance",
            x=comparison["Metric"],
            y=comparison["Current"],
            text=[
                f"{value:.1f}%"
                for value in comparison["Current"]
            ],
            textposition="outside"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Forecast Scenario",
            x=comparison["Metric"],
            y=comparison["Forecast"],
            text=[
                f"{value:.1f}%"
                for value in comparison["Forecast"]
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        title="Academic Performance Scenario",
        barmode="group",
        xaxis_title="Academic Indicator",
        yaxis_title="Percentage",
        yaxis_range=[0, 110],
        height=430,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    apply_chart_theme(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Forecast interpretation
    # -----------------------------------------------------

    st.subheader("Forecast Interpretation")

    risk = result["predicted_risk"]

    if risk == "Critical":

        st.error(
            "Critical scenario: the simulated academic decline "
            "places this student in a critical risk category."
        )

    elif risk == "High":

        st.warning(
            "High-risk scenario: the student may require "
            "immediate academic attention."
        )

    elif risk == "Moderate":

        st.info(
            "Moderate-risk scenario: monitoring and preventive "
            "academic support are recommended."
        )

    else:

        st.success(
            "Low-risk scenario: the student is projected to "
            "maintain comparatively stable academic performance."
        )

    # -----------------------------------------------------
    # Academic indicator comparison
    # -----------------------------------------------------

    st.subheader("Simulated Academic Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Attendance",
        f'{result["future_attendance"]:.1f}%',
        delta=f'{result["future_attendance"] - attendance:.1f}%'
    )

    col2.metric(
        "Marks",
        f'{result["future_marks"]:.1f}%',
        delta=f'{result["future_marks"] - marks:.1f}%'
    )

    col3.metric(
        "Assignments",
        f'{result["future_assignments"]:.1f}%',
        delta=f'{result["future_assignments"] - assignments:.1f}%'
    )

    # -----------------------------------------------------
    # Preventive recommendations
    # -----------------------------------------------------

    st.subheader("Recommended Preventive Actions")

    recommendations = []

    if result["future_attendance"] < 75:

        recommendations.append(
            "Improve attendance through regular class participation "
            "and attendance monitoring."
        )

    if result["future_marks"] < 50:

        recommendations.append(
            "Schedule academic revision sessions and subject-wise mentoring."
        )

    if result["future_assignments"] < 50:

        recommendations.append(
            "Complete pending assignments and arrange faculty guidance."
        )

    if result["predicted_risk"] in ["High", "Critical"]:

        recommendations.append(
            "Arrange an individual student counselling session "
            "and create a structured recovery plan."
        )

    if not recommendations:

        recommendations.append(
            "Maintain current academic consistency and continue "
            "periodic monitoring."
        )

    for recommendation in recommendations:

        st.markdown(
            f"- {recommendation}"
        )

    # -----------------------------------------------------
    # Forecast disclaimer
    # -----------------------------------------------------

    st.divider()

    st.caption(
        "Disclaimer: Forecast values are generated using a simulated "
        "reduction in attendance, marks and assignments. These values "
        "are intended for academic demonstration and decision support "
        "only. They do not represent actual future student performance."
    )
