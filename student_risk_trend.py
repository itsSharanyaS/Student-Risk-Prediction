
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# STUDENT RISK TREND ANALYZER
# =========================================================

def calculate_risk(attendance, marks, assignments):

    score = 0

    if attendance < 50:
        score += 40
    elif attendance < 60:
        score += 30
    elif attendance < 75:
        score += 15

    if marks < 35:
        score += 35
    elif marks < 50:
        score += 25
    elif marks < 65:
        score += 15

    if assignments < 40:
        score += 25
    elif assignments < 60:
        score += 15
    elif assignments < 75:
        score += 10

    score = min(score, 100)

    if attendance < 50 or marks < 35:
        risk = "Critical"
    elif attendance < 60 or marks < 50 or assignments < 50:
        risk = "High"
    elif attendance < 80 or marks < 75 or assignments < 75:
        risk = "Moderate"
    else:
        risk = "Low"

    return score, risk


# =========================================================
# SIMULATED HISTORICAL DATA
# =========================================================

def generate_student_history(student):

    student_id = str(student["Student ID"])

    rng = np.random.default_rng(
        sum(ord(char) for char in student_id)
    )

    current_attendance = float(student["Attendance"])
    current_marks = float(student["Marks"])
    current_assignments = float(student["Assignments"])

    periods = [
        "Previous 4 Months",
        "Previous 3 Months",
        "Previous 2 Months",
        "Previous Month",
        "Current"
    ]

    attendance_history = []
    marks_history = []
    assignment_history = []

    # Generate simulated changes
    attendance_change = rng.uniform(-12, 12)
    marks_change = rng.uniform(-15, 15)
    assignment_change = rng.uniform(-15, 15)

    for i in range(4):

        progress = (i + 1) / 4

        attendance = current_attendance - attendance_change * (1 - progress)
        marks = current_marks - marks_change * (1 - progress)
        assignments = current_assignments - assignment_change * (1 - progress)

        attendance_history.append(
            round(np.clip(attendance, 0, 100), 1)
        )

        marks_history.append(
            round(np.clip(marks, 0, 100), 1)
        )

        assignment_history.append(
            round(np.clip(assignments, 0, 100), 1)
        )

    # Current values are real values from the dataset
    attendance_history.append(current_attendance)
    marks_history.append(current_marks)
    assignment_history.append(current_assignments)

    history = pd.DataFrame({
        "Period": periods,
        "Attendance": attendance_history,
        "Marks": marks_history,
        "Assignments": assignment_history
    })

    history["Risk Score"] = history.apply(
        lambda row: calculate_risk(
            row["Attendance"],
            row["Marks"],
            row["Assignments"]
        )[0],
        axis=1
    )

    history["Risk Level"] = history.apply(
        lambda row: calculate_risk(
            row["Attendance"],
            row["Marks"],
            row["Assignments"]
        )[1],
        axis=1
    )

    return history


# =========================================================
# TREND DIRECTION
# =========================================================

def get_trend(first_value, last_value):

    difference = last_value - first_value

    if difference >= 5:
        return "Improving", difference

    elif difference <= -5:
        return "Declining", difference

    else:
        return "Stable", difference


# =========================================================
# MAIN RENDER FUNCTION
# =========================================================

def render_student_risk_trend(
    students,
    apply_chart_theme,
    risk_colors
):

    st.title("📈 Student Risk Trend Analyzer")

    st.markdown(
        """
        Analyze academic progress, performance changes and risk
        patterns across simulated historical periods.
        """
    )

    st.warning(
        "⚠️ Historical data in this module is simulated. "
        "Only the Current values are taken from the actual student dataset."
    )

    # -----------------------------------------------------
    # STUDENT SELECTION
    # -----------------------------------------------------

    student_options = students["Student"].tolist()

    selected_student = st.selectbox(
        "Select Student",
        student_options,
        key="trend_student_selector"
    )

    student = students[
        students["Student"] == selected_student
    ].iloc[0]

    history = generate_student_history(student)

    # -----------------------------------------------------
    # STUDENT PROFILE
    # -----------------------------------------------------

    st.markdown("### 🎓 Student Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Student Name", student["Student"])

    with col2:
        st.metric("Student ID", student["Student ID"])

    with col3:
        st.metric("Department", student["Department"])

    with col4:
        st.metric("Semester", student["Semester"])

    st.divider()

    # -----------------------------------------------------
    # TREND ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 📊 Academic Trend Overview")

    attendance_trend, attendance_change = get_trend(
        history["Attendance"].iloc[0],
        history["Attendance"].iloc[-1]
    )

    marks_trend, marks_change = get_trend(
        history["Marks"].iloc[0],
        history["Marks"].iloc[-1]
    )

    assignment_trend, assignment_change = get_trend(
        history["Assignments"].iloc[0],
        history["Assignments"].iloc[-1]
    )

    risk_trend, risk_change = get_trend(
        history["Risk Score"].iloc[0],
        history["Risk Score"].iloc[-1]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Attendance Trend",
            attendance_trend,
            delta=f"{attendance_change:+.1f}%"
        )

    with col2:
        st.metric(
            "Marks Trend",
            marks_trend,
            delta=f"{marks_change:+.1f}"
        )

    with col3:
        st.metric(
            "Assignment Trend",
            assignment_trend,
            delta=f"{assignment_change:+.1f}%"
        )

    with col4:
        st.metric(
            "Risk Score Change",
            risk_trend,
            delta=f"{risk_change:+.1f}",
            delta_color="inverse"
        )

    st.divider()

    # -----------------------------------------------------
    # PERFORMANCE LINE CHART
    # -----------------------------------------------------

    st.markdown("### 📈 Academic Performance History")

    performance_data = history.melt(
        id_vars="Period",
        value_vars=["Attendance", "Marks", "Assignments"],
        var_name="Metric",
        value_name="Value"
    )

    fig = px.line(
        performance_data,
        x="Period",
        y="Value",
        color="Metric",
        markers=True,
        title="Student Academic Performance Trend"
    )

    fig.update_layout(
        yaxis_title="Performance (%)",
        yaxis_range=[0, 100],
        xaxis_title="Academic Timeline",
        hovermode="x unified"
    )

    apply_chart_theme(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # RISK SCORE TREND
    # -----------------------------------------------------

    st.markdown("### 🚨 Academic Risk Trend")

    risk_fig = px.area(
        history,
        x="Period",
        y="Risk Score",
        markers=True,
        title="Academic Risk Score Over Time"
    )

    risk_fig.update_layout(
        yaxis_title="Risk Score",
        yaxis_range=[0, 100],
        xaxis_title="Academic Timeline"
    )

    apply_chart_theme(risk_fig)

    st.plotly_chart(
        risk_fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # RISK LEVEL HISTORY
    # -----------------------------------------------------

    st.markdown("### 📋 Risk Level History")

    risk_table = history[
        ["Period", "Attendance", "Marks", "Assignments",
         "Risk Score", "Risk Level"]
    ].copy()

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # ACADEMIC INSIGHTS
    # -----------------------------------------------------

    st.markdown("### 🧠 Academic Insights")

    insights = []

    if attendance_trend == "Declining":
        insights.append(
            "⚠️ Attendance is declining. Regular class attendance "
            "should be encouraged."
        )

    elif attendance_trend == "Improving":
        insights.append(
            "✅ Attendance is improving. Continue maintaining "
            "regular attendance."
        )

    if marks_trend == "Declining":
        insights.append(
            "⚠️ Marks are declining. Academic support and revision "
            "sessions are recommended."
        )

    elif marks_trend == "Improving":
        insights.append(
            "✅ Marks are improving. Continue the current study routine."
        )

    if assignment_trend == "Declining":
        insights.append(
            "⚠️ Assignment performance is declining. "
            "Monitor submission deadlines."
        )

    elif assignment_trend == "Improving":
        insights.append(
            "✅ Assignment performance is improving."
        )

    if risk_change > 0:
        insights.append(
            "🚨 The academic risk score has increased. "
            "The student may require early intervention."
        )

    elif risk_change < 0:
        insights.append(
            "🟢 The academic risk score has decreased. "
            "The student's overall academic risk has improved."
        )

    if not insights:
        insights.append(
            "➡️ Academic performance is relatively stable "
            "across the analyzed indicators."
        )

    for insight in insights:
        st.info(insight)

    # -----------------------------------------------------
    # FACULTY SUMMARY
    # -----------------------------------------------------

    st.markdown("### 👨‍🏫 Faculty Summary")

    current_risk = history["Risk Level"].iloc[-1]
    current_score = history["Risk Score"].iloc[-1]

    if current_risk in ["Critical", "High"]:
        st.error(
            f"{selected_student} currently has a {current_risk} "
            f"risk level with a risk score of {current_score}/100. "
            "Immediate academic monitoring and personalized support "
            "are recommended."
        )

    elif current_risk == "Moderate":
        st.warning(
            f"{selected_student} currently has a Moderate risk level "
            f"with a risk score of {current_score}/100. "
            "Regular monitoring and academic guidance are recommended."
        )

    else:
        st.success(
            f"{selected_student} currently has a Low risk level "
            f"with a risk score of {current_score}/100. "
            "Continue monitoring academic progress."
        )

    st.caption(
        "Trend analysis is based on simulated historical data "
        "and is intended for project demonstration purposes."
    )

