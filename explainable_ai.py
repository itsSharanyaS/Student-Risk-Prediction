
import streamlit as st
import pandas as pd
import plotly.express as px


def analyze_student_risk(attendance, marks, assignments):

    attendance_risk = max(0, 100 - attendance)
    marks_risk = max(0, 100 - marks)
    assignment_risk = max(0, 100 - assignments)

    factors = {
        "Attendance": attendance_risk * 0.40,
        "Marks": marks_risk * 0.40,
        "Assignments": assignment_risk * 0.20
    }

    total_risk = sum(factors.values())

    if attendance < 50 or marks < 35:
        risk_level = "Critical"
    elif attendance < 60 or marks < 50 or assignments < 50:
        risk_level = "High"
    elif attendance < 80 or marks < 75 or assignments < 75:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return factors, total_risk, risk_level


def generate_explanation(attendance, marks, assignments, risk_level, factors):

    issues = []
    recommendations = []

    if attendance < 75:
        issues.append(f"Attendance is low at {attendance:.1f}%")
        recommendations.append("Improve attendance and attend classes regularly.")

    if marks < 50:
        issues.append(f"Academic marks are low at {marks:.1f}%")
        recommendations.append("Focus on weak subjects and practice previous question papers.")

    elif marks < 75:
        issues.append(f"Marks are below the target at {marks:.1f}%")
        recommendations.append("Improve subject understanding and revise regularly.")

    if assignments < 50:
        issues.append(f"Assignment completion is low at {assignments:.1f}%")
        recommendations.append("Complete pending assignments as soon as possible.")

    elif assignments < 75:
        issues.append(f"Assignment performance needs improvement at {assignments:.1f}%")
        recommendations.append("Submit assignments on time.")

    if not issues:
        issues.append("The student's academic indicators are performing well.")
        recommendations.append("Maintain consistency in attendance and academics.")

    highest_factor = max(factors, key=factors.get)

    explanation = (
        f"The student is classified as {risk_level} Risk. "
        f"The most significant contributing factor based on the weighted "
        f"academic risk score is {highest_factor}. "
        f"Attendance is {attendance:.1f}%, marks are {marks:.1f}%, "
        f"and assignment performance is {assignments:.1f}%."
    )

    return issues, recommendations, explanation, highest_factor


def render_explainable_ai(students, apply_chart_theme, risk_colors):

    st.title("🧠 Explainable AI")
    st.caption(
        "Understand why students are classified into different academic risk levels."
    )

    if students.empty:
        st.warning("No student data available.")
        return

    student_ids = students["Student ID"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids,
        key="explainable_student_selector"
    )

    student = students[
        students["Student ID"].astype(str) == selected_id
    ].iloc[0]

    student_name = student["Student"]
    attendance = float(student["Attendance"])
    marks = float(student["Marks"])
    assignments = float(student["Assignments"])

    factors, total_risk, risk_level = analyze_student_risk(
        attendance, marks, assignments
    )

    issues, recommendations, explanation, highest_factor = generate_explanation(
        attendance, marks, assignments, risk_level, factors
    )

    st.divider()

    st.subheader(f"Risk Explanation for {student_name}")

    # Academic indicators
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Attendance", f"{attendance:.1f}%")
    col2.metric("Marks", f"{marks:.1f}%")
    col3.metric("Assignments", f"{assignments:.1f}%")
    col4.metric("Risk Level", risk_level)

    st.divider()

    # Main explanation
    st.subheader("Why is this student at risk?")

    st.info(explanation)

    # Contributing factors
    st.subheader("Risk Factor Contribution")

    factor_df = pd.DataFrame({
        "Factor": list(factors.keys()),
        "Weighted Risk Contribution": list(factors.values())
    })

    fig = px.bar(
        factor_df,
        x="Factor",
        y="Weighted Risk Contribution",
        color="Factor",
        title="Contribution of Academic Factors to Risk"
    )

    fig = apply_chart_theme(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Primary Risk Factor", highest_factor)

    st.caption(
        "Higher contribution values indicate a larger contribution to the "
        "weighted academic risk score."
    )

    st.divider()

    # Issues
    st.subheader("Identified Academic Concerns")

    for issue in issues:
        st.warning(issue)

    # Recommendations
    st.subheader("Recommended Improvement Actions")

    for index, recommendation in enumerate(recommendations, start=1):
        st.write(f"{index}. {recommendation}")

    st.divider()

    # Academic risk score
    st.subheader("Weighted Academic Risk Score")

    st.progress(min(total_risk / 100, 1.0))

    st.metric(
        "Risk Score",
        f"{total_risk:.2f}/100"
    )

    st.caption(
        "This score is based on weighted attendance, marks, and assignment "
        "performance indicators. It is not the probability generated by the ML model."
    )

    st.divider()

    # Improvement targets
    st.subheader("Academic Improvement Targets")

    target_data = pd.DataFrame({
        "Academic Factor": ["Attendance", "Marks", "Assignments"],
        "Current Value": [attendance, marks, assignments],
        "Target Value": [75, 75, 75]
    })

    fig_target = px.bar(
        target_data,
        x="Academic Factor",
        y=["Current Value", "Target Value"],
        barmode="group",
        title="Current Performance vs Improvement Target"
    )

    fig_target.update_layout(yaxis_range=[0, 100])

    fig_target = apply_chart_theme(fig_target)

    st.plotly_chart(fig_target, use_container_width=True)

    st.success(
        f"AI analysis completed for {student_name}. "
        f"Priority area: {highest_factor}."
    )

