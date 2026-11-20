
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


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


def render_what_if_simulator(students, apply_chart_theme, risk_colors):

    st.title("🧪 Academic What-If Simulator")
    st.caption("Explore how academic improvements or declines can affect student risk.")

    st.info(
        "This is a hypothetical simulation. It does not modify original student records "
        "and is not an actual future prediction."
    )

    if students.empty:
        st.warning("No student records available.")
        return

    student_ids = students["Student ID"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids,
        key="what_if_student"
    )

    student = students[
        students["Student ID"].astype(str) == selected_id
    ].iloc[0]

    original_attendance = int(student["Attendance"])
    original_marks = int(student["Marks"])
    original_assignments = int(student["Assignments"])

    original_score, original_risk = calculate_risk(
        original_attendance,
        original_marks,
        original_assignments
    )

    st.subheader("👤 Student Profile")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Student", student["Student"])
    col2.metric("USN", selected_id)
    col3.metric("Department", student["Department"])
    col4.metric("Semester", int(student["Semester"]))

    st.subheader("📊 Original Academic Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attendance", f"{original_attendance}%")
    col2.metric("Marks", f"{original_marks}%")
    col3.metric("Assignments", f"{original_assignments}%")
    col4.metric("Risk Score", f"{original_score}/100")

    st.markdown(f"### Original Risk Level: {original_risk}")

    st.divider()
    st.subheader("🎯 Adjust Academic Performance")

    st.write(
        "Use the sliders to simulate different academic scenarios. "
        "You can experiment with improvements or declines."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        simulated_attendance = st.slider(
            "Attendance (%)",
            0, 100, original_attendance,
            key="what_if_attendance"
        )

    with col2:
        simulated_marks = st.slider(
            "Marks (%)",
            0, 100, original_marks,
            key="what_if_marks"
        )

    with col3:
        simulated_assignments = st.slider(
            "Assignments (%)",
            0, 100, original_assignments,
            key="what_if_assignments"
        )

    simulated_score, simulated_risk = calculate_risk(
        simulated_attendance,
        simulated_marks,
        simulated_assignments
    )

    score_change = original_score - simulated_score

    st.divider()
    st.subheader("🔮 Simulation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Risk Score", f"{original_score}/100")

    col2.metric(
        "Simulated Risk Score",
        f"{simulated_score}/100",
        delta=f"{simulated_score - original_score} points",
        delta_color="inverse"
    )

    col3.metric("Risk Score Reduction", f"{score_change} points")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Risk")
        st.markdown(f"## {original_risk}")

    with col2:
        st.markdown("### Simulated Risk")
        st.markdown(f"## {simulated_risk}")

    if simulated_score < original_score:
        st.success(
            f"Great! This scenario reduces the risk score by {score_change} points."
        )
    elif simulated_score > original_score:
        st.warning(
            f"This scenario increases the risk score by {abs(score_change)} points."
        )
    else:
        st.info("The risk score remains unchanged in this scenario.")

    st.subheader("📈 Academic Performance Comparison")

    comparison = pd.DataFrame({
        "Metric": ["Attendance", "Marks", "Assignments"],
        "Original": [
            original_attendance,
            original_marks,
            original_assignments
        ],
        "Simulated": [
            simulated_attendance,
            simulated_marks,
            simulated_assignments
        ]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Original",
        x=comparison["Metric"],
        y=comparison["Original"],
        text=comparison["Original"],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Simulated",
        x=comparison["Metric"],
        y=comparison["Simulated"],
        text=comparison["Simulated"],
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        yaxis_title="Percentage",
        yaxis_range=[0, 110],
        height=420
    )

    apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚦 Risk Score Comparison")

    risk_fig = go.Figure()

    risk_fig.add_trace(go.Bar(
        x=["Original", "Simulated"],
        y=[original_score, simulated_score],
        text=[original_score, simulated_score],
        textposition="outside",
        marker_color=[
            risk_colors.get(original_risk, "#64748B"),
            risk_colors.get(simulated_risk, "#64748B")
        ]
    ))

    risk_fig.update_layout(
        yaxis_title="Risk Score",
        yaxis_range=[0, 110],
        height=350,
        showlegend=False
    )

    apply_chart_theme(risk_fig)
    st.plotly_chart(risk_fig, use_container_width=True)

    st.divider()
    st.subheader("💡 Personalized Recommendations")

    recommendations = []

    if simulated_attendance < 75:
        recommendations.append("Improve attendance and attend classes consistently.")

    if simulated_marks < 65:
        recommendations.append("Focus on weak subjects and follow a regular study timetable.")

    if simulated_assignments < 75:
        recommendations.append("Complete pending assignments and submit work on time.")

    if simulated_risk in ["High", "Critical"]:
        recommendations.append("Meet the academic mentor and create an improvement plan.")

    if simulated_risk == "Moderate":
        recommendations.append("Monitor weekly academic progress.")

    if simulated_risk == "Low":
        recommendations.append("Maintain the current performance and study habits.")

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    st.divider()
    st.subheader("📋 Scenario Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Attendance",
            "Marks",
            "Assignments",
            "Risk Score",
            "Risk Category"
        ],
        "Original": [
            f"{original_attendance}%",
            f"{original_marks}%",
            f"{original_assignments}%",
            f"{original_score}/100",
            original_risk
        ],
        "Simulated": [
            f"{simulated_attendance}%",
            f"{simulated_marks}%",
            f"{simulated_assignments}%",
            f"{simulated_score}/100",
            simulated_risk
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(
        "Simulation is hypothetical. Actual academic outcomes may differ. "
        "Original student records are not modified."
    )
