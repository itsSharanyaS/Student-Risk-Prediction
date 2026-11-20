
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# =========================================================
# 1. RISK CALCULATION
# =========================================================

def calculate_risk(attendance, marks, assignments):

    score = 0

    # Attendance risk
    if attendance < 50:
        score += 40
    elif attendance < 60:
        score += 30
    elif attendance < 75:
        score += 15

    # Marks risk
    if marks < 35:
        score += 35
    elif marks < 50:
        score += 25
    elif marks < 65:
        score += 15

    # Assignment risk
    if assignments < 40:
        score += 25
    elif assignments < 60:
        score += 15
    elif assignments < 75:
        score += 10

    score = min(score, 100)

    # Risk category
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
# 2. RISK COLOR
# =========================================================

def get_risk_color(risk, risk_colors):

    default_colors = {
        "Low": "#22C55E",
        "Moderate": "#F59E0B",
        "High": "#F97316",
        "Critical": "#EF4444"
    }

    return risk_colors.get(risk, default_colors.get(risk, "#64748B"))


# =========================================================
# 3. WHAT-IF SIMULATOR PAGE
# =========================================================

def render_what_if_simulator(students, apply_chart_theme, risk_colors):

    st.title("🧪 Academic What-If Simulator")

    st.markdown(
        """
        Simulate academic improvements or declines and instantly observe
        how a student's risk score and risk category change.
        """
    )

    st.info(
        "This is a scenario-based simulation. It does not change student "
        "records and does not represent an actual future prediction."
    )

    # -----------------------------------------------------
    # STUDENT SELECTION
    # -----------------------------------------------------

    if students.empty:
        st.warning("No student records available.")
        return

    student_ids = students["student_id"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_ids,
        key="what_if_student"
    )

    student = students[
        students["student_id"].astype(str) == selected_id
    ].iloc[0]

    student_name = student["name"]

    original_attendance = int(student["attendance"])
    original_marks = int(student["marks"])
    original_assignments = int(student["assignments"])

    # -----------------------------------------------------
    # ORIGINAL RISK
    # -----------------------------------------------------

    original_score, original_risk = calculate_risk(
        original_attendance,
        original_marks,
        original_assignments
    )

    st.divider()

    st.subheader("👤 Student Profile")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Student Name", student_name)
    col2.metric("Student ID", selected_id)
    col3.metric("Department", student["department"])
    col4.metric("Semester", student["semester"])

    # -----------------------------------------------------
    # ORIGINAL PERFORMANCE
    # -----------------------------------------------------

    st.subheader("📊 Current Academic Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Attendance", f"{original_attendance}%")
    c2.metric("Marks", f"{original_marks}%")
    c3.metric("Assignments", f"{original_assignments}%")
    c4.metric("Risk Score", f"{original_score}/100")

    st.markdown(
        f"### Current Risk Level: **{original_risk}**"
    )

    # -----------------------------------------------------
    # SIMULATION SLIDERS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🎯 Create Your Academic Scenario")

    st.write(
        "Adjust the values below to simulate an academic improvement "
        "or decline for this student."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        simulated_attendance = st.slider(
            "Simulated Attendance (%)",
            min_value=0,
            max_value=100,
            value=original_attendance,
            key="what_if_attendance"
        )

    with col2:
        simulated_marks = st.slider(
            "Simulated Marks (%)",
            min_value=0,
            max_value=100,
            value=original_marks,
            key="what_if_marks"
        )

    with col3:
        simulated_assignments = st.slider(
            "Simulated Assignments (%)",
            min_value=0,
            max_value=100,
            value=original_assignments,
            key="what_if_assignments"
        )

    # -----------------------------------------------------
    # SIMULATED RISK
    # -----------------------------------------------------

    simulated_score, simulated_risk = calculate_risk(
        simulated_attendance,
        simulated_marks,
        simulated_assignments
    )

    score_difference = original_score - simulated_score

    # -----------------------------------------------------
    # SIMULATION RESULTS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🔮 Simulation Results")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Original Risk Score",
        f"{original_score}/100"
    )

    r2.metric(
        "Simulated Risk Score",
        f"{simulated_score}/100",
        delta=f"{-score_difference} points",
        delta_color="inverse"
    )

    r3.metric(
        "Risk Score Improvement",
        f"{score_difference} points"
    )

    # Risk category comparison

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Risk")
        st.markdown(f"## {original_risk}")

    with col2:
        st.markdown("### Simulated Risk")
        st.markdown(f"## {simulated_risk}")

    # -----------------------------------------------------
    # RISK CHANGE MESSAGE
    # -----------------------------------------------------

    if simulated_score < original_score:

        st.success(
            f"🎉 Excellent! Your scenario reduces the risk score "
            f"by {score_difference} points."
        )

    elif simulated_score > original_score:

        st.warning(
            f"⚠️ This scenario increases the risk score by "
            f"{abs(score_difference)} points."
        )

    else:

        st.info(
            "The risk score remains unchanged in this scenario."
        )

    # -----------------------------------------------------
    # PERFORMANCE COMPARISON CHART
    # -----------------------------------------------------

    st.subheader("📈 Original vs Simulated Performance")

    comparison_data = pd.DataFrame({
        "Metric": [
            "Attendance",
            "Marks",
            "Assignments"
        ],
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

    fig.add_trace(
        go.Bar(
            name="Original Performance",
            x=comparison_data["Metric"],
            y=comparison_data["Original"],
            text=comparison_data["Original"],
            textposition="outside"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Simulated Performance",
            x=comparison_data["Metric"],
            y=comparison_data["Simulated"],
            text=comparison_data["Simulated"],
            textposition="outside"
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Performance (%)",
        yaxis_range=[0, 110],
        height=450,
        legend_title="Scenario",
        template="plotly_white"
    )

    try:
        apply_chart_theme(fig)
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # RISK SCORE COMPARISON
    # -----------------------------------------------------

    st.subheader("🚦 Risk Score Comparison")

    risk_fig = go.Figure()

    risk_fig.add_trace(
        go.Bar(
            x=["Original", "Simulated"],
            y=[original_score, simulated_score],
            text=[original_score, simulated_score],
            textposition="outside",
            marker_color=[
                get_risk_color(original_risk, risk_colors),
                get_risk_color(simulated_risk, risk_colors)
            ]
        )
    )

    risk_fig.update_layout(
        yaxis_title="Risk Score",
        yaxis_range=[0, 110],
        height=350,
        showlegend=False,
        template="plotly_white"
    )

    try:
        apply_chart_theme(risk_fig)
    except Exception:
        pass

    st.plotly_chart(risk_fig, use_container_width=True)

    # -----------------------------------------------------
    # PERSONALIZED RECOMMENDATIONS
    # -----------------------------------------------------

    st.divider()

    st.subheader("💡 Personalized Recommendations")

    recommendations = []

    if simulated_attendance < 75:
        recommendations.append(
            "Improve attendance through consistent classroom participation."
        )

    if simulated_marks < 65:
        recommendations.append(
            "Focus on weak subjects and follow a structured study schedule."
        )

    if simulated_assignments < 75:
        recommendations.append(
            "Complete assignments on time and practice regularly."
        )

    if simulated_risk in ["High", "Critical"]:
        recommendations.append(
            "Schedule a mentor meeting and create an academic improvement plan."
        )

    if simulated_risk == "Moderate":
        recommendations.append(
            "Monitor weekly academic progress to prevent further risk."
        )

    if simulated_risk == "Low":
        recommendations.append(
            "Maintain current performance and continue consistent study habits."
        )

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    # -----------------------------------------------------
    # SCENARIO SUMMARY
    # -----------------------------------------------------

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

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Simulation results are hypothetical. Actual academic outcomes "
        "may differ. Original student records remain unchanged."
    )

