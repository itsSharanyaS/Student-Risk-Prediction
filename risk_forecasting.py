
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


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

    # Simulate gradual academic decline
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


def render_risk_forecasting(students, apply_chart_theme, risk_colors):

    st.title("🔮 Academic Risk Forecasting")
    st.caption("Simulate future academic performance and identify students who may need early intervention.")

    st.info(
        "This module simulates a possible academic decline using current attendance, "
        "marks and assignment scores. It is a scenario-based forecast, not a real "
        "future prediction."
    )

    student_id = st.selectbox(
        "Select Student",
        student_id_col = "student_id" if "student_id" in students.columns else "USN"
        students[student_id_col].tolist()
        key="forecast_student"
    )

    student = students[students["USN"] == student_id].iloc[0]

    attendance = float(student["Attendance"])
    marks = float(student["Marks"])
    assignments = float(student["Assignments"])

    result = forecast_risk(attendance, marks, assignments)

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Risk", result["current_risk"])
    col2.metric("Forecast Risk", result["predicted_risk"])
    col3.metric(
        "Forecast Attendance",
        f'{result["future_attendance"]:.1f}%'
    )

    # Comparison chart
    comparison = pd.DataFrame({
        "Metric": ["Attendance", "Marks", "Assignments"],
        "Current": [attendance, marks, assignments],
        "Forecast": [
            result["future_attendance"],
            result["future_marks"],
            result["future_assignments"]
        ]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Current",
        x=comparison["Metric"],
        y=comparison["Current"]
    ))

    fig.add_trace(go.Bar(
        name="Forecast Scenario",
        x=comparison["Metric"],
        y=comparison["Forecast"]
    ))

    fig.update_layout(
        title="Current vs Forecast Academic Performance",
        barmode="group",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        template="plotly_dark"
    )

    apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast interpretation
    st.subheader("📊 Forecast Analysis")

    if result["predicted_risk"] == "Critical":
        st.error("🚨 The student may enter a critical academic risk zone.")
    elif result["predicted_risk"] == "High":
        st.warning("⚠️ The student may experience high academic risk.")
    elif result["predicted_risk"] == "Moderate":
        st.info("📌 The student may experience moderate academic risk.")
    else:
        st.success("✅ The student is projected to remain in a low-risk scenario.")

    st.write("### 📉 Simulated Future Academic Indicators")

    st.write(f"**Attendance:** {attendance:.1f}% → {result['future_attendance']:.1f}%")
    st.write(f"**Marks:** {marks:.1f}% → {result['future_marks']:.1f}%")
    st.write(f"**Assignments:** {assignments:.1f}% → {result['future_assignments']:.1f}%")

    st.write("### 💡 Recommended Preventive Actions")

    recommendations = []

    if result["future_attendance"] < 75:
        recommendations.append("Improve attendance through regular class participation.")

    if result["future_marks"] < 50:
        recommendations.append("Schedule additional academic revision sessions.")

    if result["future_assignments"] < 50:
        recommendations.append("Complete pending assignments and seek faculty support.")

    if not recommendations:
        recommendations.append("Maintain current academic consistency and monitor progress.")

    for recommendation in recommendations:
        st.write(f"- {recommendation}")

    st.caption(
        "Forecast values are simulated by reducing current indicators. "
        "They should not be interpreted as actual future student performance."
    )

