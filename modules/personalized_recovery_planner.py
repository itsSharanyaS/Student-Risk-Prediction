
import streamlit as st
import pandas as pd


# =========================================================
# PERSONALIZED ACADEMIC RECOVERY PLANNER
# =========================================================

def generate_recovery_plan(student):

    attendance = float(student.get("Attendance", 0))
    marks = float(student.get("Marks", 0))
    assignments = float(student.get("Assignments", 0))

    # Identify priority areas
    priority_areas = []

    if attendance < 75:
        priority_areas.append("Attendance Improvement")

    if marks < 60:
        priority_areas.append("Academic Performance")

    if assignments < 60:
        priority_areas.append("Assignment Completion")

    if not priority_areas:
        priority_areas.append("Continuous Academic Improvement")

    # Determine weakest area
    scores = {
        "Attendance": attendance,
        "Academic Performance": marks,
        "Assignment Completion": assignments
    }

    weakest_area = min(scores, key=scores.get)

    # Weekly targets
    targets = {
        "Attendance": "Maintain at least 90% attendance",
        "Academic Performance": "Study for at least 2 hours daily",
        "Assignment Completion": "Complete all pending assignments"
    }

    # Daily schedule
    schedule = [
        ("Day 1", "Identify difficult topics and organize study materials"),
        ("Day 2", "Revise weak topics and solve basic practice questions"),
        ("Day 3", "Attend classes and clarify doubts with faculty"),
        ("Day 4", "Practice problems and review class notes"),
        ("Day 5", "Complete pending assignments"),
        ("Day 6", "Take a self-test and identify mistakes"),
        ("Day 7", "Review weekly progress and prepare for next week")
    ]

    # Mentor recommendations
    mentor_actions = []

    if attendance < 75:
        mentor_actions.append("Discuss attendance issues and establish a regular attendance routine")

    if marks < 60:
        mentor_actions.append("Arrange academic support and subject revision")

    if assignments < 60:
        mentor_actions.append("Review assignment progress and help clear pending work")

    if not mentor_actions:
        mentor_actions.append("Conduct a weekly check-in to maintain academic progress")

    return {
        "Priority Areas": priority_areas,
        "Weakest Area": weakest_area,
        "Weekly Targets": targets,
        "Study Schedule": schedule,
        "Mentor Actions": mentor_actions
    }


# =========================================================
# RECOVERY PLANNER UI
# =========================================================

def render_recovery_planner(df):

    st.title("🧠 Personalized Academic Recovery Planner")

    st.caption(
        "A personalized 7-day academic improvement plan "
        "generated using student performance data."
    )

    if df is None or df.empty:
        st.warning("No student data available.")
        return

    # Student selection
    student_options = df["Student"].tolist()

    selected_student = st.selectbox(
        "Select Student",
        student_options
    )

    student = df[df["Student"] == selected_student].iloc[0]

    plan = generate_recovery_plan(student)

    # Student details
    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Attendance", f"{student['Attendance']}%")
    col2.metric("Marks", f"{student['Marks']}%")
    col3.metric("Assignments", f"{student['Assignments']}%")

    st.divider()

    # Priority areas
    st.subheader("🎯 Priority Areas")

    for area in plan["Priority Areas"]:
        st.warning(area)

    st.info(f"**Weakest Area:** {plan['Weakest Area']}")

    # Weekly targets
    st.subheader("📌 Weekly Targets")

    for area, target in plan["Weekly Targets"].items():
        st.write(f"**{area}:** {target}")

    st.divider()

    # 7-day study schedule
    st.subheader("📅 7-Day Study Plan")

    for day, activity in plan["Study Schedule"]:
        st.markdown(f"**{day}**")
        st.write(activity)

    st.divider()

    # Mentor actions
    st.subheader("👨‍🏫 Mentor Action Plan")

    for action in plan["Mentor Actions"]:
        st.info(action)

    # Export plan
    st.divider()

    st.subheader("📥 Download Recovery Plan")

    report = f"""
PERSONALIZED ACADEMIC RECOVERY PLAN

Student: {student['Student']}
Student ID: {student['Student ID']}

Attendance: {student['Attendance']}%
Marks: {student['Marks']}%
Assignments: {student['Assignments']}%

Priority Areas:
{chr(10).join(plan['Priority Areas'])}

Weakest Area:
{plan['Weakest Area']}

Weekly Targets:
{chr(10).join(plan['Weekly Targets'].values())}

7-Day Study Schedule:
{chr(10).join([f'{day}: {activity}' for day, activity in plan['Study Schedule']])}

Mentor Action Plan:
{chr(10).join(plan['Mentor Actions'])}
"""

    st.download_button(
        "Download Student Recovery Plan",
        data=report,
        file_name=f"{selected_student}_recovery_plan.txt",
        mime="text/plain"
    )
