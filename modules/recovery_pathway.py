
import streamlit as st
import pandas as pd

def render_recovery_pathway(students):

    st.title("🎯 AI-Powered Student Recovery Pathway")
    st.caption("Personalized academic recovery plans and student improvement tracking")

    if students.empty:
        st.warning("No student data available.")
        return

    df = students.copy()

    for col in ["Attendance", "Marks", "Assignments"]:
        if col not in df.columns:
            st.error(f"Required column missing: {col}")
            return

    df["Attendance"] = pd.to_numeric(df["Attendance"], errors="coerce")
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
    df["Assignments"] = pd.to_numeric(df["Assignments"], errors="coerce")
    df = df.dropna(subset=["Attendance", "Marks", "Assignments"])

    # Identify students needing academic support
    df["Recovery Priority"] = "Low"

    df.loc[
        (df["Attendance"] < 75) | (df["Marks"] < 50) | (df["Assignments"] < 50),
        "Recovery Priority"
    ] = "High"

    df.loc[
        ((df["Attendance"] >= 75) & (df["Marks"] >= 50) & (df["Assignments"] >= 50)) &
        ((df["Attendance"] < 85) | (df["Marks"] < 65) | (df["Assignments"] < 65)),
        "Recovery Priority"
    ] = "Medium"

    # Generate personalized plans
    def generate_plan(row):
        actions = []
        weekly_targets = []

        if row["Attendance"] < 75:
            actions.append("Attend all upcoming classes and meet the mentor.")
            weekly_targets.append("Maintain at least 90% attendance this week.")

        if row["Marks"] < 50:
            actions.append("Attend remedial classes and practice fundamental concepts.")
            weekly_targets.append("Complete 2 hours of focused study daily.")

        elif row["Marks"] < 65:
            actions.append("Revise weak topics and solve previous question papers.")
            weekly_targets.append("Complete 3 practice sessions this week.")

        if row["Assignments"] < 50:
            actions.append("Complete pending assignments and seek faculty guidance.")
            weekly_targets.append("Submit all pending assignments.")

        if not actions:
            actions.append("Maintain consistent performance and explore advanced learning.")
            weekly_targets.append("Maintain current academic performance.")

        if row["Recovery Priority"] == "High":
            mentor = "Immediate faculty mentor follow-up recommended."
        elif row["Recovery Priority"] == "Medium":
            mentor = "Schedule a progress review with the mentor."
        else:
            mentor = "Routine academic monitoring."

        return pd.Series({
            "Recommended Actions": " | ".join(actions),
            "Weekly Targets": " | ".join(weekly_targets),
            "Mentor Recommendation": mentor
        })

    plan_data = df.apply(generate_plan, axis=1)
    df = pd.concat([df, plan_data], axis=1)

    # Dashboard metrics
    high = int((df["Recovery Priority"] == "High").sum())
    medium = int((df["Recovery Priority"] == "Medium").sum())
    low = int((df["Recovery Priority"] == "Low").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students Requiring Support", high + medium)
    col2.metric("High Priority", high)
    col3.metric("Medium Priority", medium)
    col4.metric("Low Priority", low)

    st.divider()

    # Student selector
    student_options = df["Student"].tolist() if "Student" in df.columns else df["Student ID"].tolist()

    selected_student = st.selectbox(
        "Select Student for Recovery Plan",
        student_options
    )

    if "Student" in df.columns:
        student = df[df["Student"] == selected_student].iloc[0]
    else:
        student = df[df["Student ID"] == selected_student].iloc[0]

    st.subheader("📋 Personalized Recovery Plan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Attendance", f"{student['Attendance']:.1f}%")
    c2.metric("Marks", f"{student['Marks']:.1f}%")
    c3.metric("Assignments", f"{student['Assignments']:.1f}%")

    st.markdown(f"### 🚦 Recovery Priority: {student['Recovery Priority']}")

    st.markdown("#### Recommended Actions")
    for action in student["Recommended Actions"].split(" | "):
        st.write("✔️", action)

    st.markdown("#### Weekly Targets")
    for target in student["Weekly Targets"].split(" | "):
        st.write("📅", target)

    st.info(student["Mentor Recommendation"])

    st.divider()

    # Recovery progress tracking
    st.subheader("📈 Recovery Progress Tracker")

    if "recovery_status" not in st.session_state:
        st.session_state.recovery_status = {}

    student_id = str(student["Student ID"]) if "Student ID" in df.columns else str(selected_student)

    current_status = st.session_state.recovery_status.get(student_id, "Not Started")

    status = st.selectbox(
        "Recovery Status",
        ["Not Started", "In Progress", "Completed"],
        index=["Not Started", "In Progress", "Completed"].index(current_status),
        key=f"status_{student_id}"
    )

    st.session_state.recovery_status[student_id] = status

    if status == "Completed":
        st.success("🎉 Recovery plan completed!")
    elif status == "In Progress":
        st.warning("Keep working toward the weekly targets.")
    else:
        st.info("Recovery plan has not started yet.")

    st.divider()

    # Download recovery report
    st.subheader("📥 Download Recovery Plan")

    report = pd.DataFrame([{
        "Student": selected_student,
        "Attendance": student["Attendance"],
        "Marks": student["Marks"],
        "Assignments": student["Assignments"],
        "Recovery Priority": student["Recovery Priority"],
        "Recommended Actions": student["Recommended Actions"],
        "Weekly Targets": student["Weekly Targets"],
        "Mentor Recommendation": student["Mentor Recommendation"],
        "Recovery Status": status
    }])

    st.download_button(
        "Download Student Recovery Report",
        report.to_csv(index=False).encode("utf-8"),
        file_name="student_recovery_plan.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("📊 All Student Recovery Plans")

    display_columns = [
        col for col in [
            "Student ID", "Student", "Attendance", "Marks",
            "Assignments", "Recovery Priority",
            "Recommended Actions", "Weekly Targets"
        ] if col in df.columns
    ]

    st.dataframe(df[display_columns], use_container_width=True)

    st.download_button(
        "Download All Recovery Plans",
        df.to_csv(index=False).encode("utf-8"),
        file_name="all_student_recovery_plans.csv",
        mime="text/csv"
    )

