
import streamlit as st
import pandas as pd

# =========================================================
# SMART EARLY WARNING ENGINE
# =========================================================

def analyze_student_warnings(df):

    alerts = []

    if df is None or df.empty:
        return pd.DataFrame()

    for _, student in df.iterrows():

        warnings = []
        actions = []

        attendance = student.get("Attendance", 100)
        marks = student.get("Average Marks", 100)
        risk = student.get("Risk Score", 0)

        try:
            attendance = float(attendance)
        except (ValueError, TypeError):
            attendance = 100

        try:
            marks = float(marks)
        except (ValueError, TypeError):
            marks = 100

        try:
            risk = float(risk)
        except (ValueError, TypeError):
            risk = 0

        # Attendance warning
        if attendance < 65:
            warnings.append("Severe Attendance Shortage")
            actions.append("Schedule an attendance counselling session")

        elif attendance < 75:
            warnings.append("Low Attendance")
            actions.append("Monitor attendance and encourage regular classes")

        # Academic warning
        if marks < 40:
            warnings.append("Critical Academic Performance")
            actions.append("Provide subject-level academic support")

        elif marks < 50:
            warnings.append("Low Academic Performance")
            actions.append("Recommend remedial classes")

        # Risk score warning
        if risk >= 75:
            warnings.append("Critical Risk Score")
            actions.append("Immediate faculty intervention required")

        elif risk >= 50:
            warnings.append("Elevated Risk Score")
            actions.append("Assign mentor monitoring")

        # Combined warning
        if len(warnings) >= 3:
            priority = "Critical"

        elif len(warnings) == 2:
            priority = "High"

        elif len(warnings) == 1:
            priority = "Moderate"

        else:
            continue

        alerts.append({
            "Student ID": student.get("Student ID", "Unknown"),
            "Student Name": student.get("Name", "Unknown"),
            "Attendance": attendance,
            "Average Marks": marks,
            "Risk Score": risk,
            "Warnings": " | ".join(warnings),
            "Priority": priority,
            "Recommended Action": " | ".join(actions)
        })

    return pd.DataFrame(alerts)


# =========================================================
# DISPLAY EARLY WARNING ENGINE
# =========================================================

def render_early_warning(df):

    st.header("🚨 Smart Early Warning Engine")
    st.caption(
        "Automatically identifies students showing signs of academic deterioration."
    )

    alerts_df = analyze_student_warnings(df)

    if alerts_df.empty:
        st.success("No warning signals detected among the current students.")
        return

    # Priority metrics
    critical = len(alerts_df[alerts_df["Priority"] == "Critical"])
    high = len(alerts_df[alerts_df["Priority"] == "High"])
    moderate = len(alerts_df[alerts_df["Priority"] == "Moderate"])

    c1, c2, c3 = st.columns(3)

    c1.metric("🔴 Critical Alerts", critical)
    c2.metric("🟠 High Priority", high)
    c3.metric("🟡 Moderate Alerts", moderate)

    st.divider()

    # Filter
    priority_filter = st.selectbox(
        "Filter by Priority",
        ["All", "Critical", "High", "Moderate"]
    )

    filtered_df = alerts_df.copy()

    if priority_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Priority"] == priority_filter
        ]

    # Alert table
    st.subheader("Student Warning Records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Individual alerts
    st.subheader("Detailed Warning Analysis")

    for _, row in filtered_df.iterrows():

        if row["Priority"] == "Critical":
            emoji = "🔴"
        elif row["Priority"] == "High":
            emoji = "🟠"
        else:
            emoji = "🟡"

        with st.expander(
            f"{emoji} {row['Student Name']} — {row['Priority']} Priority"
        ):

            st.write(f"**Student ID:** {row['Student ID']}")
            st.write(f"**Attendance:** {row['Attendance']}%")
            st.write(f"**Average Marks:** {row['Average Marks']}")
            st.write(f"**Risk Score:** {row['Risk Score']}")

            st.markdown("### Detected Warning Signals")

            for warning in row["Warnings"].split(" | "):
                st.warning(warning)

            st.markdown("### Recommended Actions")

            for action in row["Recommended Action"].split(" | "):
                st.info(action)
