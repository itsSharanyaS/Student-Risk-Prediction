
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

        attendance = pd.to_numeric(
            student.get("Attendance", 0), errors="coerce"
        )

        marks = pd.to_numeric(
            student.get("Marks", 0), errors="coerce"
        )

        assignments = pd.to_numeric(
            student.get("Assignments", 0), errors="coerce"
        )

        risk_score = pd.to_numeric(
            student.get("Risk Score", 0), errors="coerce"
        )

        risk_level = student.get("Risk", "Low")

        attendance = 0 if pd.isna(attendance) else attendance
        marks = 0 if pd.isna(marks) else marks
        assignments = 0 if pd.isna(assignments) else assignments
        risk_score = 0 if pd.isna(risk_score) else risk_score

        # 1. Attendance Warning
        if attendance < 50:
            warnings.append("Critical Attendance")
            actions.append("Immediate attendance counselling")

        elif attendance < 65:
            warnings.append("Severe Attendance Shortage")
            actions.append("Schedule mentor meeting")

        elif attendance < 75:
            warnings.append("Low Attendance")
            actions.append("Monitor attendance weekly")

        # 2. Marks Warning
        if marks < 35:
            warnings.append("Critical Academic Performance")
            actions.append("Arrange academic remedial support")

        elif marks < 50:
            warnings.append("Low Academic Performance")
            actions.append("Recommend subject-wise tutoring")

        elif marks < 60:
            warnings.append("Below Average Academic Performance")
            actions.append("Provide additional practice")

        # 3. Assignment Warning
        if assignments < 40:
            warnings.append("Critical Assignment Performance")
            actions.append("Review pending assignments")

        elif assignments < 60:
            warnings.append("Low Assignment Performance")
            actions.append("Monitor assignment completion")

        # 4. Risk Score Warning
        if risk_score >= 75:
            warnings.append("Critical Risk Score")
            actions.append("Immediate faculty intervention")

        elif risk_score >= 50:
            warnings.append("High Risk Score")
            actions.append("Assign mentor monitoring")

        # 5. Overall Risk Warning
        if risk_level == "Critical":
            warnings.append("Critical Overall Risk")
            actions.append("Prepare urgent student recovery plan")

        elif risk_level == "High":
            warnings.append("High Overall Risk")
            actions.append("Initiate academic intervention")

        # Ignore students with no warning signals
        if not warnings:
            continue

        # Determine priority
        if risk_level == "Critical" or risk_score >= 75:
            priority = "Critical"

        elif len(warnings) >= 3 or risk_score >= 50:
            priority = "High"

        else:
            priority = "Moderate"

        alerts.append({
            "Student ID": student.get("Student ID", "Unknown"),
            "Student": student.get("Student", "Unknown"),
            "Attendance": attendance,
            "Marks": marks,
            "Assignments": assignments,
            "Risk Score": risk_score,
            "Risk Level": risk_level,
            "Warning Signals": ", ".join(warnings),
            "Priority": priority,
            "Recommended Actions": " | ".join(actions)
        })

    return pd.DataFrame(alerts)


# =========================================================
# EARLY WARNING CENTER UI
# =========================================================

def render_early_warning(df):

    st.title("🚨 Smart Early Warning Engine")

    st.caption(
        "Intelligent detection of academic warning signals "
        "and students requiring early intervention."
    )

    alerts_df = analyze_student_warnings(df)

    if alerts_df.empty:
        st.success("🎉 No warning signals detected.")
        return

    # -----------------------------------------------------
    # ALERT METRICS
    # -----------------------------------------------------

    critical = len(alerts_df[alerts_df["Priority"] == "Critical"])
    high = len(alerts_df[alerts_df["Priority"] == "High"])
    moderate = len(alerts_df[alerts_df["Priority"] == "Moderate"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🚨 Total Alerts", len(alerts_df))
    col2.metric("🔴 Critical", critical)
    col3.metric("🟠 High", high)
    col4.metric("🟡 Moderate", moderate)

    st.divider()

    # -----------------------------------------------------
    # PRIORITY FILTER
    # -----------------------------------------------------

    priority_filter = st.selectbox(
        "Filter alerts by priority",
        ["All", "Critical", "High", "Moderate"]
    )

    filtered = alerts_df.copy()

    if priority_filter != "All":
        filtered = filtered[filtered["Priority"] == priority_filter]

    # -----------------------------------------------------
    # ALERT TABLE
    # -----------------------------------------------------

    st.subheader("📋 Student Alert List")

    st.dataframe(
        filtered[[
            "Student ID", "Student", "Attendance",
            "Marks", "Assignments", "Risk Score",
            "Risk Level", "Priority"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # INDIVIDUAL ALERT DETAILS
    # -----------------------------------------------------

    st.subheader("🔎 Detailed Warning Analysis")

    for _, row in filtered.iterrows():

        if row["Priority"] == "Critical":
            icon = "🔴"
        elif row["Priority"] == "High":
            icon = "🟠"
        else:
            icon = "🟡"

        with st.expander(
            f"{icon} {row['Student']} — {row['Priority']} Priority"
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric("Attendance", f"{row['Attendance']}%")
            col2.metric("Marks", f"{row['Marks']}%")
            col3.metric("Risk Score", row["Risk Score"])

            st.write("**Risk Level:**", row["Risk Level"])

            st.markdown("### ⚠️ Warning Signals")

            for warning in row["Warning Signals"].split(", "):
                st.warning(warning)

            st.markdown("### 💡 Recommended Actions")

            for action in row["Recommended Actions"].split(" | "):
                st.info(action)

    # -----------------------------------------------------
    # DOWNLOAD ALERT REPORT
    # -----------------------------------------------------

    st.divider()

    st.subheader("📥 Export Warning Report")

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Early Warning Report",
        data=csv_data,
        file_name="early_warning_report.csv",
        mime="text/csv"
    )
