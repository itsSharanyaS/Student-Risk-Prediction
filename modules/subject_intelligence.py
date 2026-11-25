
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def render_subject_intelligence(students):

    st.title("📊 Advanced Academic Analytics")
    st.caption(
        "Explore academic performance, attendance patterns, "
        "risk distribution, and semester-level insights."
    )

    if students is None or students.empty:
        st.warning("No student data is available for analytics.")
        return

    df = students.copy()

    # --------------------------------------------------
    # Validate and prepare data
    # --------------------------------------------------
    required_columns = [
        "Student", "Student ID", "Department", "Semester",
        "Attendance", "Marks", "Assignments", "Risk"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "Required data columns are missing: "
            + ", ".join(missing_columns)
        )
        return

    for col in ["Attendance", "Marks", "Assignments", "Semester"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Attendance", "Marks", "Assignments", "Semester"]
    )

    if df.empty:
        st.warning("No valid academic records are available.")
        return

    # Calculate risk score if it does not exist
    if "Risk Score" not in df.columns:
        df["Risk Score"] = (
            (100 - df["Attendance"]) * 0.4
            + (100 - df["Marks"]) * 0.4
            + (100 - df["Assignments"]) * 0.2
        ).clip(0, 100)

    # --------------------------------------------------
    # Sidebar filters
    # --------------------------------------------------
    st.sidebar.subheader("🔎 Analytics Filters")

    departments = ["All"] + sorted(
        df["Department"].dropna().astype(str).unique().tolist()
    )

    selected_department = st.sidebar.selectbox(
        "Department",
        departments,
        key="subject_analytics_department"
    )

    semesters = sorted(df["Semester"].dropna().unique().tolist())

    selected_semester = st.sidebar.selectbox(
        "Semester",
        ["All"] + semesters,
        key="subject_analytics_semester"
    )

    risks = ["All"] + sorted(
        df["Risk"].dropna().astype(str).unique().tolist()
    )

    selected_risk = st.sidebar.selectbox(
        "Risk Level",
        risks,
        key="subject_analytics_risk"
    )

    filtered = df.copy()

    if selected_department != "All":
        filtered = filtered[
            filtered["Department"].astype(str) == selected_department
        ]

    if selected_semester != "All":
        filtered = filtered[
            filtered["Semester"] == selected_semester
        ]

    if selected_risk != "All":
        filtered = filtered[
            filtered["Risk"] == selected_risk
        ]

    if filtered.empty:
        st.warning("No students match the selected filters.")
        return

    # --------------------------------------------------
    # KPI metrics
    # --------------------------------------------------
    total = len(filtered)
    avg_attendance = filtered["Attendance"].mean()
    avg_marks = filtered["Marks"].mean()
    avg_assignments = filtered["Assignments'].mean()

    # Correct column name
    avg_assignments = filtered["Assignments"].mean()

    at_risk = filtered["Risk"].isin(["High", "Critical"]).sum()
    at_risk_pct = (at_risk / total * 100) if total else 0

    st.divider()
    st.subheader("📌 Academic Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Students", total)
    col2.metric("Avg Attendance", f"{avg_attendance:.1f}%")
    col3.metric("Avg Marks", f"{avg_marks:.1f}%")
    col4.metric("Avg Assignments", f"{avg_assignments:.1f}%")
    col5.metric("At-Risk Students", at_risk, f"{at_risk_pct:.1f}%")

    # --------------------------------------------------
    # Performance distribution
    # --------------------------------------------------
    st.divider()
    st.subheader("📈 Academic Performance Distribution")

    metric_choice = st.selectbox(
        "Choose academic indicator",
        ["Attendance", "Marks", "Assignments"],
        key="subject_analytics_metric"
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x=metric_choice,
            nbins=15,
            title=f"{metric_choice} Distribution",
            marginal="box"
        )
        fig.update_layout(yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered,
            y=metric_choice,
            color="Risk",
            title=f"{metric_choice} by Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Attendance vs marks
    # --------------------------------------------------
    st.divider()
    st.subheader("🎯 Attendance vs Academic Performance")

    fig = px.scatter(
        filtered,
        x="Attendance",
        y="Marks",
        color="Risk",
        size="Assignments",
        hover_name="Student",
        hover_data=["Student ID", "Department", "Semester"],
        title="Attendance and Marks Relationship"
    )

    fig.update_layout(
        xaxis_title="Attendance (%)",
        yaxis_title="Marks (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Risk analysis
    # --------------------------------------------------
    st.divider()
    st.subheader("🚨 Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = (
            filtered["Risk"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = ["Risk Level", "Students"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Students",
            hole=0.45,
            title="Student Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_performance = (
            filtered.groupby("Risk")[
                ["Attendance", "Marks", "Assignments"]
            ]
            .mean()
            .reset_index()
        )

        risk_performance = risk_performance.melt(
            id_vars="Risk",
            var_name="Indicator",
            value_name="Average"
        )

        fig = px.bar(
            risk_performance,
            x="Risk",
            y="Average",
            color="Indicator",
            barmode="group",
            title="Academic Indicators by Risk Level"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Semester analytics
    # --------------------------------------------------
    st.divider()
    st.subheader("🎓 Semester Performance Analysis")

    semester_summary = (
        filtered.groupby("Semester")
        .agg(
            Student_Count=("Student ID", "count"),
            Average_Attendance=("Attendance", "mean"),
            Average_Marks=("Marks", "mean"),
            Average_Assignments=("Assignments", "mean"),
            At_Risk_Count=(
                "Risk",
                lambda x: x.isin(["High", "Critical"]).sum()
            )
        )
        .reset_index()
        .sort_values("Semester")
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            semester_summary,
            x="Semester",
            y=["Average_Attendance", "Average_Marks", "Average_Assignments"],
            markers=True,
            title="Semester-wise Academic Performance"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            semester_summary,
            x="Semester",
            y="At_Risk_Count",
            text="At_Risk_Count",
            title="At-Risk Students by Semester"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        semester_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Department analysis
    # --------------------------------------------------
    if filtered["Department"].nunique() > 1:

        st.divider()
        st.subheader("🏫 Department Comparison")

        department_summary = (
            filtered.groupby("Department")
            .agg(
                Students=("Student ID", "count"),
                Attendance=("Attendance", "mean"),
                Marks=("Marks", "mean"),
                Assignments=("Assignments", "mean")
            )
            .reset_index()
        )

        fig = px.bar(
            department_summary,
            x="Department",
            y=["Attendance", "Marks", "Assignments"],
            barmode="group",
            title="Department-wise Performance Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            department_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------
    st.divider()
    st.subheader("🔗 Academic Indicator Correlation")

    correlation_data = filtered[
        ["Attendance", "Marks", "Assignments"]
    ].corr()

    fig = px.imshow(
        correlation_data,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Correlation Between Academic Indicators"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Student performance ranking
    # --------------------------------------------------
    st.divider()
    st.subheader("🏆 Student Performance Ranking")

    ranking = filtered.sort_values(
        "Marks",
        ascending=False
    ).copy()

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1)
    )

    st.dataframe(
        ranking[
            [
                "Rank", "Student ID", "Student",
                "Department", "Semester",
                "Attendance", "Marks", "Assignments", "Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Download filtered analytics
    # --------------------------------------------------
    st.divider()
    st.subheader("📥 Export Analytics Data")

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Student Analytics",
        data=csv_data,
        file_name="advanced_academic_analytics.csv",
        mime="text/csv"
    )

    st.success("Advanced Analytics loaded successfully.")

