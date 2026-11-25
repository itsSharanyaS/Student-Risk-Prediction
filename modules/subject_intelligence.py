
import streamlit as st
import pandas as pd
import plotly.express as px


def render_subject_intelligence(students):

    st.title("📊 Advanced Academic Analytics")
    st.caption("Academic performance, attendance, risk and semester insights")

    if students is None or students.empty:
        st.warning("No student data available.")
        return

    df = students.copy()

    required = [
        "Student ID", "Student", "Department", "Semester",
        "Attendance", "Marks", "Assignments", "Risk"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    for col in ["Attendance", "Marks", "Assignments", "Semester"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Attendance", "Marks", "Assignments", "Semester"]
    )

    if df.empty:
        st.warning("No valid student records found.")
        return

    if "Risk Score" not in df.columns:
        df["Risk Score"] = (
            (100 - df["Attendance"]) * 0.4
            + (100 - df["Marks"]) * 0.4
            + (100 - df["Assignments"]) * 0.2
        ).clip(0, 100)

    # ============================
    # FILTERS
    # ============================
    st.sidebar.subheader("🔎 Analytics Filters")

    departments = ["All"] + sorted(
        df["Department"].dropna().astype(str).unique().tolist()
    )

    department = st.sidebar.selectbox(
        "Department",
        departments,
        key="advanced_department"
    )

    semesters = sorted(df["Semester"].dropna().unique().tolist())

    semester = st.sidebar.selectbox(
        "Semester",
        ["All"] + semesters,
        key="advanced_semester"
    )

    risks = ["All"] + sorted(
        df["Risk"].dropna().astype(str).unique().tolist()
    )

    risk = st.sidebar.selectbox(
        "Risk Level",
        risks,
        key="advanced_risk"
    )

    filtered = df.copy()

    if department != "All":
        filtered = filtered[
            filtered["Department"].astype(str) == department
        ]

    if semester != "All":
        filtered = filtered[filtered["Semester"] == semester]

    if risk != "All":
        filtered = filtered[filtered["Risk"] == risk]

    if filtered.empty:
        st.warning("No students match your selected filters.")
        return

    # ============================
    # OVERVIEW
    # ============================
    total = len(filtered)
    avg_attendance = filtered["Attendance"].mean()
    avg_marks = filtered["Marks"].mean()
    avg_assignments = filtered["Assignments"].mean()

    at_risk = filtered["Risk"].isin(["High", "Critical"]).sum()
    at_risk_percentage = at_risk / total * 100 if total else 0

    st.divider()
    st.subheader("📌 Academic Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Students", total)
    c2.metric("Average Attendance", f"{avg_attendance:.1f}%")
    c3.metric("Average Marks", f"{avg_marks:.1f}%")
    c4.metric("Average Assignments", f"{avg_assignments:.1f}%")
    c5.metric("Students At Risk", at_risk, f"{at_risk_percentage:.1f}%")

    # ============================
    # PERFORMANCE DISTRIBUTION
    # ============================
    st.divider()
    st.subheader("📈 Academic Performance Distribution")

    metric = st.selectbox(
        "Select Indicator",
        ["Attendance", "Marks", "Assignments"],
        key="advanced_metric"
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered,
            x=metric,
            nbins=15,
            marginal="box",
            title=f"{metric} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered,
            x="Risk",
            y=metric,
            color="Risk",
            title=f"{metric} by Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================
    # ATTENDANCE VS MARKS
    # ============================
    st.divider()
    st.subheader("🎯 Attendance vs Marks")

    fig = px.scatter(
        filtered,
        x="Attendance",
        y="Marks",
        color="Risk",
        size="Assignments",
        hover_name="Student",
        hover_data=["Student ID", "Department", "Semester"],
        title="Attendance and Academic Performance"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # RISK ANALYSIS
    # ============================
    st.divider()
    st.subheader("🚨 Risk Analysis")

    c1, c2 = st.columns(2)

    with c1:
        risk_counts = filtered["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Students"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Students",
            hole=0.45,
            title="Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_summary = (
            filtered.groupby("Risk")[
                ["Attendance", "Marks", "Assignments"]
            ]
            .mean()
            .reset_index()
        )

        risk_summary = risk_summary.melt(
            id_vars="Risk",
            var_name="Indicator",
            value_name="Average"
        )

        fig = px.bar(
            risk_summary,
            x="Risk",
            y="Average",
            color="Indicator",
            barmode="group",
            title="Performance by Risk Level"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================
    # SEMESTER ANALYSIS
    # ============================
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

    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            semester_summary,
            x="Semester",
            y=[
                "Average_Attendance",
                "Average_Marks",
                "Average_Assignments"
            ],
            markers=True,
            title="Semester Academic Performance"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
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

    # ============================
    # DEPARTMENT ANALYSIS
    # ============================
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
            title="Department-wise Performance"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            department_summary.round(2),
            use_container_width=True,
            hide_index=True
        )

    # ============================
    # CORRELATION
    # ============================
    st.divider()
    st.subheader("🔗 Academic Indicator Correlation")

    correlation = filtered[
        ["Attendance", "Marks", "Assignments"]
    ].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # STUDENT RANKING
    # ============================
    st.divider()
    st.subheader("🏆 Student Performance Ranking")

    ranking = filtered.sort_values(
        "Marks",
        ascending=False
    ).copy()

    ranking.insert(0, "Rank", range(1, len(ranking) + 1))

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

    # ============================
    # DOWNLOAD
    # ============================
    st.divider()
    st.subheader("📥 Export Analytics")

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Analytics CSV",
        data=csv_data,
        file_name="advanced_academic_analytics.csv",
        mime="text/csv"
    )

    st.success("Advanced Analytics loaded successfully.")
