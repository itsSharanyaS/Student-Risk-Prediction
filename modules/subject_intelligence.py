import streamlit as st
import pandas as pd
import plotly.express as px


def render_subject_intelligence(students):
    st.title("📊 Advanced Academic Analytics")
    st.caption("Explore academic performance, attendance, risk, and semester insights.")

    if students is None or students.empty:
        st.warning("No student data available.")
        return

    df = students.copy()
    required = ["Student ID", "Student", "Department", "Semester",
               "Attendance", "Marks", "Assignments", "Risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    for col in ["Attendance", "Marks", "Assignments", "Semester"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Attendance", "Marks", "Assignments", "Semester"])
    if df.empty:
        st.warning("No valid student records available.")
        return

    st.sidebar.subheader("Advanced Analytics Filters")
    departments = ["All"] + sorted(df["Department"].dropna().astype(str).unique())
    department = st.sidebar.selectbox("Department", departments, key="analytics_department")

    semesters = sorted(df["Semester"].dropna().unique().tolist())
    semester = st.sidebar.selectbox("Semester", ["All"] + semesters, key="analytics_semester")

    risk_levels = ["All"] + sorted(df["Risk"].dropna().astype(str).unique())
    risk = st.sidebar.selectbox("Risk Level", risk_levels, key="analytics_risk")

    filtered = df.copy()
    if department != "All":
        filtered = filtered[filtered["Department"].astype(str) == department]
    if semester != "All":
        filtered = filtered[filtered["Semester"] == semester]
    if risk != "All":
        filtered = filtered[filtered["Risk"].astype(str) == risk]

    if filtered.empty:
        st.info("No students match these filters.")
        return

    total = len(filtered)
    avg_att = filtered["Attendance"].mean()
    avg_marks = filtered["Marks"].mean()
    avg_assignments = filtered["Assignments"].mean()
    at_risk = filtered["Risk"].astype(str).str.lower().isin(["high", "critical"]).sum()
    risk_pct = (100 * at_risk / total) if total else 0

    st.subheader("Academic Overview")
    a, b, c, d, e = st.columns(5)
    a.metric("Students", total)
    b.metric("Avg Attendance", f"{avg_att:.1f}%")
    c.metric("Avg Marks", f"{avg_marks:.1f}%")
    d.metric("Avg Assignments", f"{avg_assignments:.1f}%")
    e.metric("At Risk", at_risk, f"{risk_pct:.1f}%")

    st.divider()
    st.subheader("Performance Distribution")
    metric = st.selectbox("Indicator", ["Attendance", "Marks", "Assignments"], key="analytics_metric")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x=metric, nbins=15, title=f"{metric} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(filtered, x="Risk", y=metric, color="Risk", title=f"{metric} by Risk Level")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Attendance vs Marks")
    fig = px.scatter(
        filtered, x="Attendance", y="Marks", color="Risk",
        hover_name="Student", hover_data=["Student ID", "Department", "Semester"],
        title="Attendance and Academic Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Risk Analysis")
    left, right = st.columns(2)
    with left:
        counts = filtered["Risk"].astype(str).value_counts().rename_axis("Risk Level").reset_index(name="Students")
        fig = px.pie(counts, names="Risk Level", values="Students", hole=0.4, title="Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        risk_summary = filtered.groupby("Risk")[["Attendance", "Marks", "Assignments"]].mean().reset_index()
        melted = risk_summary.melt(id_vars="Risk", var_name="Indicator", value_name="Average")
        fig = px.bar(melted, x="Risk", y="Average", color="Indicator", barmode="group", title="Average Indicators by Risk")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Semester Analysis")
    semester_summary = filtered.groupby("Semester").agg(
        Student_Count=("Student ID", "count"),
        Average_Attendance=("Attendance", "mean"),
        Average_Marks=("Marks", "mean"),
        Average_Assignments=("Assignments", "mean"),
        At_Risk_Count=("Risk", lambda x: x.astype(str).str.lower().isin(["high", "critical"]).sum())
    ).reset_index().sort_values("Semester")

    left, right = st.columns(2)
    with left:
        fig = px.line(
            semester_summary, x="Semester",
            y=["Average_Attendance", "Average_Marks", "Average_Assignments"],
            markers=True, title="Semester Academic Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(semester_summary, x="Semester", y="At_Risk_Count", text="At_Risk_Count",
                     title="At-Risk Students by Semester")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(semester_summary.round(2), use_container_width=True, hide_index=True)

    if filtered["Department"].nunique() > 1:
        st.divider()
        st.subheader("Department Comparison")
        dept_summary = filtered.groupby("Department").agg(
            Students=("Student ID", "count"),
            Attendance=("Attendance", "mean"),
            Marks=("Marks", "mean"),
            Assignments=("Assignments", "mean")
        ).reset_index()
        fig = px.bar(dept_summary, x="Department", y=["Attendance", "Marks", "Assignments"],
                     barmode="group", title="Department-wise Performance")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dept_summary.round(2), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Academic Indicator Correlation")
    corr = filtered[["Attendance", "Marks", "Assignments"]].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Student Ranking by Marks")
    ranking = filtered.sort_values("Marks", ascending=False).copy()
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    st.dataframe(
        ranking[["Rank", "Student ID", "Student", "Department", "Semester",
                 "Attendance", "Marks", "Assignments", "Risk"]],
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("Export Analytics")
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Analytics CSV", data=csv_data,
        file_name="advanced_academic_analytics.csv", mime="text/csv"
    )
    st.success("Advanced Analytics loaded.")
