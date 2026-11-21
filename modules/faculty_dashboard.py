
import streamlit as st
import pandas as pd
import plotly.express as px


def render_faculty_dashboard(students):

    st.markdown("""
    <style>
    .faculty-header {
        font-size: 30px;
        font-weight: 700;
        color: #183B56;
        margin-bottom: 4px;
    }
    .faculty-subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 25px;
    }
    .faculty-card {
        background: #FFFFFF;
        border: 1px solid #E8EDF2;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="faculty-header">Faculty / Mentor Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="faculty-subtitle">Academic overview and student monitoring</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    df = students.copy()

    if df.empty:
        st.warning("No student records available.")
        return

    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    attendance_col = "attendance"
    marks_col = "marks"
    assignments_col = "assignments"

    for col in [attendance_col, marks_col, assignments_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Risk calculation
    def calculate_risk(row):
        attendance = row.get("attendance", 0)
        marks = row.get("marks", 0)
        assignments = row.get("assignments", 0)

        score = (
            (100 - attendance) * 0.35
            + (100 - marks) * 0.40
            + (100 - assignments) * 0.25
        )

        if score >= 65:
            return "High"
        elif score >= 40:
            return "Moderate"
        else:
            return "Low"

    df["Faculty Risk"] = df.apply(calculate_risk, axis=1)

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    total_students = len(df)
    high_risk = (df["Faculty Risk"] == "High").sum()
    moderate_risk = (df["Faculty Risk"] == "Moderate").sum()
    low_risk = (df["Faculty Risk"] == "Low").sum()

    avg_attendance = df["attendance"].mean() if "attendance" in df else 0
    avg_marks = df["marks"].mean() if "marks" in df else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)
    c2.metric("At Risk Students", high_risk + moderate_risk)
    c3.metric("Average Attendance", f"{avg_attendance:.1f}%")
    c4.metric("Average Marks", f"{avg_marks:.1f}%")

    st.markdown("---")

    # --------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Student Risk Overview")

    risk_counts = df["Faculty Risk"].value_counts().reindex(
        ["High", "Moderate", "Low"], fill_value=0
    )

    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        hole=0.55,
        color=risk_counts.index,
        color_discrete_map={
            "High": "#EF4444",
            "Moderate": "#F59E0B",
            "Low": "#10B981"
        }
    )

    fig.update_layout(
        height=350,
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
        legend_title="Risk Level"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # PERFORMANCE OVERVIEW
    # --------------------------------------------------

    st.subheader("Academic Performance Overview")

    chart1, chart2 = st.columns(2)

    with chart1:
        if "attendance" in df.columns:
            fig_att = px.histogram(
                df,
                x="attendance",
                nbins=10,
                title="Attendance Distribution",
                color_discrete_sequence=["#14B8A6"]
            )
            fig_att.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_att, use_container_width=True)

    with chart2:
        if "marks" in df.columns:
            fig_marks = px.histogram(
                df,
                x="marks",
                nbins=10,
                title="Marks Distribution",
                color_discrete_sequence=["#3B82F6"]
            )
            fig_marks.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_marks, use_container_width=True)

    # --------------------------------------------------
    # STUDENT MONITORING
    # --------------------------------------------------

    st.subheader("Student Monitoring")

    search = st.text_input("Search Student ID")

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "High", "Moderate", "Low"]
    )

    filtered = df.copy()

    if search:
        if "student_id" in filtered.columns:
            filtered = filtered[
                filtered["student_id"].astype(str).str.contains(search, case=False)
            ]

    if risk_filter != "All":
        filtered = filtered[filtered["Faculty Risk"] == risk_filter]

    display_columns = [
        col for col in
        ["student_id", "department", "semester", "attendance",
         "marks", "assignments", "Faculty Risk"]
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # PRIORITY STUDENTS
    # --------------------------------------------------

    st.subheader("Priority Students")

    priority = df[df["Faculty Risk"].isin(["High", "Moderate"])].copy()

    priority = priority.sort_values(
        by=["attendance", "marks"],
        ascending=True
    )

    if not priority.empty:
        st.dataframe(
            priority[display_columns],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No students currently require priority monitoring.")

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.subheader("Faculty Report")

    report = df.copy()

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Faculty Student Report",
        data=csv,
        file_name="faculty_student_report.csv",
        mime="text/csv"
    )

