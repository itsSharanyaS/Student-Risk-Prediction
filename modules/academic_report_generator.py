
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# =========================================================
# ACADEMIC REPORT GENERATOR
# =========================================================

RISK_ORDER = ["Low", "Moderate", "High", "Critical"]

RISK_COLORS = {
    "Low": "#16A34A",
    "Moderate": "#D97706",
    "High": "#EA580C",
    "Critical": "#DC2626"
}


def create_html_report(students, title, report_type):
    """Create a downloadable HTML academic report."""

    total = len(students)

    avg_attendance = students["Attendance"].mean() if total else 0
    avg_marks = students["Marks"].mean() if total else 0
    avg_assignments = students["Assignments"].mean() if total else 0

    risk_counts = (
        students["Risk"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
    )

    table_html = students[
        ["Student ID", "Student", "Semester", "Attendance",
         "Marks", "Assignments", "Risk"]
    ].to_html(index=False, border=0, classes="data-table")

    risk_rows = "".join(
        f"<tr><td>{risk}</td><td>{count}</td></tr>"
        for risk, count in risk_counts.items()
    )

    generated_time = datetime.now().strftime("%d %B %Y, %I:%M %p")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #F4F7FB;
                color: #1E293B;
                margin: 0;
                padding: 30px;
            }}

            .header {{
                background: linear-gradient(135deg, #0F766E, #14B8A6);
                color: white;
                padding: 30px;
                border-radius: 16px;
            }}

            .header h1 {{
                margin-top: 0;
            }}

            .metrics {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 25px 0;
            }}

            .card {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
            }}

            .card h2 {{
                font-size: 26px;
                color: #0F766E;
                margin-bottom: 5px;
            }}

            .section {{
                background: white;
                margin-top: 20px;
                padding: 25px;
                border-radius: 14px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}

            th {{
                background: #0F766E;
                color: white;
                padding: 12px;
                text-align: left;
            }}

            td {{
                border-bottom: 1px solid #E2E8F0;
                padding: 10px;
            }}

            tr:nth-child(even) {{
                background: #F8FAFC;
            }}

            .footer {{
                text-align: center;
                margin-top: 30px;
                color: #64748B;
                font-size: 13px;
            }}

            @media print {{
                body {{
                    background: white;
                    padding: 10px;
                }}

                .section, .card {{
                    break-inside: avoid;
                }}
            }}
        </style>
    </head>

    <body>

        <div class="header">
            <h1>🎓 Student Academic Intelligence Report</h1>
            <h3>{title}</h3>
            <p>Report Type: {report_type}</p>
            <p>Generated: {generated_time}</p>
        </div>

        <div class="metrics">

            <div class="card">
                <p>Total Students</p>
                <h2>{total}</h2>
            </div>

            <div class="card">
                <p>Average Attendance</p>
                <h2>{avg_attendance:.1f}%</h2>
            </div>

            <div class="card">
                <p>Average Marks</p>
                <h2>{avg_marks:.1f}%</h2>
            </div>

            <div class="card">
                <p>Average Assignments</p>
                <h2>{avg_assignments:.1f}%</h2>
            </div>

        </div>

        <div class="section">
            <h2>Risk Distribution</h2>

            <table>
                <thead>
                    <tr>
                        <th>Risk Level</th>
                        <th>Students</th>
                    </tr>
                </thead>

                <tbody>
                    {risk_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Student Academic Records</h2>
            {table_html}
        </div>

        <div class="footer">
            Student Risk Prediction System | Academic Intelligence Platform
        </div>

    </body>
    </html>
    """

    return html


def render_academic_report_generator(students):

    # =========================================================
    # HEADER
    # =========================================================

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0F766E,#14B8A6);
        padding:25px;
        border-radius:18px;
        color:white;">
        <h2 style="margin:0;">📄 Academic Report Generator</h2>
        <p style="margin-bottom:0;">
        Generate student academic reports, semester summaries,
        risk reports and downloadable performance records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if students.empty:
        st.warning("No student data available to generate reports.")
        return

    # =========================================================
    # REPORT FILTERS
    # =========================================================

    st.subheader("⚙️ Report Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        semester_options = sorted(
            students["Semester"].dropna().unique().tolist()
        )

        selected_semester = st.selectbox(
            "Semester",
            ["All Semesters"] + semester_options
        )

    with col2:
        selected_risk = st.selectbox(
            "Risk Level",
            ["All Risks"] + RISK_ORDER
        )

    with col3:
        report_type = st.selectbox(
            "Report Type",
            [
                "Overall Academic Report",
                "Semester Performance Report",
                "At-Risk Student Report",
                "Individual Student Report"
            ]
        )

    filtered = students.copy()

    if selected_semester != "All Semesters":
        filtered = filtered[
            filtered["Semester"] == selected_semester
        ]

    if selected_risk != "All Risks":
        filtered = filtered[
            filtered["Risk"] == selected_risk
        ]

    # =========================================================
    # INDIVIDUAL REPORT SELECTION
    # =========================================================

    if report_type == "Individual Student Report":

        available_students = filtered["Student"].tolist()

        if not available_students:
            st.warning("No students match the selected filters.")
            return

        selected_student = st.selectbox(
            "Select Student",
            available_students
        )

        filtered = filtered[
            filtered["Student"] == selected_student
        ].copy()

    elif report_type == "At-Risk Student Report":

        filtered = filtered[
            filtered["Risk"].isin(["High", "Critical"])
        ].copy()

    if filtered.empty:
        st.warning("No student records match the selected report filters.")
        return

    # =========================================================
    # KPI SUMMARY
    # =========================================================

    st.divider()
    st.subheader("📊 Report Overview")

    total = len(filtered)

    avg_attendance = filtered["Attendance"].mean()
    avg_marks = filtered["Marks"].mean()
    avg_assignments = filtered["Assignments"].mean()

    at_risk = filtered["Risk"].isin(["High", "Critical"]).sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Students", total)
    col2.metric("At Risk", int(at_risk))
    col3.metric("Avg Attendance", f"{avg_attendance:.1f}%")
    col4.metric("Avg Marks", f"{avg_marks:.1f}%")
    col5.metric("Avg Assignments", f"{avg_assignments:.1f}%")

    # =========================================================
    # PERFORMANCE VISUALIZATION
    # =========================================================

    st.divider()
    st.subheader("📈 Academic Performance Analysis")

    performance_data = pd.DataFrame({
        "Metric": ["Attendance", "Marks", "Assignments"],
        "Average": [
            avg_attendance,
            avg_marks,
            avg_assignments
        ]
    })

    fig = px.bar(
        performance_data,
        x="Metric",
        y="Average",
        color="Metric",
        text_auto=".1f",
        title="Average Academic Performance"
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 100],
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # RISK DISTRIBUTION
    # =========================================================

    st.subheader("🚨 Risk Distribution")

    risk_counts = (
        filtered["Risk"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )

    risk_counts.columns = ["Risk Level", "Students"]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Students",
            hole=0.55,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            title="Student Risk Distribution"
        )

        fig.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Students",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            text="Students",
            title="Students by Risk Level"
        )

        fig.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # SEMESTER ANALYSIS
    # =========================================================

    st.divider()
    st.subheader("🎓 Semester Performance Report")

    semester_report = filtered.groupby("Semester").agg(
        Total_Students=("Student ID", "count"),
        Average_Attendance=("Attendance", "mean"),
        Average_Marks=("Marks", "mean"),
        Average_Assignments=("Assignments", "mean"),
        At_Risk_Students=(
            "Risk",
            lambda x: x.isin(["High", "Critical"]).sum()
        )
    ).reset_index()

    st.dataframe(
        semester_report.round(2),
        use_container_width=True,
        hide_index=True
    )

    # =========================================================
    # STUDENT RECORDS
    # =========================================================

    st.divider()
    st.subheader("👨‍🎓 Student Academic Records")

    report_columns = [
        "Student ID",
        "Student",
        "Department",
        "Semester",
        "Attendance",
        "Marks",
        "Assignments",
        "Risk",
        "Risk Score"
    ]

    st.dataframe(
        filtered[report_columns].sort_values(
            "Risk Score",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # =========================================================
    # REPORT DOWNLOADS
    # =========================================================

    st.divider()
    st.subheader("📥 Download Reports")

    col1, col2 = st.columns(2)

    with col1:

        csv_data = filtered[report_columns].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Academic Report (CSV)",
            data=csv_data,
            file_name="academic_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        html_report = create_html_report(
            filtered,
            "Student Academic Report",
            report_type
        )

        st.download_button(
            "🌐 Download Professional Report (HTML)",
            data=html_report,
            file_name="student_academic_report.html",
            mime="text/html",
            use_container_width=True
        )

    # =========================================================
    # AT-RISK REPORT
    # =========================================================

    st.divider()
    st.subheader("🚩 Students Requiring Attention")

    priority_students = filtered[
        filtered["Risk"].isin(["High", "Critical"])
    ].sort_values(
        "Risk Score",
        ascending=False
    )

    if not priority_students.empty:

        st.dataframe(
            priority_students[[
                "Student ID",
                "Student",
                "Attendance",
                "Marks",
                "Assignments",
                "Risk",
                "Risk Score"
            ]],
            use_container_width=True,
            hide_index=True
        )

        priority_csv = priority_students.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "🚨 Download At-Risk Students Report",
            data=priority_csv,
            file_name="at_risk_students_report.csv",
            mime="text/csv"
        )

    else:
        st.success("No High or Critical risk students in this report.")
