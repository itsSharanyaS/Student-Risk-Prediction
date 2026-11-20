
import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1F2937;
}

[data-testid="stMetric"] {
    background-color: #172033;
    padding: 18px;
    border: 1px solid #273449;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263449;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SAMPLE STUDENT DATA
# =========================================================
students = pd.DataFrame({
    "Student ID": [
        "STU001", "STU002", "STU003", "STU004",
        "STU005", "STU006", "STU007", "STU008",
        "STU009", "STU010", "STU011", "STU012"
    ],

    "Student": [
        "Aarav Sharma", "Ananya Rao", "Rohan Kumar",
        "Meera Nair", "Arjun Patel", "Sneha Iyer",
        "Kiran Reddy", "Diya Shah", "Rahul Das",
        "Pooja Singh", "Vikram Joshi", "Nisha Gowda"
    ],

    "Department": [
        "CSE", "CSE", "ISE", "CSE", "ECE", "ISE",
        "CSE", "ECE", "ISE", "CSE", "ECE", "CSE"
    ],

    "Attendance": [
        92, 78, 56, 88, 64, 95, 49, 81, 58, 90, 72, 45
    ],

    "Marks": [
        89, 76, 42, 91, 54, 96, 35, 73, 48, 85, 68, 32
    ],

    "Assignments": [
        95, 80, 45, 90, 60, 98, 35, 75, 50, 88, 70, 40
    ],

    "Risk": [
        "Low", "Moderate", "High", "Low",
        "Moderate", "Low", "Critical", "Moderate",
        "High", "Low", "Moderate", "Critical"
    ]
})

# =========================================================
# RISK COLORS
# =========================================================
RISK_COLORS = {
    "Low": "#10B981",
    "Moderate": "#F59E0B",
    "High": "#F97316",
    "Critical": "#EF4444"
}

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("# 🎓 Student Risk AI")
st.sidebar.caption("Academic Intelligence Platform")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Student Directory",
        "Risk Analytics",
        "Early Warning Center",
        "Recovery Plans"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### System Status")
st.sidebar.success("System Online")

st.sidebar.markdown("---")
st.sidebar.caption("Student Risk Prediction")
st.sidebar.caption("Academic Intelligence Platform v1.0")

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    st.title("Academic Intelligence Dashboard")

    st.markdown(
        "Monitor academic performance, identify students at risk "
        "and support better academic outcomes."
    )

    # ---------------- METRICS ----------------
    total_students = len(students)

    high_risk_students = students[
        students["Risk"].isin(["High", "Critical"])
    ]

    high_risk_count = len(high_risk_students)

    average_attendance = students["Attendance"].mean()

    average_marks = students["Marks"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        total_students,
        help="Total students in the sample dataset"
    )

    col2.metric(
        "Students At Risk",
        high_risk_count,
        help="Students classified as High or Critical"
    )

    col3.metric(
        "Average Attendance",
        f"{average_attendance:.1f}%"
    )

    col4.metric(
        "Average Academic Score",
        f"{average_marks:.1f}%"
    )

    st.markdown("---")

    # ---------------- CHARTS ----------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader("Student Risk Distribution")

        risk_counts = (
            students["Risk"]
            .value_counts()
            .reindex(["Low", "Moderate", "High", "Critical"], fill_value=0)
            .reset_index()
        )

        risk_counts.columns = ["Risk Level", "Students"]

        fig_pie = px.pie(
            risk_counts,
            names="Risk Level",
            values="Students",
            hole=0.65,
            color="Risk Level",
            color_discrete_map=RISK_COLORS
        )

        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
            margin=dict(t=30, b=20, l=20, r=20)
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with chart_col2:

        st.subheader("Attendance vs Academic Performance")

        fig_scatter = px.scatter(
            students,
            x="Attendance",
            y="Marks",
            color="Risk",
            color_discrete_map=RISK_COLORS,
            hover_name="Student",
            hover_data=["Department", "Assignments"],
            size="Assignments",
            size_max=18
        )

        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            xaxis_title="Attendance (%)",
            yaxis_title="Academic Marks (%)"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    # ---------------- RISK ALERT TABLE ----------------
    st.subheader("🚨 Students Requiring Attention")

    alerts = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values("Attendance")

    st.dataframe(
        alerts[
            ["Student ID", "Student", "Department",
             "Attendance", "Marks", "Risk"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Note: This dashboard currently uses sample data. "
        "Risk classifications are illustrative and are not generated "
        "by a trained machine learning model."
    )

# =========================================================
# STUDENT DIRECTORY
# =========================================================
elif page == "Student Directory":

    st.title("Student Directory")

    st.caption("Search, filter and explore student academic records.")

    search = st.text_input(
        "🔍 Search by student name or ID",
        placeholder="Enter name or student ID..."
    )

    department_filter = st.selectbox(
        "Filter by Department",
        ["All Departments"] + sorted(students["Department"].unique().tolist())
    )

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All Risk Levels", "Low", "Moderate", "High", "Critical"]
    )

    filtered_students = students.copy()

    if search:
        filtered_students = filtered_students[
            filtered_students["Student"].str.contains(
                search, case=False, na=False
            )
            |
            filtered_students["Student ID"].str.contains(
                search, case=False, na=False
            )
        ]

    if department_filter != "All Departments":
        filtered_students = filtered_students[
            filtered_students["Department"] == department_filter
        ]

    if risk_filter != "All Risk Levels":
        filtered_students = filtered_students[
            filtered_students["Risk"] == risk_filter
        ]

    st.metric("Matching Students", len(filtered_students))

    st.dataframe(
        filtered_students,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# RISK ANALYTICS
# =========================================================
elif page == "Risk Analytics":

    st.title("Risk Analytics")

    st.caption(
        "Explore student academic performance and department-level patterns."
    )

    # Marks comparison
    st.subheader("Student Performance Comparison")

    fig_marks = px.bar(
        students,
        x="Student",
        y="Marks",
        color="Risk",
        color_discrete_map=RISK_COLORS,
        hover_data=["Department", "Attendance"]
    )

    fig_marks.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        xaxis_title="Student",
        yaxis_title="Marks (%)"
    )

    st.plotly_chart(
        fig_marks,
        use_container_width=True
    )

    # Department analysis
    st.subheader("Department Performance")

    department_analysis = students.groupby(
        "Department"
    ).agg(
        Students=("Student", "count"),
        Average_Marks=("Marks", "mean"),
        Average_Attendance=("Attendance", "mean"),
        Average_Assignments=("Assignments", "mean")
    ).reset_index()

    department_analysis = department_analysis.round(2)

    st.dataframe(
        department_analysis,
        use_container_width=True,
        hide_index=True
    )

    # Attendance chart
    st.subheader("Department Attendance")

    fig_attendance = px.bar(
        department_analysis,
        x="Department",
        y="Average_Attendance",
        text="Average_Attendance"
    )

    fig_attendance.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        yaxis_title="Average Attendance (%)"
    )

    st.plotly_chart(
        fig_attendance,
        use_container_width=True
    )

# =========================================================
# EARLY WARNING CENTER
# =========================================================
elif page == "Early Warning Center":

    st.title("🚨 Early Warning Center")

    st.caption(
        "Identify students who may require academic attention."
    )

    warnings = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values("Marks")

    if warnings.empty:

        st.success("No students currently require attention.")

    else:

        st.warning(
            f"{len(warnings)} students require academic review."
        )

        for _, row in warnings.iterrows():

            with st.expander(
                f"{row['Student']} — {row['Risk']} Risk"
            ):

                st.write(f"**Student ID:** {row['Student ID']}")
                st.write(f"**Department:** {row['Department']}")
                st.write(f"**Attendance:** {row['Attendance']}%")
                st.write(f"**Marks:** {row['Marks']}%")
                st.write(f"**Assignments:** {row['Assignments']}%")

                if row["Attendance"] < 60:

                    st.warning(
                        "Attendance is below 60%. "
                        "Attendance improvement is recommended."
                    )

                if row["Marks"] < 50:

                    st.warning(
                        "Academic performance is low. "
                        "Subject-specific support is recommended."
                    )

                if row["Assignments"] < 50:

                    st.warning(
                        "Assignment completion is low. "
                        "Pending assignments should be reviewed."
                    )

                st.info(
                    "Suggested action: Faculty mentor review, "
                    "academic counseling and a personalized support plan."
                )

# =========================================================
# RECOVERY PLANS
# =========================================================
elif page == "Recovery Plans":

    st.title("Student Recovery & Support Plans")

    st.caption(
        "Track supportive actions that may help students improve."
    )

    recovery_students = students[
        students["Risk"].isin(["High", "Critical"])
    ]

    if recovery_students.empty:

        st.success("No recovery plans required.")

    else:

        for _, row in recovery_students.iterrows():

            st.subheader(row["Student"])

            st.write(
                f"**Risk Level:** {row['Risk']}  |  "
                f"**Attendance:** {row['Attendance']}%  |  "
                f"**Marks:** {row['Marks']}%"
            )

            st.write("### Recommended Action Plan")

            tasks = [
                "Schedule a meeting with faculty mentor",
                "Improve class attendance",
                "Complete pending assignments",
                "Attend remedial classes for weak subjects",
                "Create a weekly study timetable",
                "Review academic progress with mentor"
            ]

            for task_index, task in enumerate(tasks):

                st.checkbox(
                    task,
                    key=f"{row['Student ID']}_{task_index}"
                )

            st.divider()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Student Risk Prediction | AI-Powered Academic Intelligence Platform"
)
