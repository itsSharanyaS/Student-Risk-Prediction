
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. LIGHT / DARK THEME SWITCH
# =========================================================

if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Light"

st.sidebar.markdown("## 🎨 Appearance")

theme_choice = st.sidebar.radio(
    "Select theme",
    ["☀️ Light", "🌙 Dark"],
    index=0 if st.session_state.app_theme == "Light" else 1
)

st.session_state.app_theme = (
    "Light" if theme_choice == "☀️ Light" else "Dark"
)

is_dark = st.session_state.app_theme == "Dark"

if is_dark:
    bg_color = "#0F172A"
    card_color = "#1E293B"
    sidebar_color = "#111827"
    text_color = "#F1F5F9"
    secondary_text = "#CBD5E1"
    border_color = "#334155"
    chart_template = "plotly_dark"
else:
    bg_color = "#F4F7FB"
    card_color = "#FFFFFF"
    sidebar_color = "#FFFFFF"
    text_color = "#1E293B"
    secondary_text = "#64748B"
    border_color = "#E2E8F0"
    chart_template = "plotly_white"

# =========================================================
# 3. DYNAMIC CSS
# =========================================================

st.markdown(f"""
<style>

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_color};
    border-right: 1px solid {border_color};
}}

[data-testid="stMetric"] {{
    background-color: {card_color};
    border: 1px solid {border_color};
    padding: 18px;
    border-radius: 14px;
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {{
    color: {text_color} !important;
}}

h1, h2, h3, h4 {{
    color: {text_color} !important;
}}

p, label {{
    color: {text_color};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {border_color};
    border-radius: 10px;
}}

hr {{
    border-color: {border_color};
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. RISK COLORS
# =========================================================

risk_colors = {
    "Low": "#16A34A",
    "Moderate": "#D97706",
    "High": "#EA580C",
    "Critical": "#DC2626"
}

risk_order = ["Low", "Moderate", "High", "Critical"]

# =========================================================
# 5. RISK CALCULATION
# =========================================================

def calculate_risk(row):

    attendance = row["Attendance"]
    marks = row["Marks"]
    assignments = row["Assignments"]

    if attendance < 50 or marks < 35:
        return "Critical"

    elif attendance < 60 or marks < 50 or assignments < 50:
        return "High"

    elif attendance < 80 or marks < 75 or assignments < 75:
        return "Moderate"

    else:
        return "Low"


def calculate_risk_score(row):

    score = 0

    if row["Attendance"] < 50:
        score += 40
    elif row["Attendance"] < 60:
        score += 30
    elif row["Attendance"] < 75:
        score += 15

    if row["Marks"] < 35:
        score += 35
    elif row["Marks"] < 50:
        score += 25
    elif row["Marks"] < 65:
        score += 15

    if row["Assignments"] < 40:
        score += 25
    elif row["Assignments"] < 60:
        score += 15
    elif row["Assignments"] < 75:
        score += 10

    return min(score, 100)


# =========================================================
# 6. LOAD STUDENT DATA
# =========================================================

@st.cache_data
def load_students():

    file_path = Path("data/students.csv")

    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "student_id",
        "name",
        "department",
        "semester",
        "attendance",
        "marks",
        "assignments"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing_columns)
        )

    df = df.rename(columns={
        "student_id": "Student ID",
        "name": "Student",
        "department": "Department",
        "semester": "Semester",
        "attendance": "Attendance",
        "marks": "Marks",
        "assignments": "Assignments"
    })

    for col in ["Attendance", "Marks", "Assignments", "Semester"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "Student ID",
            "Student",
            "Attendance",
            "Marks",
            "Assignments"
        ]
    )

    # Keep percentages within a valid range
    for col in ["Attendance", "Marks", "Assignments"]:
        df[col] = df[col].clip(0, 100)

    df["Risk"] = df.apply(calculate_risk, axis=1)
    df["Risk Score"] = df.apply(calculate_risk_score, axis=1)

    return df


try:
    students = load_students()

except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

if students is None:
    st.error("Student dataset not found.")
    st.info("Make sure data/students.csv exists in your GitHub repository.")
    st.stop()

if students.empty:
    st.error("The student dataset is empty or contains no valid records.")
    st.stop()

# =========================================================
# 7. GLOBAL METRICS
# =========================================================

total_students = len(students)

low_count = (students["Risk"] == "Low").sum()
moderate_count = (students["Risk"] == "Moderate").sum()
high_count = (students["Risk"] == "High").sum()
critical_count = (students["Risk"] == "Critical").sum()

at_risk_count = high_count + critical_count

risk_percentage = (
    (at_risk_count / total_students) * 100
    if total_students > 0 else 0
)

average_attendance = students["Attendance"].mean()
average_marks = students["Marks"].mean()
average_assignments = students["Assignments"].mean()

# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.divider()

st.sidebar.title("🎓 Student Intelligence")

st.sidebar.caption(
    "Academic monitoring and risk analysis platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Directory",
        "Risk Analytics",
        "Early Warning Center",
        "Recovery Plans",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Risk Legend")
st.sidebar.markdown("🟢 Low")
st.sidebar.markdown("🟡 Moderate")
st.sidebar.markdown("🟠 High")
st.sidebar.markdown("🔴 Critical")

st.sidebar.divider()

st.sidebar.metric("Students Monitored", total_students)

st.sidebar.caption("Student Risk Prediction System")

# =========================================================
# 9. COMMON CHART STYLE
# =========================================================

def apply_chart_theme(fig, height=400):

    fig.update_layout(
        template=chart_template,
        paper_bgcolor=bg_color,
        plot_bgcolor=card_color,
        font=dict(color=text_color),
        height=height,
        margin=dict(l=30, r=20, t=60, b=40)
    )

    return fig

# =========================================================
# 10. SMART DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("🎓 Student Academic Intelligence")

    st.caption(
        "Academic monitoring • Risk intelligence • Student success analytics"
    )

    st.divider()

    # KPI CARDS

    st.subheader("📌 Academic Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        total_students
    )

    col2.metric(
        "Students at Risk",
        at_risk_count,
        f"{risk_percentage:.1f}% of students"
    )

    col3.metric(
        "Average Attendance",
        f"{average_attendance:.1f}%"
    )

    col4.metric(
        "Average Marks",
        f"{average_marks:.1f}%"
    )

    st.divider()

    # ACADEMIC INDICATORS

    st.subheader("📚 Academic Performance Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Attendance**")
        st.progress(float(np.clip(average_attendance / 100, 0, 1)))
        st.write(f"{average_attendance:.1f}%")

    with col2:
        st.markdown("**Marks**")
        st.progress(float(np.clip(average_marks / 100, 0, 1)))
        st.write(f"{average_marks:.1f}%")

    with col3:
        st.markdown("**Assignments**")
        st.progress(float(np.clip(average_assignments / 100, 0, 1)))
        st.write(f"{average_assignments:.1f}%")

    st.divider()

    # RISK DISTRIBUTION

    st.subheader("🚨 Risk Intelligence Overview")

    col1, col2 = st.columns(2)

    risk_data = pd.DataFrame({
        "Risk Level": risk_order,
        "Students": [
            low_count,
            moderate_count,
            high_count,
            critical_count
        ]
    })

    with col1:

        fig = px.pie(
            risk_data,
            names="Risk Level",
            values="Students",
            hole=0.55,
            color="Risk Level",
            color_discrete_map=risk_colors
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        apply_chart_theme(fig, 400)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            risk_data,
            x="Risk Level",
            y="Students",
            color="Risk Level",
            text="Students",
            color_discrete_map=risk_colors
        )

        apply_chart_theme(fig, 400)

        st.plotly_chart(fig, use_container_width=True)

    # RISK SUMMARY

    st.subheader("Risk Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🟢 Low", low_count)
    col2.metric("🟡 Moderate", moderate_count)
    col3.metric("🟠 High", high_count)
    col4.metric("🔴 Critical", critical_count)

    if critical_count > 0:
        st.error(f"{critical_count} students require urgent attention.")

    elif high_count > 0:
        st.warning(f"{high_count} high-risk students require monitoring.")

    else:
        st.success("No high or critical risk students detected.")

    st.divider()

    # ATTENDANCE VS MARKS

    st.subheader("📈 Academic Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            students,
            x="Attendance",
            y="Marks",
            size="Assignments",
            color="Risk",
            color_discrete_map=risk_colors,
            hover_name="Student",
            hover_data=["Student ID", "Assignments", "Risk Score"],
            title="Attendance vs Marks"
        )

        apply_chart_theme(fig, 400)

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        performance_data = pd.DataFrame({
            "Metric": ["Attendance", "Marks", "Assignments"],
            "Average": [
                average_attendance,
                average_marks,
                average_assignments
            ]
        })

        fig = px.bar(
            performance_data,
            x="Metric",
            y="Average",
            color="Metric",
            text_auto=".1f",
            title="Average Performance Indicators"
        )

        fig.update_layout(yaxis_range=[0, 100])

        apply_chart_theme(fig, 400)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # STUDENTS NEEDING ATTENTION

    st.subheader("🚨 Priority Student Monitoring")

    priority_students = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values(
        "Risk Score",
        ascending=False
    )

    if not priority_students.empty:

        st.dataframe(
            priority_students[
                [
                    "Student ID",
                    "Student",
                    "Attendance",
                    "Marks",
                    "Assignments",
                    "Risk",
                    "Risk Score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success("No high-priority students at this time.")

    st.divider()

    # TOP PERFORMERS

    st.subheader("🏆 Top Academic Performers")

    top_students = students.sort_values(
        "Marks",
        ascending=False
    ).head(5)

    st.dataframe(
        top_students[
            [
                "Student ID",
                "Student",
                "Attendance",
                "Marks",
                "Assignments",
                "Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # INSIGHTS

    st.subheader("💡 Academic Insights")

    col1, col2 = st.columns(2)

    with col1:

        if average_attendance < 75:
            st.warning(
                f"Average attendance is {average_attendance:.1f}%. "
                "Attendance support may be needed."
            )
        else:
            st.success(
                f"Average attendance is {average_attendance:.1f}%."
            )

        if average_marks < 50:
            st.warning(
                f"Average marks are {average_marks:.1f}%. "
                "Academic support may be beneficial."
            )
        else:
            st.success(
                f"Average marks are {average_marks:.1f}%."
            )

    with col2:

        if at_risk_count > 0:
            st.warning(
                f"{at_risk_count} students are classified High or Critical."
            )
        else:
            st.success("No High or Critical risk students.")

        if average_assignments < 60:
            st.warning("Assignment performance needs attention.")
        else:
            st.info("Assignment performance is satisfactory.")

    st.divider()

    # SEMESTER ANALYSIS

    st.subheader("🎓 Semester Performance Summary")

    semester_summary = students.groupby("Semester").agg(
        Total_Students=("Student ID", "count"),
        Average_Attendance=("Attendance", "mean"),
        Average_Marks=("Marks", "mean"),
        Average_Assignments=("Assignments", "mean"),
        At_Risk_Students=(
            "Risk",
            lambda x: x.isin(["High", "Critical"]).sum()
        )
    ).reset_index()

    semester_summary = semester_summary.rename(columns={
        "Total_Students": "Total Students",
        "Average_Attendance": "Average Attendance",
        "Average_Marks": "Average Marks",
        "Average_Assignments": "Average Assignments",
        "At_Risk_Students": "At-Risk Students"
    })

    st.dataframe(
        semester_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Risk classification is currently rule-based, not generated by a trained ML model."
    )

# =========================================================
# 11. STUDENT DIRECTORY
# =========================================================

elif page == "Student Directory":

    st.title("👨‍🎓 Student Directory")

    st.caption("Search and explore student academic information.")

    col1, col2 = st.columns(2)

    with col1:
        search_query = st.text_input("🔍 Search student or USN")

    with col2:
        selected_risk = st.selectbox(
            "Filter Risk",
            ["All", "Low", "Moderate", "High", "Critical"]
        )

    filtered_students = students.copy()

    if search_query:

        filtered_students = filtered_students[
            filtered_students["Student"].str.contains(
                search_query,
                case=False,
                na=False
            )
            |
            filtered_students["Student ID"].str.contains(
                search_query,
                case=False,
                na=False
            )
        ]

    if selected_risk != "All":

        filtered_students = filtered_students[
            filtered_students["Risk"] == selected_risk
        ]

    st.metric("Students Found", len(filtered_students))

    st.dataframe(
        filtered_students[
            [
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
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🔎 Student Profile")

    if not filtered_students.empty:

        selected_student = st.selectbox(
            "Choose Student",
            filtered_students["Student"].tolist()
        )

        profile = filtered_students[
            filtered_students["Student"] == selected_student
        ].iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Attendance", f"{profile['Attendance']}%")
        col2.metric("Marks", f"{profile['Marks']}%")
        col3.metric("Assignments", f"{profile['Assignments']}%")
        col4.metric("Risk Score", profile["Risk Score"])

        st.markdown(f"### {profile['Student']}")

        st.write("**USN:**", profile["Student ID"])
        st.write("**Department:**", profile["Department"])
        st.write("**Semester:**", profile["Semester"])
        st.write("**Risk Level:**", profile["Risk"])

        profile_data = pd.DataFrame({
            "Metric": ["Attendance", "Marks", "Assignments"],
            "Value": [
                profile["Attendance"],
                profile["Marks"],
                profile["Assignments"]
            ]
        })

        fig = px.bar(
            profile_data,
            x="Metric",
            y="Value",
            color="Metric",
            text_auto=".1f"
        )

        fig.update_layout(yaxis_range=[0, 100])

        apply_chart_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No students match your search.")

# =========================================================
# 12. RISK ANALYTICS
# =========================================================

elif page == "Risk Analytics":

    st.title("📊 Academic Risk Analytics")

    st.caption("Explore academic indicators and risk patterns.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Attendance", f"{average_attendance:.1f}%")
    col2.metric("Average Marks", f"{average_marks:.1f}%")
    col3.metric("Average Assignments", f"{average_assignments:.1f}%")

    st.divider()

    st.subheader("Risk Score Distribution")

    fig = px.histogram(
        students,
        x="Risk Score",
        color="Risk",
        nbins=10,
        color_discrete_map=risk_colors
    )

    apply_chart_theme(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Academic Performance Distribution")

    performance_data = students[
        ["Student", "Attendance", "Marks", "Assignments"]
    ].melt(
        id_vars="Student",
        var_name="Metric",
        value_name="Value"
    )

    fig = px.box(
        performance_data,
        x="Metric",
        y="Value",
        color="Metric"
    )

    apply_chart_theme(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk-Level Academic Comparison")

    risk_analysis = students.groupby("Risk")[
        ["Attendance", "Marks", "Assignments"]
    ].mean().reindex(risk_order)

    st.dataframe(
        risk_analysis.round(2),
        use_container_width=True
    )

# =========================================================
# 13. EARLY WARNING CENTER
# =========================================================

elif page == "Early Warning Center":

    st.title("🚨 Early Warning Center")

    st.caption("Monitor attendance, marks and assignment concerns.")

    attendance_alerts = students[
        students["Attendance"] < 60
    ]

    marks_alerts = students[
        students["Marks"] < 50
    ]

    assignment_alerts = students[
        students["Assignments"] < 50
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Attendance Alerts", len(attendance_alerts))
    col2.metric("Marks Alerts", len(marks_alerts))
    col3.metric("Assignment Alerts", len(assignment_alerts))

    st.divider()

    st.subheader("⚠️ Attendance Below 60%")

    st.dataframe(
        attendance_alerts[
            ["Student ID", "Student", "Attendance", "Marks", "Risk"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📉 Marks Below 50%")

    st.dataframe(
        marks_alerts[
            ["Student ID", "Student", "Marks", "Attendance", "Risk"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📝 Assignment Scores Below 50%")

    st.dataframe(
        assignment_alerts[
            ["Student ID", "Student", "Assignments", "Risk"]
        ],
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# 14. RECOVERY PLANS
# =========================================================

elif page == "Recovery Plans":

    st.title("🛠️ Student Recovery Plans")

    st.caption("Personalized academic support and mentor follow-up.")

    recovery_students = students[
        students["Risk"].isin(["High", "Critical", "Moderate"])
    ].sort_values(
        "Risk Score",
        ascending=False
    )

    if recovery_students.empty:

        st.success("No students currently require recovery plans.")

    else:

        selected_student = st.selectbox(
            "Select Student",
            recovery_students["Student"].tolist()
        )

        student = recovery_students[
            recovery_students["Student"] == selected_student
        ].iloc[0]

        st.divider()

        st.subheader("Student Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Attendance", f"{student['Attendance']}%")
        col2.metric("Marks", f"{student['Marks']}%")
        col3.metric("Assignments", f"{student['Assignments']}%")
        col4.metric("Risk Level", student["Risk"])

        st.subheader("🎯 Recommended Actions")

        if student["Attendance"] < 75:

            st.warning("Attendance Improvement Plan")

            st.write("- Attend upcoming classes regularly.")
            st.write("- Meet the mentor weekly.")
            st.write("- Review attendance every week.")

        if student["Marks"] < 75:

            st.warning("Academic Improvement Plan")

            st.write("- Create a subject-wise study timetable.")
            st.write("- Attend remedial sessions.")
            st.write("- Practice previous examination questions.")

        if student["Assignments"] < 75:

            st.warning("Assignment Improvement Plan")

            st.write("- Complete pending assignments.")
            st.write("- Submit work on time.")
            st.write("- Review faculty feedback.")

        st.divider()

        st.subheader("✅ Mentor Follow-up Checklist")

        st.checkbox("Student contacted")
        st.checkbox("Mentor meeting completed")
        st.checkbox("Study plan created")
        st.checkbox("Attendance reviewed")
        st.checkbox("Academic progress reviewed")
        st.checkbox("Follow-up scheduled")

        st.info(
            "Recommended actions should be reviewed by faculty or mentors."
        )

# =========================================================
# 15. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("🎓 About Student Risk Prediction")

    st.markdown("""
    ## AI-Powered Student Risk Prediction & Academic Intelligence Platform

    A student academic monitoring platform designed to identify
    students who may require additional academic support.

    ### Current Features

    - Academic dashboard
    - Student directory
    - Student profiles
    - Risk classification
    - Risk score calculation
    - Risk analytics
    - Attendance monitoring
    - Marks monitoring
    - Assignment monitoring
    - Early warning center
    - Academic recovery plans
    - Mentor follow-up checklist

    ### Academic Indicators

    - Attendance percentage
    - Academic marks
    - Assignment performance

    ### Risk Categories

    🟢 Low Risk

    🟡 Moderate Risk

    🟠 High Risk

    🔴 Critical Risk

    ### Technology Stack

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-learn

    ### Planned Enhancements

    - Machine Learning risk prediction
    - Explainable AI
    - Risk forecasting
    - Student clustering
    - What-if simulator
    - Intervention effectiveness analysis
    - Database integration
    - Role-based login
    - Automated reports
    - Personalized recovery pathways
    """)

    st.divider()

    st.info(
        "Current risk predictions are rule-based. "
        "Machine learning integration is planned for a later phase."
    )

# =========================================================
# 16. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption("Student Risk Prediction")
st.sidebar.caption("Academic Intelligence Platform")
