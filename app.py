
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# =========================================================
# PAGE CONFIG
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
}
.metric-card {
    background: linear-gradient(135deg, #172033, #202B43);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #2B3953;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #F8FAFC;
}
.small-text {
    color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# RISK CALCULATION
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

    # Attendance risk
    if row["Attendance"] < 50:
        score += 40
    elif row["Attendance"] < 60:
        score += 30
    elif row["Attendance"] < 75:
        score += 15

    # Marks risk
    if row["Marks"] < 35:
        score += 35
    elif row["Marks"] < 50:
        score += 25
    elif row["Marks"] < 65:
        score += 15

    # Assignment risk
    if row["Assignments"] < 40:
        score += 25
    elif row["Assignments"] < 60:
        score += 15
    elif row["Assignments"] < 75:
        score += 10

    return min(score, 100)


# =========================================================
# LOAD DATASET
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
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "Missing columns in students.csv: "
            + ", ".join(missing_columns)
        )
        st.stop()

    df = df.rename(columns={
        "student_id": "Student ID",
        "name": "Student",
        "department": "Department",
        "semester": "Semester",
        "attendance": "Attendance",
        "marks": "Marks",
        "assignments": "Assignments"
    })

    numeric_columns = [
        "Attendance",
        "Marks",
        "Assignments",
        "Semester"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Student ID",
            "Student",
            "Attendance",
            "Marks",
            "Assignments"
        ]
    )

    df["Risk"] = df.apply(calculate_risk, axis=1)

    df["Risk Score"] = df.apply(
        calculate_risk_score,
        axis=1
    )

    return df


students = load_students()

if students is None:
    st.error("students.csv not found!")
    st.info(
        "Please create the file data/students.csv "
        "inside your GitHub repository."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Student Intelligence")
st.sidebar.caption("Academic Risk Monitoring System")

st.sidebar.divider()

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

st.sidebar.markdown("### 🎯 Risk Categories")

st.sidebar.markdown("🟢 **Low Risk**")
st.sidebar.markdown("🟡 **Moderate Risk**")
st.sidebar.markdown("🟠 **High Risk**")
st.sidebar.markdown("🔴 **Critical Risk**")

st.sidebar.divider()

st.sidebar.caption("AI-Powered Student Risk Prediction")


# =========================================================
# COLOR SETTINGS
# =========================================================

risk_colors = {
    "Low": "#22C55E",
    "Moderate": "#F59E0B",
    "High": "#F97316",
    "Critical": "#EF4444"
}

risk_order = ["Low", "Moderate", "High", "Critical"]


# =========================================================
# COMMON METRICS
# =========================================================

total_students = len(students)

low_count = (students["Risk"] == "Low").sum()
moderate_count = (students["Risk"] == "Moderate").sum()
high_count = (students["Risk"] == "High").sum()
critical_count = (students["Risk"] == "Critical").sum()

at_risk_count = high_count + critical_count

average_attendance = students["Attendance"].mean()
average_marks = students["Marks"].mean()
average_assignments = students["Assignments"].mean()


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("🎓 Student Risk Prediction Dashboard")

    st.markdown(
        "Monitor academic performance, identify at-risk students, "
        "and understand student progress."
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        total_students
    )

    col2.metric(
        "Students at Risk",
        at_risk_count
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

    # RISK DISTRIBUTION

    st.subheader("📊 Student Risk Distribution")

    risk_distribution = (
        students["Risk"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .reset_index()
    )

    risk_distribution.columns = ["Risk Level", "Students"]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            risk_distribution,
            names="Risk Level",
            values="Students",
            color="Risk Level",
            color_discrete_map=risk_colors,
            hole=0.55
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Students",
            color="Risk Level",
            color_discrete_map=risk_colors,
            text="Students"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # PERFORMANCE ANALYSIS

    st.subheader("📈 Attendance vs Academic Performance")

    fig = px.scatter(
        students,
        x="Attendance",
        y="Marks",
        color="Risk",
        color_discrete_map=risk_colors,
        hover_data=[
            "Student",
            "Student ID",
            "Assignments"
        ],
        size="Assignments",
        title="Student Performance Analysis"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # STUDENTS NEEDING ATTENTION

    st.subheader("🚨 Students Requiring Attention")

    attention_students = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values(
        by="Risk Score",
        ascending=False
    )

    if len(attention_students) > 0:

        st.dataframe(
            attention_students[
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

        st.success("No high-risk students identified.")

    st.divider()

    st.caption(
        "Risk levels are currently generated using preliminary "
        "rule-based thresholds. They are not ML model predictions."
    )


# =========================================================
# STUDENT DIRECTORY
# =========================================================

elif page == "Student Directory":

    st.title("👨‍🎓 Student Directory")

    st.write(
        "Search, filter, and explore student academic information."
    )

    col1, col2 = st.columns(2)

    with col1:

        search_query = st.text_input(
            "🔍 Search Student Name or USN"
        )

    with col2:

        selected_risk = st.selectbox(
            "Filter by Risk Level",
            [
                "All",
                "Low",
                "Moderate",
                "High",
                "Critical"
            ]
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

    st.metric(
        "Students Found",
        len(filtered_students)
    )

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

    st.subheader("🔎 Individual Student Profile")

    if len(filtered_students) > 0:

        selected_student = st.selectbox(
            "Select a student",
            filtered_students["Student"].tolist()
        )

        profile = filtered_students[
            filtered_students["Student"] == selected_student
        ].iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Attendance",
            f"{profile['Attendance']}%"
        )

        col2.metric(
            "Marks",
            f"{profile['Marks']}%"
        )

        col3.metric(
            "Assignments",
            f"{profile['Assignments']}%"
        )

        col4.metric(
            "Risk Score",
            profile["Risk Score"]
        )

        st.markdown(f"### Student: {profile['Student']}")

        st.write("**USN:**", profile["Student ID"])
        st.write("**Department:**", profile["Department"])
        st.write("**Semester:**", profile["Semester"])

        risk = profile["Risk"]

        if risk == "Low":
            st.success(f"Risk Status: {risk}")

        elif risk == "Moderate":
            st.warning(f"Risk Status: {risk}")

        else:
            st.error(f"Risk Status: {risk}")

        profile_data = pd.DataFrame({
            "Metric": [
                "Attendance",
                "Marks",
                "Assignments"
            ],
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
            text="Value",
            color="Metric"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# RISK ANALYTICS
# =========================================================

elif page == "Risk Analytics":

    st.title("📊 Academic Risk Analytics")

    st.write(
        "Explore academic indicators and identify performance patterns."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Attendance",
        f"{average_attendance:.2f}%"
    )

    col2.metric(
        "Average Marks",
        f"{average_marks:.2f}%"
    )

    col3.metric(
        "Average Assignments",
        f"{average_assignments:.2f}%"
    )

    st.divider()

    st.subheader("📉 Risk Score Distribution")

    fig = px.histogram(
        students,
        x="Risk Score",
        nbins=10,
        color="Risk",
        color_discrete_map=risk_colors
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📚 Academic Performance Comparison")

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

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("🎯 Risk-Level Performance")

    risk_analysis = students.groupby("Risk")[
        ["Attendance", "Marks", "Assignments"]
    ].mean().reindex(risk_order)

    st.dataframe(
        risk_analysis.round(2),
        use_container_width=True
    )


# =========================================================
# EARLY WARNING CENTER
# =========================================================

elif page == "Early Warning Center":

    st.title("🚨 Early Warning Center")

    st.write(
        "Identify students with low attendance, poor marks, "
        "or incomplete academic activities."
    )

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

    col1.metric(
        "Attendance Alerts",
        len(attendance_alerts)
    )

    col2.metric(
        "Academic Alerts",
        len(marks_alerts)
    )

    col3.metric(
        "Assignment Alerts",
        len(assignment_alerts)
    )

    st.divider()

    st.subheader("⚠️ Students with Attendance Below 60%")

    st.dataframe(
        attendance_alerts[
            [
                "Student ID",
                "Student",
                "Attendance",
                "Marks",
                "Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📉 Students with Marks Below 50%")

    st.dataframe(
        marks_alerts[
            [
                "Student ID",
                "Student",
                "Marks",
                "Attendance",
                "Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📝 Students with Assignment Scores Below 50%")

    st.dataframe(
        assignment_alerts[
            [
                "Student ID",
                "Student",
                "Assignments",
                "Risk"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# RECOVERY PLANS
# =========================================================

elif page == "Recovery Plans":

    st.title("🛠️ Student Recovery Plans")

    st.write(
        "Suggested academic improvement actions for students "
        "who require additional support."
    )

    recovery_students = students[
        students["Risk"].isin(["High", "Critical", "Moderate"])
    ].sort_values(
        by="Risk Score",
        ascending=False
    )

    if len(recovery_students) == 0:

        st.success(
            "No students currently require recovery plans."
        )

    else:

        selected_student = st.selectbox(
            "Select Student",
            recovery_students["Student"].tolist()
        )

        student = recovery_students[
            recovery_students["Student"] == selected_student
        ].iloc[0]

        st.divider()

        st.subheader("👤 Student Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Attendance",
            f"{student['Attendance']}%"
        )

        col2.metric(
            "Marks",
            f"{student['Marks']}%"
        )

        col3.metric(
            "Assignments",
            f"{student['Assignments']}%"
        )

        col4.metric(
            "Risk",
            student["Risk"]
        )

        st.subheader("🎯 Recommended Actions")

        if student["Attendance"] < 75:

            st.warning("📅 Attendance Improvement Plan")

            st.write(
                "- Attend all upcoming classes."
            )

            st.write(
                "- Meet the class mentor weekly."
            )

            st.write(
                "- Monitor attendance every week."
            )

        if student["Marks"] < 75:

            st.warning("📚 Academic Improvement Plan")

            st.write(
                "- Create a subject-wise study timetable."
            )

            st.write(
                "- Attend remedial classes."
            )

            st.write(
                "- Practice previous examination questions."
            )

        if student["Assignments"] < 75:

            st.warning("📝 Assignment Improvement Plan")

            st.write(
                "- Complete pending assignments."
            )

            st.write(
                "- Submit work before deadlines."
            )

            st.write(
                "- Review assignment feedback."
            )

        st.divider()

        st.subheader("✅ Mentor Follow-up Checklist")

        st.checkbox("Student contacted")

        st.checkbox("Mentor meeting completed")

        st.checkbox("Study plan created")

        st.checkbox("Attendance reviewed")

        st.checkbox("Academic progress reviewed")

        st.checkbox("Follow-up scheduled")

        st.info(
            "These are suggested support actions. "
            "They should be reviewed by faculty or mentors."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("🎓 About Student Risk Prediction")

    st.markdown("""
    ### AI-Powered Student Risk Prediction & Academic Intelligence Platform

    This project is designed to analyze student academic indicators
    and identify students who may require academic support.

    ### Current Features

    - Student academic dashboard
    - Student directory and search
    - Risk classification
    - Risk score calculation
    - Academic analytics
    - Attendance monitoring
    - Assignment monitoring
    - Early warning center
    - Individual student profile
    - Recovery plans
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
    - Scikit-learn (planned for ML model development)

    ### Future Enhancements

    - Machine Learning predictions
    - Explainable AI
    - Student risk forecasting
    - Academic performance prediction
    - Student clustering
    - What-if simulation
    - Intervention effectiveness analytics
    - Database integration
    - Login and role-based access
    - PDF and Excel reports
    """)

    st.divider()

    st.caption(
        "Student Risk Prediction | Academic Intelligence Platform"
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Student Risk Prediction System"
)

st.sidebar.caption(
    "Prototype • Rule-Based Risk Analysis"
)
