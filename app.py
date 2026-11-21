
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from modules.anomaly_detection import render_anomaly_detection
from modules.recovery_pathway import render_recovery_pathway
from modules.faculty_dashboard import render_faculty_dashboard
from modules.intervention_effectiveness import render_intervention_effectiveness

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path("data/students.csv")

# =========================================================
# 2. LIGHT / DARK THEME
# =========================================================

if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Light"

st.sidebar.markdown("## 🎨 Appearance")

theme_choice = st.sidebar.radio(
    "Select theme",
    ["☀️ Light", "🌙 Dark"],
    index=0 if st.session_state.app_theme == "Light" else 1
)

st.session_state.app_theme = "Light" if theme_choice == "☀️ Light" else "Dark"
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

st.markdown(f"""
<style>
.stApp {{background-color: {bg_color}; color: {text_color};}}
[data-testid="stSidebar"] {{background-color: {sidebar_color}; border-right: 1px solid {border_color};}}
[data-testid="stMetric"] {{background-color: {card_color}; border: 1px solid {border_color}; padding: 18px; border-radius: 14px;}}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{color: {text_color} !important;}}
h1, h2, h3, h4 {{color: {text_color} !important;}}
p, label {{color: {text_color};}}
[data-testid="stDataFrame"] {{border: 1px solid {border_color}; border-radius: 10px;}}
hr {{border-color: {border_color};}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. RISK COLORS & CALCULATIONS
# =========================================================

risk_colors = {
    "Low": "#16A34A",
    "Moderate": "#D97706",
    "High": "#EA580C",
    "Critical": "#DC2626"
}

risk_order = ["Low", "Moderate", "High", "Critical"]


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


def refresh_student_data(df):
    df = df.copy()

    for col in ["Attendance", "Marks", "Assignments", "Semester"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "Student ID", "Student", "Attendance",
        "Marks", "Assignments", "Semester"
    ])

    for col in ["Attendance", "Marks", "Assignments"]:
        df[col] = df[col].clip(0, 100)

    df["Risk"] = df.apply(calculate_risk, axis=1)
    df["Risk Score"] = df.apply(calculate_risk_score, axis=1)

    return df


# =========================================================
# 4. LOAD DATA
# =========================================================

@st.cache_data
def load_students():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "student_id", "name", "department",
        "semester", "attendance", "marks", "assignments"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))

    df = df.rename(columns={
        "student_id": "Student ID",
        "name": "Student",
        "department": "Department",
        "semester": "Semester",
        "attendance": "Attendance",
        "marks": "Marks",
        "assignments": "Assignments"
    })

    return refresh_student_data(df)


try:
    loaded_students = load_students()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

if loaded_students is None:
    st.error("Dataset not found. Make sure data/students.csv exists.")
    st.stop()

if "student_data" not in st.session_state:
    st.session_state.student_data = loaded_students.copy()

students = refresh_student_data(st.session_state.student_data)

# =========================================================
# 5. DATA STORAGE HELPERS
# =========================================================

def save_student_data(df):
    df = refresh_student_data(df)

    st.session_state.student_data = df.copy()

    save_df = df.drop(columns=["Risk", "Risk Score"], errors="ignore")

    save_df = save_df.rename(columns={
        "Student ID": "student_id",
        "Student": "name",
        "Department": "department",
        "Semester": "semester",
        "Attendance": "attendance",
        "Marks": "marks",
        "Assignments": "assignments"
    })

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(DATA_PATH, index=False)

    st.cache_data.clear()


def generate_student_id(df):
    existing_ids = df["Student ID"].astype(str).tolist()

    numbers = []

    for student_id in existing_ids:
        if student_id.startswith("4VM24CS"):
            suffix = student_id[-3:]
            if suffix.isdigit():
                numbers.append(int(suffix))

    next_number = max(numbers, default=100) + 1

    while True:
        new_id = f"4VM24CS{next_number:03d}"
        if new_id not in existing_ids:
            return new_id
        next_number += 1


# =========================================================
# 6. GLOBAL METRICS
# =========================================================

total_students = len(students)

low_count = (students["Risk"] == "Low").sum()
moderate_count = (students["Risk"] == "Moderate").sum()
high_count = (students["Risk"] == "High").sum()
critical_count = (students["Risk"] == "Critical").sum()

at_risk_count = high_count + critical_count

risk_percentage = at_risk_count / total_students * 100 if total_students else 0

average_attendance = students["Attendance"].mean()
average_marks = students["Marks"].mean()
average_assignments = students["Assignments"].mean()

# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.divider()
st.sidebar.title("🎓 Student Intelligence")
st.sidebar.caption("Academic monitoring and risk analysis platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Management",
        "Risk Prediction Center",
        "Explainable AI",
        "Risk Forecasting",
        "What-If Simulator",
        "Student Risk Trend Analyzer",
        "Subject-Level Intelligence",
        "Student Clustering",
        "Anomaly Detection",
        "AI Recovery Pathway",
        "Faculty / Mentor Dashboard",
        "Intervention Effectiveness Analyzer",
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
# 8. SMART DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("🎓 Student Academic Intelligence")
    st.caption("Academic monitoring • Risk intelligence • Student success analytics")
    st.divider()

    st.subheader("📌 Academic Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Students at Risk", at_risk_count, f"{risk_percentage:.1f}% of students")
    col3.metric("Average Attendance", f"{average_attendance:.1f}%")
    col4.metric("Average Marks", f"{average_marks:.1f}%")

    st.divider()
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
    st.subheader("🚨 Risk Intelligence Overview")

    risk_data = pd.DataFrame({
        "Risk Level": risk_order,
        "Students": [low_count, moderate_count, high_count, critical_count]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            risk_data,
            names="Risk Level",
            values="Students",
            hole=0.55,
            color="Risk Level",
            color_discrete_map=risk_colors
        )
        fig.update_traces(textinfo="percent+label")
        apply_chart_theme(fig)
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
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🟢 Low", low_count)
    col2.metric("🟡 Moderate", moderate_count)
    col3.metric("🟠 High", high_count)
    col4.metric("🔴 Critical", critical_count)

    if critical_count:
        st.error(f"{critical_count} students require urgent attention.")
    elif high_count:
        st.warning(f"{high_count} high-risk students require monitoring.")
    else:
        st.success("No high or critical risk students detected.")

    st.divider()
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
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        performance_data = pd.DataFrame({
            "Metric": ["Attendance", "Marks", "Assignments"],
            "Average": [average_attendance, average_marks, average_assignments]
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
        apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🚨 Priority Student Monitoring")

    priority_students = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values("Risk Score", ascending=False)

    if not priority_students.empty:
        st.dataframe(
            priority_students[[
                "Student ID", "Student", "Attendance",
                "Marks", "Assignments", "Risk", "Risk Score"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No high-priority students at this time.")

    st.divider()
    st.subheader("🏆 Top Academic Performers")

    top_students = students.sort_values("Marks", ascending=False).head(5)

    st.dataframe(
        top_students[[
            "Student ID", "Student", "Attendance",
            "Marks", "Assignments", "Risk"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("💡 Academic Insights")

    col1, col2 = st.columns(2)

    with col1:
        if average_attendance < 75:
            st.warning(f"Average attendance is {average_attendance:.1f}%. Attendance support may be needed.")
        else:
            st.success(f"Average attendance is {average_attendance:.1f}%.")

        if average_marks < 50:
            st.warning(f"Average marks are {average_marks:.1f}%. Academic support may be beneficial.")
        else:
            st.success(f"Average marks are {average_marks:.1f}%.")

    with col2:
        if at_risk_count:
            st.warning(f"{at_risk_count} students are classified High or Critical.")
        else:
            st.success("No High or Critical risk students.")

        if average_assignments < 60:
            st.warning("Assignment performance needs attention.")
        else:
            st.info("Assignment performance is satisfactory.")

    st.divider()
    st.subheader("🎓 Semester Performance Summary")

    semester_summary = students.groupby("Semester").agg(
        Total_Students=("Student ID", "count"),
        Average_Attendance=("Attendance", "mean"),
        Average_Marks=("Marks", "mean"),
        Average_Assignments=("Assignments", "mean"),
        At_Risk_Students=("Risk", lambda x: x.isin(["High", "Critical"]).sum())
    ).reset_index()

    st.dataframe(semester_summary.round(2), use_container_width=True, hide_index=True)


# =========================================================
# 9. STUDENT MANAGEMENT SYSTEM
# =========================================================

elif page == "Student Management":

    st.title("👨‍🎓 Student Management System")
    st.caption("Register, update, search, manage, and explore student academic records.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Student Records",
        "➕ Add Student",
        "✏️ Edit Student",
        "🗑️ Delete Student"
    ])

    # TAB 1: RECORDS, SEARCH & PROFILE
    with tab1:

        st.subheader("Student Records")

        col1, col2, col3 = st.columns(3)

        with col1:
            search = st.text_input("Search Name or USN", key="management_search")

        with col2:
            risk_filter = st.selectbox(
                "Risk Filter",
                ["All", "Low", "Moderate", "High", "Critical"],
                key="management_risk"
            )

        with col3:
            semester_filter = st.selectbox(
                "Semester",
                ["All"] + sorted(students["Semester"].dropna().unique().tolist()),
                key="management_semester"
            )

        filtered = students.copy()

        if search:
            filtered = filtered[
                filtered["Student"].str.contains(search, case=False, na=False)
                | filtered["Student ID"].str.contains(search, case=False, na=False)
            ]

        if risk_filter != "All":
            filtered = filtered[filtered["Risk"] == risk_filter]

        if semester_filter != "All":
            filtered = filtered[filtered["Semester"] == semester_filter]

        st.metric("Students Found", len(filtered))

        st.dataframe(
            filtered[[
                "Student ID", "Student", "Department", "Semester",
                "Attendance", "Marks", "Assignments", "Risk", "Risk Score"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("📄 Download Student Records")

        csv_data = filtered.drop(
            columns=["Risk Score"], errors="ignore"
        ).to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            file_name="student_records.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("🔎 Individual Student Profile")

        if not filtered.empty:

            chosen = st.selectbox(
                "Select Student",
                filtered["Student"].tolist(),
                key="profile_selection"
            )

            profile = filtered[filtered["Student"] == chosen].iloc[0]

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

            profile_chart = pd.DataFrame({
                "Metric": ["Attendance", "Marks", "Assignments"],
                "Value": [
                    profile["Attendance"],
                    profile["Marks"],
                    profile["Assignments"]
                ]
            })

            fig = px.bar(
                profile_chart,
                x="Metric",
                y="Value",
                color="Metric",
                text_auto=".1f"
            )
            fig.update_layout(yaxis_range=[0, 100])
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No students match the selected filters.")

    # TAB 2: ADD STUDENT
    with tab2:

        st.subheader("➕ Register New Student")

        new_id = generate_student_id(students)

        st.info(f"Automatically generated student USN: **{new_id}**")

        with st.form("add_student_form"):

            new_name = st.text_input("Student Name")
            new_department = st.selectbox("Department", ["CSE"])
            new_semester = st.selectbox("Semester", list(range(1, 9)), index=4)

            col1, col2, col3 = st.columns(3)

            with col1:
                new_attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)

            with col2:
                new_marks = st.number_input("Marks (%)", 0.0, 100.0, 60.0)

            with col3:
                new_assignments = st.number_input("Assignments (%)", 0.0, 100.0, 60.0)

            submitted = st.form_submit_button(
                "➕ Register Student",
                use_container_width=True
            )

            if submitted:

                if not new_name.strip():
                    st.error("Please enter the student's name.")

                elif new_id in students["Student ID"].values:
                    st.error("This USN already exists.")

                else:

                    new_record = pd.DataFrame([{
                        "Student ID": new_id,
                        "Student": new_name.strip(),
                        "Department": new_department,
                        "Semester": new_semester,
                        "Attendance": new_attendance,
                        "Marks": new_marks,
                        "Assignments": new_assignments
                    }])

                    updated = pd.concat([students, new_record], ignore_index=True)
                    save_student_data(updated)

                    st.success(f"Student {new_name} registered successfully!")
                    st.rerun()

    # TAB 3: EDIT STUDENT
    with tab3:

        st.subheader("✏️ Update Student Records")

        if not students.empty:

            edit_student_name = st.selectbox(
                "Select student to edit",
                students["Student"].tolist(),
                key="edit_student_select"
            )

            edit_record = students[
                students["Student"] == edit_student_name
            ].iloc[0]

            st.info(f"USN: {edit_record['Student ID']}")

            with st.form("edit_student_form"):

                edit_name = st.text_input(
                    "Student Name",
                    value=edit_record["Student"]
                )

                edit_department = st.selectbox(
                    "Department",
                    ["CSE"],
                    index=0
                )

                edit_semester = st.selectbox(
                    "Semester",
                    list(range(1, 9)),
                    index=int(edit_record["Semester"]) - 1
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    edit_attendance = st.number_input(
                        "Attendance (%)",
                        0.0,
                        100.0,
                        float(edit_record["Attendance"])
                    )

                with col2:
                    edit_marks = st.number_input(
                        "Marks (%)",
                        0.0,
                        100.0,
                        float(edit_record["Marks"])
                    )

                with col3:
                    edit_assignments = st.number_input(
                        "Assignments (%)",
                        0.0,
                        100.0,
                        float(edit_record["Assignments"])
                    )

                save_changes = st.form_submit_button(
                    "💾 Save Changes",
                    use_container_width=True
                )

                if save_changes:

                    if not edit_name.strip():
                        st.error("Student name cannot be empty.")

                    else:

                        updated = students.copy()
                        idx = updated.index[
                            updated["Student"] == edit_student_name
                        ][0]

                        updated.loc[idx, "Student"] = edit_name.strip()
                        updated.loc[idx, "Department"] = edit_department
                        updated.loc[idx, "Semester"] = edit_semester
                        updated.loc[idx, "Attendance"] = edit_attendance
                        updated.loc[idx, "Marks"] = edit_marks
                        updated.loc[idx, "Assignments"] = edit_assignments

                        save_student_data(updated)

                        st.success("Student records updated successfully!")
                        st.rerun()

    # TAB 4: DELETE STUDENT
    with tab4:

        st.subheader("🗑️ Remove Student Record")

        st.warning("Deleting a student will remove their record from the dataset.")

        if not students.empty:

            delete_student_name = st.selectbox(
                "Select student to delete",
                students["Student"].tolist(),
                key="delete_student_select"
            )

            delete_record = students[
                students["Student"] == delete_student_name
            ].iloc[0]

            st.write("**USN:**", delete_record["Student ID"])
            st.write("**Name:**", delete_record["Student"])
            st.write("**Risk Level:**", delete_record["Risk"])

            confirm_delete = st.checkbox(
                "I confirm that I want to delete this student.",
                key="delete_confirmation"
            )

            if st.button("🗑️ Delete Student", disabled=not confirm_delete):

                updated = students[
                    students["Student"] != delete_student_name
                ].copy()

                save_student_data(updated)

                st.success(f"{delete_student_name} was deleted.")
                st.rerun()


# =========================================================
# 10. RISK PREDICTION CENTER
# =========================================================

elif page == "Risk Prediction Center":

    from risk_prediction import render_risk_prediction_page

    render_risk_prediction_page(
        students,
        apply_chart_theme,
        risk_colors
    )


# =========================================================
# 11. EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    from explainable_ai import render_explainable_ai

    render_explainable_ai(
        students,
        apply_chart_theme,
        risk_colors
    )


# =========================================================
# 12. RISK FORECASTING
# =========================================================

elif page == "Risk Forecasting":

    from risk_forecasting import render_risk_forecasting

    render_risk_forecasting(
        students,
        apply_chart_theme,
        risk_colors
    )


# =========================================================
# 13. ACADEMIC WHAT-IF SIMULATOR
# =========================================================

elif page == "What-If Simulator":

    from what_if_simulator import render_what_if_simulator

    render_what_if_simulator(
        students,
        apply_chart_theme,
        risk_colors
    )


# =========================================================
# 14. STUDENT RISK TREND ANALYZER (FEATURE 3.7)
# =========================================================

elif page == "Student Risk Trend Analyzer":

    from student_risk_trend import render_student_risk_trend

    render_student_risk_trend(
        students,
        apply_chart_theme,
        risk_colors
    )


# =========================================================
# 15. SUBJECT-LEVEL INTELLIGENCE (FEATURE 3.8)
# =========================================================

elif page == "Subject-Level Intelligence":

    from modules.subject_intelligence import render_subject_intelligence

    render_subject_intelligence(students)


# =========================================================
# 16. STUDENT CLUSTERING (FEATURE 3.9)
# =========================================================

elif page == "Student Clustering":

    from modules.student_clustering import render_student_clustering

    render_student_clustering(students)


# =========================================================
# 16A. AI ANOMALY DETECTION (FEATURE 3.10)
# =========================================================

elif page == "Anomaly Detection":

    render_anomaly_detection(students)


# =========================================================
# 17. RISK ANALYTICS
# =========================================================

# =========================================================
# 16B. FACULTY / MENTOR DASHBOARD (FEATURE 3.12)
# =========================================================

elif page == "Faculty / Mentor Dashboard":

    render_faculty_dashboard(students)


# =========================================================
# 16C. INTERVENTION EFFECTIVENESS ANALYZER (FEATURE 3.13)
# =========================================================

elif page == "Intervention Effectiveness Analyzer":

    render_intervention_effectiveness(students)


# =========================================================
# 17. RISK ANALYTICS
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

    st.dataframe(risk_analysis.round(2), use_container_width=True)


# =========================================================
# 17. EARLY WARNING CENTER
# =========================================================

elif page == "Early Warning Center":

    st.title("🚨 Early Warning Center")
    st.caption("Monitor attendance, marks and assignment concerns.")

    attendance_alerts = students[students["Attendance"] < 60]
    marks_alerts = students[students["Marks"] < 50]
    assignment_alerts = students[students["Assignments"] < 50]

    col1, col2, col3 = st.columns(3)

    col1.metric("Attendance Alerts", len(attendance_alerts))
    col2.metric("Marks Alerts", len(marks_alerts))
    col3.metric("Assignment Alerts", len(assignment_alerts))

    st.divider()
    st.subheader("⚠️ Attendance Below 60%")

    st.dataframe(
        attendance_alerts[[
            "Student ID", "Student", "Attendance", "Marks", "Risk"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📉 Marks Below 50%")

    st.dataframe(
        marks_alerts[[
            "Student ID", "Student", "Marks", "Attendance", "Risk"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📝 Assignment Scores Below 50%")

    st.dataframe(
        assignment_alerts[[
            "Student ID", "Student", "Assignments", "Risk"
        ]],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 18. RECOVERY PLANS
# =========================================================

elif page == "Recovery Plans":

    st.title("🛠️ Student Recovery Plans")
    st.caption("Personalized academic support and mentor follow-up.")

    recovery_students = students[
        students["Risk"].isin(["High", "Critical", "Moderate"])
    ].sort_values("Risk Score", ascending=False)

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

        st.info("Recommended actions should be reviewed by faculty or mentors.")


# =========================================================
# 18A. AI RECOVERY PATHWAY (FEATURE 3.11)
# =========================================================

elif page == "AI Recovery Pathway":

    render_recovery_pathway(students)


# =========================================================
# 19. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("🎓 About Student Risk Prediction")

    st.markdown("""
    ## AI-Powered Student Risk Prediction & Academic Intelligence Platform

    An academic monitoring platform designed to identify students
    who may require additional academic support.

    ### Current Features

    - Smart academic dashboard
    - Student management and registration
    - Student profiles
    - Student record editing and deletion
    - Risk classification and scoring
    - Machine learning risk prediction
    - Explainable AI risk analysis
    - Risk forecasting simulation
    - Academic What-If Simulator
    - Student Risk Trend Analyzer
    - Subject-Level Intelligence
    - Student Clustering (K-Means)
    - AI Anomaly Detection (Isolation Forest)
    - AI Recovery Pathway
    - Faculty / Mentor Dashboard
    - Intervention Effectiveness Analyzer
    - Risk analytics
    - Attendance, marks and assignment monitoring
    - Early warning center
    - Recovery plans
    - CSV data export
    - Light and dark themes

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

    ### Advanced AI Features
    - Faculty / Mentor Dashboard

    - Machine learning risk prediction
    - Explainable AI
    - Risk forecasting
    - Academic What-If Simulator
    - Student Risk Trend Analyzer
    - Subject-Level Intelligence
    - Student clustering
    - AI anomaly detection
    - AI recovery pathway
    - Intervention effectiveness analysis
    - Database integration
    - Role-based login
    - Automated reports
    - Personalized recovery pathways
    """)

    st.divider()

    st.info(
        "The current risk classification is rule-based. "
        "The machine learning module is trained using synthetic academic data "
        "and is intended for demonstration and experimentation."
    )


# =========================================================
# 20. FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption("Student Risk Prediction")
st.sidebar.caption("Academic Intelligence Platform")
