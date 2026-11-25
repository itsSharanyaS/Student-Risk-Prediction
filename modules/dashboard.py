
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Student Risk Intelligence",
    page_icon="🎓",
    layout="wide"
)

# -------------------- SAMPLE DATA --------------------
np.random.seed(42)

n = 250

df = pd.DataFrame({
    "Student ID": [f"4VM24CS{np.random.randint(100, 999)}" for _ in range(n)],
    "Semester": np.random.choice([3, 4, 5, 6, 7, 8], n),
    "Section": np.random.choice(["A", "B"], n),
    "Attendance": np.random.randint(45, 101, n),
    "Marks": np.random.randint(35, 101, n),
    "Assignment": np.random.randint(40, 101, n),
})

# -------------------- RISK SCORE --------------------
df["Risk Score"] = (
    (100 - df["Attendance"]) * 0.45
    + (100 - df["Marks"]) * 0.40
    + (100 - df["Assignment"]) * 0.15
)

df["Risk Level"] = pd.cut(
    df["Risk Score"],
    bins=[-1, 30, 55, 100],
    labels=["Low", "Medium", "High"]
)

# -------------------- SIDEBAR --------------------
st.sidebar.title("🎓 EduGuard AI")
st.sidebar.caption("Student Risk Intelligence System")
st.sidebar.divider()

semester = st.sidebar.selectbox(
    "Semester",
    ["All"] + sorted(df["Semester"].unique().tolist())
)

section = st.sidebar.selectbox(
    "Section",
    ["All", "A", "B"]
)

risk_filter = st.sidebar.selectbox(
    "Risk Category",
    ["All", "High", "Medium", "Low"]
)

search = st.sidebar.text_input("Search Student ID")

# -------------------- FILTER DATA --------------------
filtered_df = df.copy()

if semester != "All":
    filtered_df = filtered_df[filtered_df["Semester"] == semester]

if section != "All":
    filtered_df = filtered_df[filtered_df["Section"] == section]

if risk_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Risk Level"] == risk_filter
    ]

if search:
    filtered_df = filtered_df[
        filtered_df["Student ID"].str.contains(search, case=False)
    ]

# -------------------- HEADER --------------------
st.title("Academic Overview")
st.caption(
    "Monitor student performance, identify academic risks "
    "and support students through data-driven insights."
)

st.divider()

# -------------------- KPI CARDS --------------------
total_students = len(filtered_df)

high_risk = len(
    filtered_df[filtered_df["Risk Level"] == "High"]
)

avg_attendance = (
    filtered_df["Attendance"].mean()
    if total_students else 0
)

avg_marks = (
    filtered_df["Marks"].mean()
    if total_students else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", total_students)
c2.metric("High Risk Students", high_risk)
c3.metric("Average Attendance", f"{avg_attendance:.1f}%")
c4.metric("Average Marks", f"{avg_marks:.1f}%")

st.divider()

# -------------------- RISK DISTRIBUTION --------------------
left, right = st.columns(2)

with left:
    st.subheader("Risk Distribution")

    risk_counts = filtered_df["Risk Level"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    ).reset_index()

    risk_counts.columns = ["Risk Level", "Students"]

    fig_risk = px.pie(
        risk_counts,
        names="Risk Level",
        values="Students",
        hole=0.55,
        color="Risk Level",
        color_discrete_map={
            "Low": "#22C55E",
            "Medium": "#F59E0B",
            "High": "#EF4444"
        }
    )

    fig_risk.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        legend_title="Risk Category"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

# -------------------- PERFORMANCE DISTRIBUTION --------------------
with right:
    st.subheader("Academic Performance")

    fig_marks = px.histogram(
        filtered_df,
        x="Marks",
        nbins=15,
        color_discrete_sequence=["#3B82F6"],
        labels={"Marks": "Academic Marks"}
    )

    fig_marks.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="Marks (%)",
        yaxis_title="Number of Students"
    )

    st.plotly_chart(fig_marks, use_container_width=True)

st.divider()

# -------------------- ATTENDANCE VS MARKS --------------------
left, right = st.columns(2)

with left:
    st.subheader("Attendance vs Academic Marks")

    fig_scatter = px.scatter(
        filtered_df,
        x="Attendance",
        y="Marks",
        color="Risk Level",
        hover_data=["Student ID", "Semester"],
        color_discrete_map={
            "Low": "#22C55E",
            "Medium": "#F59E0B",
            "High": "#EF4444"
        }
    )

    fig_scatter.update_layout(
        height=400,
        xaxis_title="Attendance (%)",
        yaxis_title="Marks (%)"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------- SEMESTER ANALYSIS --------------------
with right:
    st.subheader("Semester Performance")

    sem_data = df.groupby("Semester", as_index=False).agg(
        Average_Marks=("Marks", "mean"),
        Average_Attendance=("Attendance", "mean")
    )

    fig_sem = px.bar(
        sem_data,
        x="Semester",
        y="Average_Marks",
        color_discrete_sequence=["#6366F1"],
        labels={"Average_Marks": "Average Marks"}
    )

    fig_sem.update_layout(
        height=400,
        xaxis_title="Semester",
        yaxis_title="Average Marks (%)"
    )

    st.plotly_chart(fig_sem, use_container_width=True)

st.divider()

# -------------------- PRIORITY STUDENTS --------------------
st.subheader("🚨 Students Requiring Attention")

priority_df = filtered_df[
    filtered_df["Risk Level"] == "High"
].sort_values("Risk Score", ascending=False)

priority_df = priority_df[
    ["Student ID", "Semester", "Attendance", "Marks",
     "Assignment", "Risk Score", "Risk Level"]
]

st.dataframe(
    priority_df,
    use_container_width=True,
    hide_index=True
)

# -------------------- SMART INSIGHTS --------------------
st.divider()
st.subheader("🧠 Academic Insights")

if total_students > 0:

    low_attendance = len(
        filtered_df[filtered_df["Attendance"] < 75]
    )

    low_marks = len(
        filtered_df[filtered_df["Marks"] < 50]
    )

    st.info(
        f"📌 {high_risk} students are currently classified as high risk."
    )

    st.warning(
        f"📉 {low_attendance} students have attendance below 75%."
    )

    st.info(
        f"📚 {low_marks} students have marks below 50%."
    )

    if avg_attendance < 75:
        st.warning(
            "Overall attendance is below the 75% threshold. "
            "Consider reviewing attendance interventions."
        )

    if avg_marks < 50:
        st.warning(
            "Average academic marks are low. "
            "Additional academic support may be required."
        )

else:
    st.info("No students found for the selected filters.")

# -------------------- EXPORT --------------------
st.divider()

st.subheader("📥 Export Dashboard Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Student Data CSV",
    data=csv,
    file_name="student_risk_dashboard.csv",
    mime="text/csv"
)

st.caption(
    "Student Risk Intelligence System | Academic Analytics Dashboard"
)
