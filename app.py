
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------- PREMIUM DARK THEME ----------
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
    background: linear-gradient(135deg, #172033, #1E293B);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #263449;
}
.metric-label {
    color: #9CA3AF;
    font-size: 14px;
}
.metric-value {
    color: #F8FAFC;
    font-size: 30px;
    font-weight: 700;
}
.metric-change {
    color: #34D399;
    font-size: 13px;
}
.section-title {
    color: #F8FAFC;
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- SAMPLE DATA ----------
students = pd.DataFrame({
    "Student": [
        "Aarav Sharma", "Ananya Rao", "Rohan Kumar",
        "Meera Nair", "Arjun Patel", "Sneha Iyer",
        "Kiran Reddy", "Diya Shah", "Rahul Das",
        "Pooja Singh", "Vikram Joshi", "Nisha Gowda"
    ],
    "Attendance": [92, 78, 56, 88, 64, 95, 49, 81, 58, 90, 72, 45],
    "Marks": [89, 76, 42, 91, 54, 96, 35, 73, 48, 85, 68, 32],
    "Risk": [
        "Low", "Moderate", "High", "Low",
        "Moderate", "Low", "Critical", "Moderate",
        "High", "Low", "Moderate", "Critical"
    ],
    "Department": [
        "CSE", "CSE", "ISE", "CSE", "ECE", "ISE",
        "CSE", "ECE", "ISE", "CSE", "ECE", "CSE"
    ]
})

# ---------- SIDEBAR ----------
st.sidebar.markdown("## 🎓 Student Risk AI")
st.sidebar.caption("Academic Intelligence Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Student Directory",
        "Risk Analytics",
        "Early Warnings",
        "Recovery Plans"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("🟢 System Status: Online")
st.sidebar.caption("Student Risk Prediction v1.0")

# ---------- DASHBOARD ----------
if page == "Dashboard":

    st.title("Academic Intelligence Dashboard")
    st.caption("Monitor academic performance, identify risks and support student success.")

    # Metrics
    total_students = len(students)
    high_risk = students["Risk"].isin(["High", "Critical"]).sum()
    avg_attendance = students["Attendance"].mean()
    avg_marks = students["Marks"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-change">Academic population</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Students At Risk</div>
            <div class="metric-value">{high_risk}</div>
            <div class="metric-change">High + Critical risk</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Attendance</div>
            <div class="metric-value">{avg_attendance:.1f}%</div>
            <div class="metric-change">Across all students</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value">{avg_marks:.1f}%</div>
            <div class="metric-change">Academic performance</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Charts
    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")

        risk_counts = students["Risk"].value_counts().reindex(
            ["Low", "Moderate", "High", "Critical"], fill_value=0
        ).reset_index()

        risk_counts.columns = ["Risk Level", "Students"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Students",
            hole=0.65,
            color="Risk Level",
            color_discrete_map={
                "Low": "#10B981",
                " "Moderate": "#F59E0B",
                "High": "#F97316",
                "Critical": "#EF4444"
            }
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB",
            legend_title="",
            margin=dict(t=20, b=20, l=20, r=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Attendance vs Academic Performance")

        fig2 = px.scatter(
            students,
            x="Attendance",
            y="Marks",
            color="Risk",
            hover_name="Student",
            color_discrete_map={
                "Low": "#10B981",
                "Moderate": "#F59E0B",
                "High": "#F97316",
                "Critical": "#EF4444"
            }
        )

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            font_color="#E5E7EB",
            xaxis_title="Attendance (%)",
            yaxis_title="Academic Score (%)"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Alerts
    st.subheader("🚨 Students Requiring Attention")

    alerts = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values("Attendance")

    st.dataframe(
        alerts[["Student", "Department", "Attendance", "Marks", "Risk"]],
        use_container_width=True,
        hide_index=True
    )

# ---------- STUDENT DIRECTORY ----------
elif page == "Student Directory":

    st.title("Student Directory")
    st.caption("Browse and filter student academic records.")

    search = st.text_input("Search student")

    filtered = students[
        students["Student"].str.contains(search, case=False, na=False)
    ]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

# ---------- RISK ANALYTICS ----------
elif page == "Risk Analytics":

    st.title("Risk Analytics")
    st.caption("Explore academic indicators and risk patterns.")

    fig = px.bar(
        students,
        x="Student",
        y="Marks",
        color="Risk",
        color_discrete_map={
            "Low": "#10B981",
            "Moderate": "#F59E0B",
            "High": "#F97316",
            "Critical": "#EF4444"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font_color="#E5E7EB"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Department Analysis")

    dept = students.groupby("Department", as_index=False).agg(
        Average_Marks=("Marks", "mean"),
        Average_Attendance=("Attendance", "mean"),
        Students=("Student", "count")
    )

    st.dataframe(dept.round(2), use_container_width=True, hide_index=True)

# ---------- EARLY WARNINGS ----------
elif page == "Early Warnings":

    st.title("Early Warning Center")
    st.caption("Identify students who may require academic support.")

    warnings = students[
        students["Risk"].isin(["High", "Critical"])
    ].sort_values("Marks")

    for _, row in warnings.iterrows():

        with st.expander(
            f"⚠️ {row['Student']} — {row['Risk']} Risk"
        ):
            st.write(f"**Department:** {row['Department']}")
            st.write(f"**Attendance:** {row['Attendance']}%")
            st.write(f"**Academic Score:** {row['Marks']}%")

            if row["Attendance"] < 60:
                st.warning("Low attendance detected.")

            if row["Marks"] < 50:
                st.warning("Academic performance needs attention.")

            st.info("Suggested action: Faculty review and personalized academic support.")

# ---------- RECOVERY PLANS ----------
elif page == "Recovery Plans":

    st.title("Student Recovery & Support")
    st.caption("Suggested actions to help students improve academic performance.")

    for _, row in students[
        students["Risk"].isin(["High", "Critical"])
    ].iterrows():

        st.markdown(f"### {row['Student']}")

        st.write(f"Risk Level: **{row['Risk']}**")

        tasks = [
            "Meet the faculty mentor",
            "Improve attendance",
            "Complete pending assignments",
            "Attend subject-specific remedial sessions",
            "Review weekly academic progress"
        ]

        for task in tasks:
            st.checkbox(task, key=f"{row['Student']}_{task}")

        st.divider()

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Student Risk Prediction | AI-Powered Academic Intelligence Platform")

