
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Student Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. THEME
# ==========================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.session_state.dark_mode

if dark_mode:
    BG = "#0B1120"
    CARD = "#151F32"
    TEXT = "#E5E7EB"
    MUTED = "#94A3B8"
    BORDER = "#263449"
else:
    BG = "#F4F7FB"
    CARD = "#FFFFFF"
    TEXT = "#172033"
    MUTED = "#64748B"
    BORDER = "#E5EAF1"

st.markdown(f"""
<style>
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}
[data-testid="stSidebar"] {{
    background-color: {"#111827" if dark_mode else "#FFFFFF"};
}}
[data-testid="stSidebar"] * {{
    color: {"#E5E7EB" if dark_mode else "#263244"};
}}
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}}
div[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    padding: 18px;
    border-radius: 14px;
}}
div[data-testid="stMetricLabel"] {{
    color: {MUTED};
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT};
}}
div[data-testid="stMetricDelta"] {{
    font-size: 0.85rem;
}}
h1, h2, h3 {{
    color: {TEXT} !important;
}}
.custom-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 20px;
}}
.small-label {{
    color: {MUTED};
    font-size: 13px;
}}
.main-title {{
    font-size: 30px;
    font-weight: 750;
    letter-spacing: -0.5px;
}}
.subtitle {{
    color: {MUTED};
    font-size: 14px;
}}
div[data-testid="stPlotlyChart"] {{
    background: {CARD};
    border-radius: 12px;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. SAMPLE DATA
# ==========================================================
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n = 250

    attendance = np.clip(np.random.normal(82, 12, n), 35, 100)
    marks = np.clip(
        attendance * 0.48 + np.random.normal(30, 12, n),
        10, 100
    )
    assignments = np.random.randint(40, 101, n)

    risk_score = (
        (100 - attendance) * 0.45 +
        (100 - marks) * 0.40 +
        (100 - assignments) * 0.15
    )

    risk_score = np.clip(risk_score, 0, 100)

    risk_level = np.where(
        risk_score >= 65, "High Risk",
        np.where(risk_score >= 40, "Medium Risk", "Low Risk")
    )

    data = pd.DataFrame({
        "Student_ID": [f"4VM24CS{np.random.randint(100, 999)}" for _ in range(n)],
        "Department": "CSE",
        "Semester": np.random.choice([3, 5, 7], n),
        "Section": np.random.choice(["A", "B"], n),
        "Attendance": np.round(attendance, 1),
        "Average_Marks": np.round(marks, 1),
        "Assignment_Score": np.round(assignments, 1),
        "Risk_Score": np.round(risk_score, 1),
        "Risk_Level": risk_level
    })

    # Make IDs unique
    data["Student_ID"] = [
        f"4VM24CS{i:03d}" for i in range(1, n + 1)
    ]

    return data


# ==========================================================
# 4. DATA LOADING
# ==========================================================
st.sidebar.markdown("## 🎓 Student Risk Intelligence")
st.sidebar.caption("Academic Support & Analytics Platform")

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Preferences")
st.session_state.dark_mode = st.sidebar.toggle(
    "Dark Mode",
    value=st.session_state.dark_mode
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Student CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully")
        is_sample = False
    except Exception as e:
        st.sidebar.error(f"Unable to read file: {e}")
        df = create_sample_data()
        is_sample = True
else:
    df = create_sample_data()
    is_sample = True

# ==========================================================
# 5. COLUMN NORMALIZATION
# ==========================================================
def normalize_data(data):
    data = data.copy()

    data.columns = [
        str(c).strip().replace(" ", "_") for c in data.columns
    ]

    # Common column-name mapping
    aliases = {
        "Student_ID": ["Student_ID", "student_id", "USN", "usn", "ID"],
        "Attendance": ["Attendance", "attendance", "Attendance_Percentage"],
        "Average_Marks": ["Average_Marks", "average_marks", "Marks", "Score", "Final_Score"],
        "Assignment_Score": ["Assignment_Score", "Assignments", "assignment_score"],
        "Department": ["Department", "department", "Dept"],
        "Semester": ["Semester", "semester", "Sem"],
        "Section": ["Section", "section"]
    }

    for standard, possible_names in aliases.items():
        for name in possible_names:
            if name in data.columns:
                if standard != name:
                    data.rename(columns={name: standard}, inplace=True)
                break

    # Create missing columns where possible
    if "Student_ID" not in data.columns:
        data["Student_ID"] = [
            f"STU{i:03d}" for i in range(1, len(data) + 1)
        ]

    if "Attendance" not in data.columns:
        data["Attendance"] = 75

    if "Average_Marks" not in data.columns:
        data["Average_Marks"] = 50

    if "Assignment_Score" not in data.columns:
        data["Assignment_Score"] = 50

    if "Department" not in data.columns:
        data["Department"] = "CSE"

    if "Semester" not in data.columns:
        data["Semester"] = 5

    if "Section" not in data.columns:
        data["Section"] = "A"

    for col in ["Attendance", "Average_Marks", "Assignment_Score"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())
        data[col] = data[col].clip(0, 100)

    # Rule-based risk score
    data["Risk_Score"] = (
        (100 - data["Attendance"]) * 0.45 +
        (100 - data["Average_Marks"]) * 0.40 +
        (100 - data["Assignment_Score"]) * 0.15
    ).clip(0, 100).round(1)

    data["Risk_Level"] = np.select(
        [
            data["Risk_Score"] >= 65,
            data["Risk_Score"] >= 40
        ],
        ["High Risk", "Medium Risk"],
        default="Low Risk"
    )

    return data


df = normalize_data(df)

# ==========================================================
# 6. HEADER
# ==========================================================
st.markdown(
    """
    <div class="main-title">Academic Overview</div>
    <div class="subtitle">
        Monitor academic performance, identify risk patterns and support student success.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

if is_sample:
    st.info(
        "You are currently viewing sample data. Upload your student dataset "
        "from the sidebar to analyse your own records."
    )

# ==========================================================
# 7. FILTERS
# ==========================================================
filter1, filter2, filter3 = st.columns(3)

with filter1:
    semesters = sorted(df["Semester"].dropna().unique().tolist())
    selected_semesters = st.multiselect(
        "Semester",
        semesters,
        default=semesters
    )

with filter2:
    sections = sorted(df["Section"].dropna().astype(str).unique().tolist())
    selected_sections = st.multiselect(
        "Section",
        sections,
        default=sections
    )

with filter3:
    risk_options = ["Low Risk", "Medium Risk", "High Risk"]
    selected_risks = st.multiselect(
        "Risk Category",
        risk_options,
        default=risk_options
    )

filtered_df = df[
    df["Semester"].isin(selected_semesters) &
    df["Section"].astype(str).isin(selected_sections) &
    df["Risk_Level"].isin(selected_risks)
].copy()

# ==========================================================
# 8. KPI CALCULATIONS
# ==========================================================
total_students = len(filtered_df)

high_risk = int(
    (filtered_df["Risk_Level"] == "High Risk").sum()
)

medium_risk = int(
    (filtered_df["Risk_Level"] == "Medium Risk").sum()
)

low_risk = int(
    (filtered_df["Risk_Level"] == "Low Risk").sum()
)

avg_attendance = (
    filtered_df["Attendance"].mean()
    if total_students else 0
)

avg_marks = (
    filtered_df["Average_Marks"].mean()
    if total_students else 0
)

improving_students = int(
    (filtered_df["Average_Marks"] >= 75).sum()
)

# ==========================================================
# 9. KPI CARDS
# ==========================================================
st.markdown("### Performance Summary")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "👥 Total Students",
    f"{total_students:,}",
    help="Total students in the selected filters"
)

k2.metric(
    "🔴 High Risk",
    f"{high_risk:,}",
    help="Students identified as high risk"
)

k3.metric(
    "📅 Avg Attendance",
    f"{avg_attendance:.1f}%",
    help="Average attendance percentage"
)

k4.metric(
    "🎓 Avg Marks",
    f"{avg_marks:.1f}/100",
    help="Average academic score"
)

k5.metric(
    "⭐ High Performers",
    f"{improving_students:,}",
    help="Students with average marks of 75 or more"
)

st.write("")

# ==========================================================
# 10. RISK DISTRIBUTION
# ==========================================================
st.markdown("### Academic Risk & Performance")

chart1, chart2 = st.columns(2)

with chart1:
    risk_counts = pd.DataFrame({
        "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
        "Students": [low_risk, medium_risk, high_risk]
    })

    fig_risk = px.pie(
        risk_counts,
        names="Risk Level",
        values="Students",
        hole=0.68,
        color="Risk Level",
        color_discrete_map={
            "Low Risk": "#16A34A",
            "Medium Risk": "#F59E0B",
            "High Risk": "#DC2626"
        }
    )

    fig_risk.update_traces(
        textposition="outside",
        textinfo="percent",
        marker=dict(line=dict(color=BG, width=3))
    )

    fig_risk.update_layout(
        title="Risk Distribution",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        height=350,
        margin=dict(t=60, b=20, l=20, r=20),
        legend_title=""
    )

    st.plotly_chart(fig_risk, use_container_width=True)

with chart2:
    bins = [0, 40, 50, 60, 70, 80, 90, 100]
    labels = [
        "0–40", "41–50", "51–60", "61–70",
        "71–80", "81–90", "91–100"
    ]

    performance = pd.cut(
        filtered_df["Average_Marks"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    performance_counts = (
        performance.value_counts(sort=False)
        .reindex(labels, fill_value=0)
        .reset_index()
    )

    performance_counts.columns = ["Score Range", "Students"]

    fig_marks = px.bar(
        performance_counts,
        x="Score Range",
        y="Students",
        color="Students",
        color_continuous_scale="Blues"
    )

    fig_marks.update_layout(
        title="Academic Score Distribution",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        height=350,
        coloraxis_showscale=False,
        margin=dict(t=60, b=30, l=20, r=20)
    )

    fig_marks.update_xaxes(title="")
    fig_marks.update_yaxes(title="Number of Students")

    st.plotly_chart(fig_marks, use_container_width=True)

# ==========================================================
# 11. ATTENDANCE VS MARKS
# ==========================================================
st.markdown("### Academic Performance Insights")

chart3, chart4 = st.columns(2)

with chart3:
    fig_scatter = px.scatter(
        filtered_df,
        x="Attendance",
        y="Average_Marks",
        color="Risk_Level",
        color_discrete_map={
            "Low Risk": "#16A34A",
            "Medium Risk": "#F59E0B",
            "High Risk": "#DC2626"
        },
        hover_data=["Student_ID", "Risk_Score"],
        opacity=0.75
    )

    fig_scatter.update_layout(
        title="Attendance vs Academic Marks",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        height=380,
        margin=dict(t=60, b=30, l=20, r=20),
        legend_title="Risk"
    )

    fig_scatter.update_xaxes(title="Attendance (%)")
    fig_scatter.update_yaxes(title="Average Marks")

    st.plotly_chart(fig_scatter, use_container_width=True)

with chart4:
    risk_by_semester = (
        filtered_df.groupby("Semester", observed=False)["Risk_Score"]
        .mean()
        .reset_index()
    )

    fig_sem = px.bar(
        risk_by_semester,
        x="Semester",
        y="Risk_Score",
        color="Risk_Score",
        color_continuous_scale="RdYlGn_r"
    )

    fig_sem.update_layout(
        title="Average Risk Score by Semester",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font_color=TEXT,
        height=380,
        coloraxis_showscale=False,
        margin=dict(t=60, b=30, l=20, r=20)
    )

    fig_sem.update_yaxes(title="Average Risk Score", range=[0, 100])
    fig_sem.update_xaxes(title="Semester")

    st.plotly_chart(fig_sem, use_container_width=True)

# ==========================================================
# 12. STUDENTS REQUIRING ATTENTION
# ==========================================================
st.markdown("### 🚨 Students Requiring Attention")

priority_df = filtered_df.sort_values(
    "Risk_Score",
    ascending=False
).copy()

priority_df = priority_df.head(10)

display_df = priority_df[
    [
        "Student_ID",
        "Semester",
        "Section",
        "Attendance",
        "Average_Marks",
        "Assignment_Score",
        "Risk_Score",
        "Risk_Level"
    ]
].copy()

display_df.columns = [
    "Student ID",
    "Semester",
    "Section",
    "Attendance (%)",
    "Average Marks",
    "Assignment Score",
    "Risk Score",
    "Risk Level"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Attendance (%)": st.column_config.ProgressColumn(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            format="%.1f%%"
        ),
        "Average Marks": st.column_config.ProgressColumn(
            "Average Marks",
            min_value=0,
            max_value=100,
            format="%.1f"
        ),
        "Assignment Score": st.column_config.ProgressColumn(
            "Assignment Score",
            min_value=0,
            max_value=100,
            format="%.1f"
        ),
        "Risk Score": st.column_config.NumberColumn(
            "Risk Score",
            format="%.1f"
        )
    }
)

# ==========================================================
# 13. SEARCH STUDENTS
# ==========================================================
st.markdown("### 🔎 Student Search")

search_text = st.text_input(
    "Search by Student ID",
    placeholder="Enter student ID..."
)

if search_text:
    search_results = filtered_df[
        filtered_df["Student_ID"].astype(str).str.contains(
            search_text,
            case=False,
            na=False
        )
    ]

    if not search_results.empty:
        st.dataframe(
            search_results,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No matching students found.")

# ==========================================================
# 14. DOWNLOAD REPORT
# ==========================================================
st.markdown("### 📥 Export Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Student Analytics CSV",
    data=csv_data,
    file_name="student_risk_analytics.csv",
    mime="text/csv",
    use_container_width=False
)

# ==========================================================
# 15. FOOTER
# ==========================================================
st.divider()

st.markdown(
    f"""
    <div style="text-align:center; color:{MUTED}; font-size:12px;">
        Student Risk Intelligence System · Academic Analytics & Student Support<br>
        Dashboard generated using {total_students} student records
    </div>
    """,
    unsafe_allow_html=True
)
