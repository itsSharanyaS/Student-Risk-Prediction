
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

SUBJECT_DATA_PATH = Path("data/subject_marks.csv")
PASS_MARK = 40


def load_subject_data():
    if SUBJECT_DATA_PATH.exists():
        try:
            return pd.read_csv(SUBJECT_DATA_PATH)
        except Exception:
            pass

    return pd.DataFrame(
        columns=["Student ID", "Student", "Subject", "Marks"]
    )


def save_subject_data(df):
    SUBJECT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUBJECT_DATA_PATH, index=False)


def render_subject_intelligence(students):

    st.title("📚 Subject-Level Intelligence")
    st.caption(
        "Subject-wise academic performance analysis, weak subject detection "
        "and student performance insights."
    )

    st.divider()

    # Load subject marks
    subject_data = load_subject_data()

    # -------------------------------------------------
    # EMPTY STATE
    # -------------------------------------------------

    if subject_data.empty:

        st.info(
            "No subject marks have been added yet. "
            "Start by entering student marks below."
        )

    # -------------------------------------------------
    # ADD SUBJECT MARKS
    # -------------------------------------------------

    st.subheader("➕ Add Subject Marks")

    student_options = students["Student ID"].astype(str).tolist()

    with st.form("add_subject_marks_form"):

        selected_student_id = st.selectbox(
            "Select Student ID",
            student_options
        )

        student_name = students.loc[
            students["Student ID"].astype(str) == selected_student_id,
            "Student"
        ].iloc[0]

        st.write(f"**Student Name:** {student_name}")

        subject_name = st.text_input(
            "Subject Name",
            placeholder="Example: Database Management Systems"
        )

        marks = st.number_input(
            "Marks (out of 100)",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )

        add_button = st.form_submit_button(
            "Add Marks",
            use_container_width=True
        )

        if add_button:

            if not subject_name.strip():
                st.warning("Please enter a subject name.")

            else:

                subject_data = load_subject_data()

                # Prevent duplicate subject entry for same student
                duplicate = (
                    (subject_data["Student ID"].astype(str)
                     == str(selected_student_id))
                    & (subject_data["Subject"].str.lower()
                       == subject_name.strip().lower())
                ).any() if not subject_data.empty else False

                if duplicate:
                    st.warning(
                        "Marks for this student and subject already exist. "
                        "Please edit the existing record instead."
                    )

                else:

                    new_record = pd.DataFrame([{
                        "Student ID": selected_student_id,
                        "Student": student_name,
                        "Subject": subject_name.strip(),
                        "Marks": marks
                    }])

                    subject_data = pd.concat(
                        [subject_data, new_record],
                        ignore_index=True
                    )

                    save_subject_data(subject_data)

                    st.success("Subject marks added successfully!")
                    st.rerun()

    st.divider()

    # -------------------------------------------------
    # DISPLAY DATA
    # -------------------------------------------------

    subject_data = load_subject_data()

    if subject_data.empty:
        st.stop()

    subject_data["Marks"] = pd.to_numeric(
        subject_data["Marks"], errors="coerce"
    )

    subject_data = subject_data.dropna(subset=["Marks"])

    subject_data["Result"] = subject_data["Marks"].apply(
        lambda x: "Pass" if x >= PASS_MARK else "Fail"
    )

    # -------------------------------------------------
    # OVERALL SUBJECT METRICS
    # -------------------------------------------------

    st.subheader("📊 Academic Overview")

    total_records = len(subject_data)
    average_marks = subject_data["Marks"].mean()
    pass_count = (subject_data["Marks"] >= PASS_MARK).sum()
    fail_count = (subject_data["Marks"] < PASS_MARK).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total_records)
    col2.metric("Average Marks", f"{average_marks:.1f}%")
    col3.metric("Pass Records", pass_count)
    col4.metric("Fail Records", fail_count)

    st.divider()

    # -------------------------------------------------
    # SUBJECT PERFORMANCE
    # -------------------------------------------------

    st.subheader("📖 Subject Performance")

    subject_summary = (
        subject_data.groupby("Subject")
        .agg(
            Average_Marks=("Marks", "mean"),
            Highest_Marks=("Marks", "max"),
            Lowest_Marks=("Marks", "min"),
            Students=("Student ID", "nunique"),
            Pass_Count=("Result", lambda x: (x == "Pass").sum()),
            Fail_Count=("Result", lambda x: (x == "Fail").sum())
        )
        .reset_index()
    )

    subject_summary["Average_Marks"] = subject_summary[
        "Average_Marks"
    ].round(2)

    subject_summary = subject_summary.sort_values(
        "Average_Marks", ascending=False
    )

    st.dataframe(
        subject_summary,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------
    # SUBJECT AVERAGE CHART
    # -------------------------------------------------

    st.subheader("📈 Subject Average Comparison")

    fig = px.bar(
        subject_summary,
        x="Subject",
        y="Average_Marks",
        color="Average_Marks",
        text_auto=".1f",
        color_continuous_scale="Blues",
        labels={
            "Average_Marks": "Average Marks",
            "Subject": "Subject"
        }
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # PASS / FAIL ANALYSIS
    # -------------------------------------------------

    st.subheader("🟢 Pass vs Fail Analysis")

    result_summary = (
        subject_data.groupby(["Subject", "Result"])
        .size()
        .reset_index(name="Students")
    )

    fig = px.bar(
        result_summary,
        x="Subject",
        y="Students",
        color="Result",
        barmode="group",
        text="Students",
        color_discrete_map={
            "Pass": "#16A34A",
            "Fail": "#DC2626"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------------------------------
    # WEAK SUBJECT DETECTION
    # -------------------------------------------------

    st.subheader("⚠️ Weak Subject Detection")

    weak_subjects = subject_summary[
        subject_summary["Average_Marks"] < 50
    ]

    if not weak_subjects.empty:

        st.warning(
            f"{len(weak_subjects)} subject(s) have an average below 50 marks."
        )

        st.dataframe(
            weak_subjects[
                ["Subject", "Average_Marks", "Fail_Count"]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No subjects currently have an average below 50 marks."
        )

    st.divider()

    # -------------------------------------------------
    # INDIVIDUAL STUDENT ANALYSIS
    # -------------------------------------------------

    st.subheader("👨‍🎓 Individual Student Analysis")

    student_id = st.selectbox(
        "Select a student to analyze",
        subject_data["Student ID"].unique(),
        key="subject_analysis_student"
    )

    student_subjects = subject_data[
        subject_data["Student ID"].astype(str) == str(student_id)
    ].copy()

    student_subjects = student_subjects.sort_values(
        "Marks", ascending=True
    )

    st.write(
        f"**Student:** {student_subjects['Student'].iloc[0]}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Marks",
        f"{student_subjects['Marks'].mean():.1f}%"
    )

    col2.metric(
        "Subjects Passed",
        int((student_subjects["Marks"] >= PASS_MARK).sum())
    )

    col3.metric(
        "Subjects Failed",
        int((student_subjects["Marks"] < PASS_MARK).sum())
    )

    fig = px.bar(
        student_subjects,
        x="Subject",
        y="Marks",
        color="Marks",
        text="Marks",
        color_continuous_scale="RdYlGn"
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # AUTOMATED INSIGHTS
    # -------------------------------------------------

    st.subheader("🧠 Academic Insights")

    student_average = student_subjects["Marks"].mean()

    failed_subjects = student_subjects[
        student_subjects["Marks"] < PASS_MARK
    ]

    best_subject = student_subjects.loc[
        student_subjects["Marks"].idxmax()
    ]

    worst_subject = student_subjects.loc[
        student_subjects["Marks"].idxmin()
    ]

    st.info(
        f"**Overall subject average:** {student_average:.1f}% "
        f"across {len(student_subjects)} subject(s)."
    )

    st.success(
        f"**Strongest subject:** {best_subject['Subject']} "
        f"({best_subject['Marks']:.0f} marks)."
    )

    if not failed_subjects.empty:

        for _, row in failed_subjects.iterrows():

            st.warning(
                f"Improvement needed in **{row['Subject']}** — "
                f"{row['Marks']:.0f} marks."
            )

    else:

        st.success(
            "The student has passed all recorded subjects."
        )

    if worst_subject["Marks"] >= PASS_MARK:

        st.info(
            f"**Lowest scoring subject:** {worst_subject['Subject']} "
            f"({worst_subject['Marks']:.0f} marks). "
            "Consider additional practice in this subject."
        )

    st.divider()

    # -------------------------------------------------
    # EDIT / DELETE MARKS
    # -------------------------------------------------

    st.subheader("🛠️ Manage Subject Records")

    st.dataframe(
        subject_data,
        use_container_width=True,
        hide_index=True
    )

    record_index = st.selectbox(
        "Select record to manage",
        subject_data.index,
        format_func=lambda i: (
            f"{subject_data.loc[i, 'Student']} | "
            f"{subject_data.loc[i, 'Subject']} | "
            f"{subject_data.loc[i, 'Marks']} marks"
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🗑️ Delete Selected Record", use_container_width=True):

            subject_data = subject_data.drop(
                index=record_index
            ).reset_index(drop=True)

            save_subject_data(
                subject_data.drop(columns=["Result"], errors="ignore")
            )

            st.success("Record deleted.")
            st.rerun()

    with col2:

        st.download_button(
            "📥 Download Subject Data",
            subject_data.drop(columns=["Result"], errors="ignore").to_csv(
                index=False
            ),
            file_name="subject_marks.csv",
            mime="text/csv",
            use_container_width=True
        )
