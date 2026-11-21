
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import date

# ============================================================
# INTERVENTION EFFECTIVENESS ANALYZER
# ============================================================

DATA_FILE = Path("data/interventions.csv")

INTERVENTION_TYPES = [
    "Mentor Meeting",
    "Remedial Classes",
    "Attendance Support",
    "Personal Study Plan",
    "Assignment Support",
    "Parent/Guardian Meeting",
    "Peer Learning",
    "Counselling / Guidance",
    "Other"
]


def load_interventions():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass

    return pd.DataFrame(columns=[
        "Student ID",
        "Student Name",
        "Intervention Type",
        "Intervention Date",
        "Faculty / Mentor",
        "Reason",
        "Before Attendance",
        "After Attendance",
        "Before Marks",
        "After Marks",
        "Before Assignments",
        "After Assignments",
        "Notes"
    ])


def save_interventions(df):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)


def render_intervention_effectiveness(students):

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0F766E,#14B8A6);
        padding:25px;
        border-radius:18px;
        color:white;">
        <h2 style="margin:0;">🎯 Intervention Effectiveness Analyzer</h2>
        <p style="margin-bottom:0;">
        Track student support activities and analyze academic improvement
        after interventions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    interventions = load_interventions()

    # ============================================================
    # STUDENT DATA
    # ============================================================

    student_options = {}

    if not students.empty:
        for _, row in students.iterrows():
            student_id = str(row.get("Student ID", ""))
            student_name = str(row.get("Student", "Unknown"))
            if student_id:
                student_options[f"{student_name} ({student_id})"] = student_id

    # ============================================================
    # ADD INTERVENTION
    # ============================================================

    st.subheader("➕ Record a Student Intervention")

    if not student_options:
        st.warning("No student records available.")
    else:

        with st.form("intervention_record_form"):

            selected_student = st.selectbox(
                "Select Student",
                list(student_options.keys())
            )

            intervention_type = st.selectbox(
                "Intervention Type",
                INTERVENTION_TYPES
            )

            intervention_date = st.date_input(
                "Intervention Date",
                value=date.today()
            )

            faculty_name = st.text_input(
                "Faculty / Mentor Name"
            )

            reason = st.text_area(
                "Reason for Intervention",
                placeholder="Explain why the student was selected for support..."
            )

            st.markdown("### 📊 Student Performance Before Intervention")

            c1, c2, c3 = st.columns(3)

            with c1:
                before_attendance = st.number_input(
                    "Attendance (%) Before",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )

            with c2:
                before_marks = st.number_input(
                    "Marks (%) Before",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0
                )

            with c3:
                before_assignments = st.number_input(
                    "Assignments (%) Before",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )

            st.markdown("### 📈 Follow-up Performance")

            c4, c5, c6 = st.columns(3)

            with c4:
                after_attendance = st.number_input(
                    "Attendance (%) After",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )

            with c5:
                after_marks = st.number_input(
                    "Marks (%) After",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0
                )

            with c6:
                after_assignments = st.number_input(
                    "Assignments (%) After",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )

            notes = st.text_area(
                "Follow-up Notes",
                placeholder="Describe the student's progress or observations..."
            )

            submitted = st.form_submit_button(
                "💾 Save Intervention Record",
                use_container_width=True
            )

            if submitted:

                new_record = {
                    "Student ID": student_options[selected_student],
                    "Student Name": selected_student.split(" (")[0],
                    "Intervention Type": intervention_type,
                    "Intervention Date": str(intervention_date),
                    "Faculty / Mentor": faculty_name,
                    "Reason": reason,
                    "Before Attendance": before_attendance,
                    "After Attendance": after_attendance,
                    "Before Marks": before_marks,
                    "After Marks": after_marks,
                    "Before Assignments": before_assignments,
                    "After Assignments": after_assignments,
                    "Notes": notes
                }

                interventions = pd.concat(
                    [interventions, pd.DataFrame([new_record])],
                    ignore_index=True
                )

                save_interventions(interventions)

                st.success("Intervention record saved successfully!")
                st.rerun()

    st.divider()

    # ============================================================
    # ANALYTICS
    # ============================================================

    st.subheader("📊 Intervention Analytics")

    if interventions.empty:

        st.info(
            "No intervention records available yet. "
            "Add intervention records above to unlock analytics."
        )

    else:

        # Calculate improvements
        interventions["Attendance Improvement"] = (
            interventions["After Attendance"] -
            interventions["Before Attendance"]
        )

        interventions["Marks Improvement"] = (
            interventions["After Marks"] -
            interventions["Before Marks"]
        )

        interventions["Assignment Improvement"] = (
            interventions["After Assignments"] -
            interventions["Before Assignments"]
        )

        interventions["Overall Improvement"] = (
            interventions["Attendance Improvement"] +
            interventions["Marks Improvement"] +
            interventions["Assignment Improvement"]
        ) / 3

        # ========================================================
        # KPI CARDS
        # ========================================================

        total_records = len(interventions)

        improved_records = (
            interventions["Overall Improvement"] > 0
        ).sum()

        avg_improvement = (
            interventions["Overall Improvement"].mean()
        )

        unique_students = interventions["Student ID"].nunique()

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Interventions Recorded", total_records)

        k2.metric("Students Supported", unique_students)

        k3.metric(
            "Improvement Records",
            f"{improved_records} / {total_records}"
        )

        k4.metric(
            "Average Improvement",
            f"{avg_improvement:+.2f}%"
        )

        st.write("")

        # ========================================================
        # BEFORE VS AFTER
        # ========================================================

        st.markdown("### Before vs After Intervention")

        before_after = pd.DataFrame({
            "Metric": ["Attendance", "Marks", "Assignments"],
            "Before": [
                interventions["Before Attendance"].mean(),
                interventions["Before Marks"].mean(),
                interventions["Before Assignments"].mean()
            ],
            "After": [
                interventions["After Attendance"].mean(),
                interventions["After Marks"].mean(),
                interventions["After Assignments"].mean()
            ]
        })

        before_after_long = before_after.melt(
            id_vars="Metric",
            value_vars=["Before", "After"],
            var_name="Period",
            value_name="Average Score"
        )

        fig = px.bar(
            before_after_long,
            x="Metric",
            y="Average Score",
            color="Period",
            barmode="group",
            text_auto=".1f",
            title="Average Student Performance Before and After"
        )

        fig.update_layout(
            template="plotly_white",
            yaxis_range=[0, 100],
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        # ========================================================
        # IMPROVEMENT BY INTERVENTION TYPE
        # ========================================================

        st.markdown("### 🏆 Intervention Type Effectiveness")

        intervention_summary = (
            interventions.groupby("Intervention Type")
            .agg(
                Records=("Student ID", "count"),
                Avg_Attendance_Improvement=("Attendance Improvement", "mean"),
                Avg_Marks_Improvement=("Marks Improvement", "mean"),
                Avg_Assignment_Improvement=("Assignment Improvement", "mean"),
                Avg_Overall_Improvement=("Overall Improvement", "mean")
            )
            .reset_index()
        )

        intervention_summary = intervention_summary.sort_values(
            "Avg_Overall_Improvement",
            ascending=False
        )

        fig2 = px.bar(
            intervention_summary,
            x="Intervention Type",
            y="Avg_Overall_Improvement",
            color="Avg_Overall_Improvement",
            text_auto=".2f",
            title="Average Improvement by Intervention",
            color_continuous_scale="Teal"
        )

        fig2.update_layout(
            template="plotly_white",
            height=450,
            xaxis_tickangle=-30
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ========================================================
        # IMPROVEMENT DISTRIBUTION
        # ========================================================

        st.markdown("### 📈 Improvement Distribution")

        fig3 = px.histogram(
            interventions,
            x="Overall Improvement",
            nbins=15,
            title="Distribution of Student Improvement",
            color_discrete_sequence=["#14B8A6"]
        )

        fig3.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(fig3, use_container_width=True)

        # ========================================================
        # STUDENT INTERVENTION HISTORY
        # ========================================================

        st.markdown("### 👨‍🎓 Student Intervention History")

        display_data = interventions.sort_values(
            "Intervention Date",
            ascending=False
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        # ========================================================
        # DOWNLOAD REPORT
        # ========================================================

        st.markdown("### 📥 Download Reports")

        csv_data = interventions.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Intervention Report (CSV)",
            data=csv_data,
            file_name="intervention_effectiveness_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ========================================================
        # INTERVENTION SUMMARY
        # ========================================================

        st.markdown("### 📋 Intervention Summary")

        st.dataframe(
            intervention_summary.rename(columns={
                "Intervention Type": "Intervention Type",
                "Records": "Total Records",
                "Avg_Attendance_Improvement": "Average Attendance Improvement",
                "Avg_Marks_Improvement": "Average Marks Improvement",
                "Avg_Assignment_Improvement": "Average Assignment Improvement",
                "Avg_Overall_Improvement": "Overall Improvement"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Note: These metrics represent observed changes after interventions. "
            "They do not prove that the intervention alone caused the improvement."
        )
