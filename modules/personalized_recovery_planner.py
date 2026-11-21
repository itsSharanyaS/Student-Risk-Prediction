
import streamlit as st
import pandas as pd


# =========================================================
# PERSONALIZED ACADEMIC RECOVERY PLANNER
# =========================================================

def safe_number(value):
    value = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(value) else float(value)


def generate_recovery_plan(student):

    attendance = safe_number(student.get("Attendance", 0))
    marks = safe_number(student.get("Marks", 0))
    assignments = safe_number(student.get("Assignments", 0))

    # Identify priority areas
    priority_areas = []

    if attendance < 75:
        priority_areas.append("Attendance Improvement")

    if marks < 60:
        priority_areas.append("Academic Performance")

    if assignments < 60:
        priority_areas.append("Assignment Completion")

    if not priority_areas:
        priority_areas.append("Continuous Academic Improvement")

    # Weakest area
    scores = {
        "Attendance": attendance,
        "Academic Performance": marks,
        "Assignment Completion": assignments
    }

    weakest_area = min(scores, key=scores.get)

    # Personalized weekly targets
    targets = {
        "Attendance": "Attend all scheduled classes and aim for 90%+ attendance.",
        "Academic Performance": "Study for 2 hours daily and practice difficult concepts.",
        "Assignment Completion": "Complete pending assignments and submit them on time."
    }

    # Personalized 7-day schedule
    if weakest_area == "Attendance":

        schedule = [
            ("Attendance Reset", "Create a class attendance routine and attend all scheduled classes.", "30 min", "Attendance"),
            ("Concept Catch-up", "Revise topics missed during previous classes.", "2 hrs", "Learning"),
            ("Doubt Clearing", "Ask faculty or classmates about missed concepts.", "45 min", "Support"),
            ("Focused Learning", "Revise class notes and understand important concepts.", "2 hrs", "Revision"),
            ("Consistency Check", "Review attendance and prepare for upcoming classes.", "30 min", "Attendance"),
            ("Practice Day", "Solve questions from the topics studied this week.", "1.5 hrs", "Practice"),
            ("Weekly Review", "Review progress and prepare next week's class routine.", "30 min", "Review")
        ]

    elif weakest_area == "Academic Performance":

        schedule = [
            ("Learning Reset", "Identify difficult concepts and organize study materials.", "1 hr", "Planning"),
            ("Concept Building", "Learn weak topics and write short concept notes.", "2 hrs", "Learning"),
            ("Guided Practice", "Solve basic questions from the topics studied.", "2 hrs", "Practice"),
            ("Deep Revision", "Revise difficult concepts and work through examples.", "2 hrs", "Revision"),
            ("Problem Solving", "Practice questions without referring to notes.", "2 hrs", "Practice"),
            ("Mini Assessment", "Take a short test and identify mistakes.", "1 hr", "Assessment"),
            ("Weekly Review", "Review mistakes and create next week's improvement goals.", "45 min", "Review")
        ]

    elif weakest_area == "Assignment Completion":

        schedule = [
            ("Task Audit", "List incomplete assignments and their deadlines.", "30 min", "Planning"),
            ("Assignment Sprint", "Start the most urgent incomplete assignment.", "2 hrs", "Assignments"),
            ("Concept Support", "Study concepts needed to complete pending work.", "1.5 hrs", "Learning"),
            ("Assignment Sprint", "Continue and complete pending assignment questions.", "2 hrs", "Assignments"),
            ("Submission Check", "Review answers, formatting and submission requirements.", "1 hr", "Review"),
            ("Final Completion", "Finish remaining work and submit assignments.", "2 hrs", "Completion"),
            ("Weekly Review", "Check completed work and plan upcoming assignments.", "30 min", "Review")
        ]

    else:

        schedule = [
            ("Weekly Planning", "Organize notes and set learning goals for the week.", "45 min", "Planning"),
            ("Concept Revision", "Revise important topics and strengthen understanding.", "2 hrs", "Learning"),
            ("Practice Session", "Solve coding, theory or subject-related questions.", "2 hrs", "Practice"),
            ("Focused Learning", "Study a challenging topic and prepare short notes.", "2 hrs", "Learning"),
            ("Assignment Review", "Review assignments and complete remaining work.", "1 hr", "Assignments"),
            ("Self Assessment", "Take a mini-test and review your mistakes.", "1 hr", "Assessment"),
            ("Weekly Reflection", "Evaluate progress and prepare next week's targets.", "45 min", "Review")
        ]

    # Mentor action plan
    mentor_actions = []

    if attendance < 75:
        mentor_actions.append(
            "Discuss attendance challenges and establish a consistent class routine."
        )

    if marks < 60:
        mentor_actions.append(
            "Identify difficult topics and arrange faculty support or remedial learning."
        )

    if assignments < 60:
        mentor_actions.append(
            "Review pending assignments and set realistic submission deadlines."
        )

    if not mentor_actions:
        mentor_actions.append(
            "Conduct a weekly check-in to maintain consistent academic performance."
        )

    return {
        "Priority Areas": priority_areas,
        "Weakest Area": weakest_area,
        "Weekly Targets": targets,
        "Study Schedule": schedule,
        "Mentor Actions": mentor_actions
    }


# =========================================================
# RECOVERY PLANNER UI
# =========================================================

def render_recovery_planner(df):

    st.title("🧠 Personalized Academic Recovery Planner")

    st.caption(
        "A structured 7-day academic improvement journey "
        "based on individual student performance."
    )

    if df is None or df.empty:
        st.warning("No student data available.")
        return

    # Student selection
    student_options = df["Student"].tolist()

    selected_student = st.selectbox(
        "Select Student",
        student_options,
        key="recovery_planner_student"
    )

    student = df[df["Student"] == selected_student].iloc[0]

    plan = generate_recovery_plan(student)

    student_id = str(student.get("Student ID", selected_student))

    # -----------------------------------------------------
    # STUDENT OVERVIEW
    # -----------------------------------------------------

    st.divider()

    st.subheader("📊 Student Academic Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Attendance", f"{safe_number(student['Attendance']):.1f}%")
    col2.metric("Marks", f"{safe_number(student['Marks']):.1f}%")
    col3.metric("Assignments", f"{safe_number(student['Assignments']):.1f}%")

    # -----------------------------------------------------
    # PRIORITY AREAS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🎯 Priority Areas")

    for area in plan["Priority Areas"]:
        st.warning(f"Focus Area: {area}")

    st.info(f"**Primary Improvement Area:** {plan['Weakest Area']}")

    # -----------------------------------------------------
    # WEEKLY TARGETS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🏆 Weekly Goals")

    target_cols = st.columns(3)

    target_list = list(plan["Weekly Targets"].items())

    for i, (area, target) in enumerate(target_list):
        with target_cols[i]:
            with st.container(border=True):
                st.markdown(f"**{area}**")
                st.write(target)

    # -----------------------------------------------------
    # ATTRACTIVE 7-DAY STUDY ROADMAP
    # -----------------------------------------------------

    st.divider()

    st.subheader("📅 Your 7-Day Recovery Journey")

    st.markdown(
        """
        Build better habits one day at a time.
        Complete your daily tasks and track your weekly progress below.
        """
    )

    # Progress tracking
    tracker_key = f"recovery_progress_{student_id}"

    if tracker_key not in st.session_state:
        st.session_state[tracker_key] = [False] * 7

    completed_days = sum(st.session_state[tracker_key])
    progress = completed_days / 7

    st.progress(
        progress,
        text=f"Weekly Progress: {completed_days}/7 days completed"
    )

    # Daily roadmap cards
    day_colors = [
        "#3B82F6",
        "#8B5CF6",
        "#10B981",
        "#F59E0B",
        "#EC4899",
        "#F97316",
        "#14B8A6"
    ]

    for i, (title, activity, duration, category) in enumerate(
        plan["Study Schedule"]
    ):

        with st.container(border=True):

            col1, col2 = st.columns([1, 5])

            with col1:
                st.markdown(
                    f"""
                    <div style="
                        background:{day_colors[i]};
                        padding:14px 8px;
                        border-radius:12px;
                        text-align:center;
                        color:white;">
                        <h2 style="margin:0;color:white;">{i + 1:02}</h2>
                        <small>DAY</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(f"### {title}")

                st.caption(f"📌 {category}  ·  ⏱ {duration}")

                st.write(activity)

            # Completion checkbox
            st.session_state[tracker_key][i] = st.checkbox(
                f"Mark Day {i + 1} as completed",
                value=st.session_state[tracker_key][i],
                key=f"recovery_day_{student_id}_{i}"
            )

    # Updated progress after checkboxes
    completed_days = sum(st.session_state[tracker_key])

    st.progress(
        completed_days / 7,
        text=f"Completed {completed_days} of 7 days"
    )

    if completed_days == 7:
        st.success("🎉 Amazing! You completed your entire 7-day recovery plan!")

    elif completed_days >= 4:
        st.success("👏 Great progress! Keep building your academic routine.")

    elif completed_days > 0:
        st.info("🌱 Good start! Keep going and complete the remaining days.")

    # -----------------------------------------------------
    # MENTOR ACTION PLAN
    # -----------------------------------------------------

    st.divider()

    st.subheader("👨‍🏫 Mentor Action Plan")

    for i, action in enumerate(plan["Mentor Actions"], 1):
        st.markdown(f"**{i}.** {action}")

    # -----------------------------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------------------------

    st.divider()

    st.subheader("📥 Download Recovery Plan")

    schedule_text = "\n".join(
        [
            f"Day {i + 1}: {title} | {activity} | Duration: {duration}"
            for i, (title, activity, duration, category)
            in enumerate(plan["Study Schedule"])
        ]
    )

    targets_text = "\n".join(
        [f"{area}: {target}" for area, target in plan["Weekly Targets"].items()]
    )

    mentor_text = "\n".join(plan["Mentor Actions"])

    report = f"""
PERSONALIZED ACADEMIC RECOVERY PLAN
===================================

Student: {student['Student']}
Student ID: {student_id}

Attendance: {student['Attendance']}%
Marks: {student['Marks']}%
Assignments: {student['Assignments']}%

PRIMARY IMPROVEMENT AREA
-------------------------
{plan['Weakest Area']}

PRIORITY AREAS
--------------
{chr(10).join(plan['Priority Areas'])}

WEEKLY TARGETS
--------------
{targets_text}

7-DAY STUDY ROADMAP
--------------------
{schedule_text}

MENTOR ACTION PLAN
-----------------
{mentor_text}

WEEKLY PLAN COMPLETION
---------------------
{completed_days}/7 days completed
"""

    st.download_button(
        "⬇️ Download Personalized Recovery Plan",
        data=report,
        file_name=f"{student_id}_recovery_plan.txt",
        mime="text/plain",
        key=f"download_recovery_{student_id}"
    )
