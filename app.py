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

    - Machine learning risk prediction
    - Explainable AI
    - Risk forecasting
    - Academic What-If Simulator
    - Student Risk Trend Analyzer
    - Subject-Level Intelligence
    - Student clustering
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
