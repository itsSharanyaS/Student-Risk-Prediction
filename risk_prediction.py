
import streamlit as st
import pandas as pd
import plotly.express as px

from ml_engine import (
    train_models,
    predict_student,
    get_model_comparison,
    get_best_model,
    get_feature_importance
)


# =========================================================
# 1. TRAIN AND CACHE MODELS
# =========================================================

@st.cache_resource
def load_trained_models():

    return train_models()


# =========================================================
# 2. MAIN PAGE
# =========================================================

def render_risk_prediction_page(
    students,
    apply_chart_theme,
    risk_colors
):

    st.title("🤖 Risk Prediction Center")

    st.caption(
        "Machine Learning powered student risk analysis, "
        "model evaluation and feature importance."
    )

    # Load models
    trained_models, evaluation_results, X = load_trained_models()

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.divider()

    st.subheader("📊 Model Comparison")

    comparison = get_model_comparison(evaluation_results)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    # Model performance chart
    chart_data = comparison.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig_comparison = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Machine Learning Model Performance",
        text_auto=".1f"
    )

    fig_comparison.update_layout(
        yaxis_title="Performance (%)",
        yaxis_range=[0, 100]
    )

    fig_comparison = apply_chart_theme(fig_comparison)

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )

    # =====================================================
    # BEST MODEL
    # =====================================================

    best_model_name, best_model = get_best_model(
        trained_models,
        evaluation_results
    )

    best_f1 = evaluation_results[best_model_name]["F1 Score"]

    st.subheader("🏆 Best Performing Model")

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", best_model_name)

    col2.metric(
        "F1 Score",
        f"{best_f1}%"
    )

    col3.metric(
        "Models Compared",
        len(trained_models)
    )

    st.success(
        f"{best_model_name} is currently the best-performing "
        "model based on the F1 Score."
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.subheader("🔍 Feature Importance")

    feature_df = get_feature_importance(best_model)

    fig_importance = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title=f"Factors Influencing Risk — {best_model_name}"
    )

    fig_importance.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title="Student Feature",
        xaxis_range=[0, 100],
        yaxis=dict(categoryorder="total ascending")
    )

    fig_importance = apply_chart_theme(fig_importance)

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # STUDENT PREDICTION
    # =====================================================

    st.divider()

    st.subheader("🎓 Individual Student Risk Prediction")

    prediction_mode = st.radio(
        "Select Prediction Mode",
        ["Select Existing Student", "Manual Input"],
        horizontal=True
    )

    # -----------------------------------------------------
    # EXISTING STUDENT
    # -----------------------------------------------------

    if prediction_mode == "Select Existing Student":

        if students is None or students.empty:

            st.warning("No student records are available.")

            return

        student_ids = students["Student ID"].astype(str).tolist()

        selected_id = st.selectbox(
            "Select Student ID",
            student_ids
        )

        student_data = students[
            students["Student ID"].astype(str) == selected_id
        ].iloc[0]

        attendance = float(student_data["Attendance"])

        marks = float(student_data["Marks"])

        assignments = float(student_data["Assignments"])

        student_name = str(
            student_data.get("Student", selected_id)
        )

        st.info(f"Selected Student: {student_name}")

    # -----------------------------------------------------
    # MANUAL INPUT
    # -----------------------------------------------------

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        with col2:

            marks = st.number_input(
                "Marks (%)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0
            )

        with col3:

            assignments = st.number_input(
                "Assignments (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )

        student_name = "Manual Student"

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "🔮 Predict Student Risk",
        type="primary",
        use_container_width=True
    ):

        prediction, probabilities, high_risk_probability = (
            predict_student(
                best_model,
                attendance,
                marks,
                assignments
            )
        )

        st.divider()

        st.subheader("📋 Prediction Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Predicted Risk",
            prediction
        )

        col2.metric(
            "Attendance",
            f"{attendance:.1f}%"
        )

        col3.metric(
            "Marks",
            f"{marks:.1f}%"
        )

        col4.metric(
            "High/Critical Risk Probability",
            f"{high_risk_probability:.1f}%"
        )

        # -------------------------------------------------
        # RISK STATUS
        # -------------------------------------------------

        if prediction == "Critical":

            st.error(
                "🚨 Critical Risk: Immediate academic intervention is recommended."
            )

        elif prediction == "High":

            st.warning(
                "⚠️ High Risk: Student needs urgent monitoring and support."
            )

        elif prediction == "Moderate":

            st.info(
                "🟡 Moderate Risk: Regular monitoring and improvement are recommended."
            )

        else:

            st.success(
                "🟢 Low Risk: Student is currently performing well."
            )

        # -------------------------------------------------
        # PROBABILITY CHART
        # -------------------------------------------------

        st.subheader("📈 Risk Probability Distribution")

        probability_df = pd.DataFrame({
            "Risk Level": list(probabilities.keys()),
            "Probability (%)": list(probabilities.values())
        })

        fig_probability = px.bar(
            probability_df,
            x="Risk Level",
            y="Probability (%)",
            color="Risk Level",
            text="Probability (%)",
            title="Predicted Probability for Each Risk Category"
        )

        fig_probability.update_layout(
            yaxis_range=[0, 100]
        )

        fig_probability = apply_chart_theme(fig_probability)

        st.plotly_chart(
            fig_probability,
            use_container_width=True
        )

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        st.subheader("💡 Recommended Actions")

        if prediction == "Critical":

            st.markdown("""
            - Arrange an immediate faculty meeting with the student.
            - Develop a personalized academic recovery plan.
            - Monitor attendance and assignment submissions weekly.
            - Provide additional tutoring and mentoring.
            - Contact parents or guardians when appropriate.
            """)

        elif prediction == "High":

            st.markdown("""
            - Schedule faculty mentoring sessions.
            - Identify subjects where marks are low.
            - Improve attendance and assignment completion.
            - Review student progress every week.
            """)

        elif prediction == "Moderate":

            st.markdown("""
            - Monitor academic performance regularly.
            - Encourage consistent attendance.
            - Provide subject-specific learning resources.
            - Track assignment completion.
            """)

        else:

            st.markdown("""
            - Maintain consistent attendance.
            - Continue regular assignment submissions.
            - Encourage participation in academic activities.
            - Monitor performance to maintain progress.
            """)

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        st.subheader("📊 Student Input Summary")

        input_summary = pd.DataFrame({
            "Parameter": [
                "Attendance",
                "Marks",
                "Assignments"
            ],
            "Value (%)": [
                attendance,
                marks,
                assignments
            ]
        })

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )
