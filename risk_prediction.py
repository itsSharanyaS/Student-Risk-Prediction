
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
# 1. LOAD AI MODELS
# =========================================================

@st.cache_resource
def load_trained_models():
    return train_models()


# =========================================================
# 2. RISK COLORS
# =========================================================

DEFAULT_RISK_COLORS = {
    "Low": "#22C55E",
    "Moderate": "#F59E0B",
    "High": "#F97316",
    "Critical": "#EF4444"
}


# =========================================================
# 3. MAIN RISK PREDICTION PAGE
# =========================================================

def render_risk_prediction_page(students, apply_chart_theme, risk_colors):

    st.title("🤖 AI Risk Prediction Center")

    st.caption(
        "Machine Learning-powered student risk analysis "
        "using attendance, marks and assignment performance."
    )

    colors = risk_colors if isinstance(risk_colors, dict) else DEFAULT_RISK_COLORS

    # -----------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------

    try:
        with st.spinner("Training AI models..."):
            trained_models, evaluation_results, training_data = (
                load_trained_models()
            )

    except Exception as e:
        st.error("Unable to train the AI models.")
        st.exception(e)
        return

    if not trained_models:
        st.error("No AI models were trained successfully.")
        return

    # -----------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------

    st.divider()

    st.subheader("🧠 Model Selection")

    model_names = list(trained_models.keys())

    selected_model_name = st.selectbox(
        "Choose Machine Learning Model",
        model_names,
        index=0
    )

    selected_model = trained_models[selected_model_name]

    st.info(f"Selected Model: **{selected_model_name}**")

    # -----------------------------------------------------
    # INPUT MODE
    # -----------------------------------------------------

    st.divider()

    st.subheader("🎓 Student Data")

    input_mode = st.radio(
        "Choose prediction input method",
        ["Existing Student", "Manual Input"],
        horizontal=True
    )

    attendance = 0
    marks = 0
    assignments = 0
    student_name = "New Student"
    student_id = "Manual Input"

    # =====================================================
    # EXISTING STUDENT INPUT
    # =====================================================

    if input_mode == "Existing Student":

        if students is None or students.empty:

            st.warning("Student dataset is empty.")

            input_mode = "Manual Input"

        else:

            required_columns = [
                "Student ID",
                "Student",
                "Attendance",
                "Marks",
                "Assignments"
            ]

            missing_columns = [
                col for col in required_columns
                if col not in students.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

                return

            student_ids = students["Student ID"].astype(str).tolist()

            selected_student_id = st.selectbox(
                "Select Student ID",
                student_ids
            )

            student_row = students[
                students["Student ID"].astype(str)
                == selected_student_id
            ].iloc[0]

            student_name = str(student_row["Student"])
            student_id = str(student_row["Student ID"])

            attendance = float(student_row["Attendance"])
            marks = float(student_row["Marks"])
            assignments = float(student_row["Assignments"])

            st.markdown(f"### 👤 {student_name}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Attendance", f"{attendance:.1f}%")
            col2.metric("Marks", f"{marks:.1f}%")
            col3.metric("Assignments", f"{assignments:.1f}%")

    # =====================================================
    # MANUAL INPUT
    # =====================================================

    if input_mode == "Manual Input":

        st.markdown("### 📝 Enter Student Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            attendance = st.slider(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=75
            )

        with col2:
            marks = st.slider(
                "Marks (%)",
                min_value=0,
                max_value=100,
                value=65
            )

        with col3:
            assignments = st.slider(
                "Assignments (%)",
                min_value=0,
                max_value=100,
                value=70
            )

    # =====================================================
    # PREDICTION
    # =====================================================

    st.divider()

    st.subheader("🔮 Risk Analysis")

    try:

        prediction, probabilities, high_risk_probability = predict_student(
            selected_model,
            attendance,
            marks,
            assignments
        )

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)
        return

    risk_color = colors.get(
        prediction,
        DEFAULT_RISK_COLORS.get(prediction, "#64748B")
    )

    risk_icons = {
        "Low": "🟢",
        "Moderate": "🟡",
        "High": "🟠",
        "Critical": "🔴"
    }

    risk_icon = risk_icons.get(prediction, "⚪")

    # -----------------------------------------------------
    # RISK RESULT
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div style="
            background-color:{risk_color}18;
            border:2px solid {risk_color};
            padding:22px;
            border-radius:15px;
            text-align:center;
        ">
            <h2 style="margin:0;color:{risk_color};">
                {risk_icon} {prediction} Risk
            </h2>
            <p style="margin-top:10px;">
                Student Risk Prediction Result
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Risk", prediction)

    col2.metric(
        "High / Critical Risk Probability",
        f"{high_risk_probability:.1f}%"
    )

    col3.metric(
        "Selected AI Model",
        selected_model_name
    )

    # =====================================================
    # PROBABILITY DISTRIBUTION
    # =====================================================

    st.subheader("📊 Risk Probability Distribution")

    probability_df = pd.DataFrame({
        "Risk Level": list(probabilities.keys()),
        "Probability (%)": list(probabilities.values())
    })

    probability_df = probability_df.sort_values(
        "Probability (%)",
        ascending=False
    )

    fig_probability = px.bar(
        probability_df,
        x="Risk Level",
        y="Probability (%)",
        color="Risk Level",
        color_discrete_map=colors,
        text="Probability (%)",
        title="Predicted Probability for Each Risk Category"
    )

    fig_probability.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_probability.update_layout(
        yaxis_range=[0, 110],
        showlegend=False
    )

    try:
        apply_chart_theme(fig_probability)
    except Exception:
        pass

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

    # =====================================================
    # STUDENT INTERPRETATION
    # =====================================================

    st.subheader("📋 Student Performance Interpretation")

    col1, col2, col3 = st.columns(3)

    with col1:

        if attendance < 60:
            st.error(f"Attendance is low: {attendance:.1f}%")
        elif attendance < 75:
            st.warning(f"Attendance needs improvement: {attendance:.1f}%")
        else:
            st.success(f"Attendance is satisfactory: {attendance:.1f}%")

    with col2:

        if marks < 40:
            st.error(f"Academic marks are poor: {marks:.1f}%")
        elif marks < 60:
            st.warning(f"Academic marks need improvement: {marks:.1f}%")
        else:
            st.success(f"Academic marks are satisfactory: {marks:.1f}%")

    with col3:

        if assignments < 50:
            st.error(f"Assignment completion is low: {assignments:.1f}%")
        elif assignments < 75:
            st.warning(f"Assignment completion needs improvement: {assignments:.1f}%")
        else:
            st.success(f"Assignment completion is good: {assignments:.1f}%")

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.subheader("💡 Recommended Academic Interventions")

    recommendations = {

        "Low": [
            "Maintain consistent attendance.",
            "Continue regular academic performance monitoring.",
            "Participate in advanced learning activities.",
            "Maintain timely assignment submissions."
        ],

        "Moderate": [
            "Schedule regular mentoring sessions.",
            "Improve attendance consistency.",
            "Create a weekly study plan.",
            "Monitor assignment submission progress."
        ],

        "High": [
            "Arrange individual academic counselling.",
            "Identify subjects with low performance.",
            "Create a personalized improvement plan.",
            "Conduct weekly progress reviews."
        ],

        "Critical": [
            "Arrange immediate faculty intervention.",
            "Conduct a detailed academic assessment.",
            "Assign a dedicated mentor.",
            "Create an intensive academic recovery plan.",
            "Review progress every week."
        ]
    }

    for recommendation in recommendations.get(prediction, []):
        st.markdown(f"- {recommendation}")

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.divider()

    st.subheader("🏆 Machine Learning Model Performance")

    try:

        comparison_df = get_model_comparison(evaluation_results)

        if not comparison_df.empty:

            st.dataframe(
                comparison_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            # Best model
            best_model_name, best_model = get_best_model(
                trained_models,
                evaluation_results
            )

            if best_model_name is not None:

                best_f1 = evaluation_results[best_model_name]["F1 Score"]

                col1, col2 = st.columns(2)

                col1.metric(
                    "🏆 Best Performing Model",
                    best_model_name
                )

                col2.metric(
                    "Best F1 Score",
                    f"{best_f1:.4f}"
                )

            # Performance chart
            metric_columns = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]

            available_metrics = [
                col for col in metric_columns
                if col in comparison_df.columns
            ]

            if available_metrics:

                fig_model = px.bar(
                    comparison_df,
                    x="Model",
                    y=available_metrics,
                    barmode="group",
                    title="Machine Learning Model Comparison"
                )

                fig_model.update_layout(
                    yaxis_range=[0, 1.1]
                )

                try:
                    apply_chart_theme(fig_model)
                except Exception:
                    pass

                st.plotly_chart(
                    fig_model,
                    use_container_width=True
                )

    except Exception as e:

        st.warning("Unable to display model comparison.")
        st.exception(e)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.subheader("🔍 Feature Importance Analysis")

    try:

        importance_df = get_feature_importance(selected_model)

        if (
            importance_df is not None
            and not importance_df.empty
            and "Importance" in importance_df.columns
        ):

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Feature Importance - {selected_model_name}",
                text="Importance"
            )

            fig_importance.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            try:
                apply_chart_theme(fig_importance)
            except Exception:
                pass

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

            st.caption(
                "Feature importance shows how strongly input features "
                "contribute to the trained model. It does not prove causation."
            )

        else:

            st.info("Feature importance is unavailable for this model.")

    except Exception as e:

        st.warning("Unable to calculate feature importance.")
        st.exception(e)

    # =====================================================
    # FINAL DISCLAIMER
    # =====================================================

    st.divider()

    st.caption(
        "⚠️ This prediction is based on experimental machine learning "
        "models and synthetic training data. It should be used as an "
        "academic support tool only, not as the sole basis for decisions "
        "about students."
    )
