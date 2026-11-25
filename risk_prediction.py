
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


@st.cache_resource
def load_trained_models():
    return train_models()


def render_risk_prediction_page(
    students,
    apply_chart_theme=None,
    risk_colors=None
):

    st.title("🎯 Risk Prediction Center")

    st.caption(
        "AI-powered student risk analysis, model evaluation "
        "and academic performance insights."
    )

    if risk_colors is None:
        risk_colors = {
            "Low": "#22C55E",
            "Moderate": "#F59E0B",
            "High": "#F97316",
            "Critical": "#EF4444"
        }

    try:
        trained_models, evaluation_results, training_data = (
            load_trained_models()
        )

    except Exception as e:
        st.error(f"Model training error: {e}")
        return

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.divider()
    st.subheader("📊 Model Comparison")

    comparison_df = get_model_comparison(evaluation_results)

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )

    metric_options = ["Accuracy", "Precision", "Recall", "F1 Score"]

    selected_metric = st.selectbox(
        "Select metric to compare",
        metric_options,
        key="model_comparison_metric"
    )

    fig_comparison = px.bar(
        comparison_df,
        x="Model",
        y=selected_metric,
        color="Model",
        text=selected_metric,
        title=f"Model Comparison — {selected_metric}"
    )

    fig_comparison.update_layout(
        yaxis_title="Score (%)",
        yaxis_range=[0, 100]
    )

    fig_comparison.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    if apply_chart_theme:
        fig_comparison = apply_chart_theme(fig_comparison)

    st.plotly_chart(
        fig_comparison,
        width="stretch"
    )

    best_model_name, best_model = get_best_model(
        trained_models,
        evaluation_results
    )

    st.success(
        f"🏆 Best performing model: {best_model_name} "
        f"(F1 Score: {evaluation_results[best_model_name]['F1 Score']}%)"
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()
    st.subheader("🎯 Feature Importance")

    importance_df = get_feature_importance(best_model)

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title=f"Factors Influencing Risk — {best_model_name}"
    )

    fig_importance.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_importance.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title="Feature"
    )

    if apply_chart_theme:
        fig_importance = apply_chart_theme(fig_importance)

    st.plotly_chart(
        fig_importance,
        width="stretch"
    )

    st.caption(
        "Feature importance represents the relative contribution of "
        "each feature to the trained model. It does not indicate causation."
    )

    # =====================================================
    # STUDENT PREDICTION
    # =====================================================

    st.divider()
    st.subheader("🔍 Individual Student Risk Prediction")

    input_mode = st.radio(
        "Choose prediction method",
        ["Select Existing Student", "Manual Input"],
        horizontal=True,
        key="risk_prediction_mode"
    )

    attendance = 0
    marks = 0
    assignments = 0

    if input_mode == "Select Existing Student":

        if students is None or students.empty:
            st.warning("No student records are available.")
            return

        student_id_col = "Student ID"

        if student_id_col not in students.columns:
            st.warning("Student ID column not found.")
            return

        student_ids = students[student_id_col].astype(str).tolist()

        selected_id = st.selectbox(
            "Select Student ID",
            student_ids,
            key="risk_selected_student"
        )

        student_row = students[
            students[student_id_col].astype(str) == selected_id
        ].iloc[0]

        required_columns = ["Attendance", "Marks", "Assignments"]

        missing = [
            column for column in required_columns
            if column not in students.columns
        ]

        if missing:
            st.error(f"Missing columns: {missing}")
            return

        attendance = float(student_row["Attendance"])
        marks = float(student_row["Marks"])
        assignments = float(student_row["Assignments"])

        st.write(f"**Student:** {student_row.get('Student', selected_id)}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Attendance", f"{attendance:.1f}%")
        col2.metric("Marks", f"{marks:.1f}%")
        col3.metric("Assignments", f"{assignments:.1f}%")

    else:

        col1, col2, col3 = st.columns(3)

        attendance = col1.slider(
            "Attendance (%)",
            0, 100, 75,
            key="manual_attendance"
        )

        marks = col2.slider(
            "Marks (%)",
            0, 100, 65,
            key="manual_marks"
        )

        assignments = col3.slider(
            "Assignments (%)",
            0, 100, 70,
            key="manual_assignments"
        )

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "🚀 Predict Student Risk",
        type="primary",
        width="stretch"
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

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Risk",
            prediction
        )

        col2.metric(
            "High / Critical Risk Probability",
            f"{high_risk_probability:.2f}%"
        )

        st.markdown(
            f"""
            <div style="
                background-color:{risk_colors.get(prediction, '#808080')};
                padding:18px;
                border-radius:12px;
                color:white;
                text-align:center;
                font-size:22px;
                font-weight:bold;
            ">
                Predicted Risk Level: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Probability chart

        probability_df = pd.DataFrame({
            "Risk Level": list(probabilities.keys()),
            "Probability (%)": list(probabilities.values())
        })

        fig_probability = px.bar(
            probability_df,
            x="Risk Level",
            y="Probability (%)",
            color="Risk Level",
            color_discrete_map=risk_colors,
            text="Probability (%)",
            title="Risk Probability Distribution"
        )

        fig_probability.update_layout(
            yaxis_range=[0, 100]
        )

        fig_probability.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        if apply_chart_theme:
            fig_probability = apply_chart_theme(fig_probability)

        st.plotly_chart(
            fig_probability,
            width="stretch"
        )

        # Recommendations

        st.subheader("💡 Recommended Actions")

        recommendations = {
            "Low": [
                "Maintain current academic performance.",
                "Continue regular attendance.",
                "Participate in academic activities."
            ],

            "Moderate": [
                "Monitor academic performance regularly.",
                "Improve assignment submission consistency.",
                "Encourage additional learning practice."
            ],

            "High": [
                "Arrange faculty mentoring sessions.",
                "Improve attendance and assignment completion.",
                "Provide targeted academic support."
            ],

            "Critical": [
                "Arrange immediate faculty intervention.",
                "Create a personalized improvement plan.",
                "Monitor attendance and progress weekly.",
                "Provide additional academic assistance."
            }
        }

        for action in recommendations.get(prediction, []):
            st.write(f"• {action}")

    # =====================================================
    # MODEL METRICS
    # =====================================================

    st.divider()
    st.subheader("📈 Model Performance Metrics")

    metric_columns = st.columns(len(evaluation_results))

    for column, (model_name, metrics) in zip(
        metric_columns,
        evaluation_results.items()
    ):

        with column:
            st.markdown(f"**{model_name}**")

            st.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.2f}%"
            )

            st.metric(
                "Precision",
                f"{metrics['Precision']:.2f}%"
            )

            st.metric(
                "Recall",
                f"{metrics['Recall']:.2f}%"
            )

            st.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.2f}%"
            )

    st.caption(
        "Models are trained on synthetic data for demonstration. "
        "Predictions should not be used as the sole basis for "
        "academic decisions."
    )
