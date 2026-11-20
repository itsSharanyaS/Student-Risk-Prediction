
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


def render_risk_prediction_page(students, apply_chart_theme, risk_colors):

    st.title("🤖 AI Risk Prediction Center")
    st.caption(
        "Machine learning based academic risk assessment "
        "using attendance, marks, and assignment performance."
    )

    # Train and load models
    with st.spinner("Training AI models..."):
        trained_models, evaluation_results, training_data = load_trained_models()

    # Model selection
    model_names = list(trained_models.keys())

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        model_names
    )

    selected_model = trained_models[selected_model_name]

    st.divider()

    # Prediction mode
    prediction_mode = st.radio(
        "Prediction Mode",
        ["Existing Student", "Manual Input"],
        horizontal=True
    )

    if prediction_mode == "Existing Student":

        student_options = students["Student ID"].astype(str).tolist()

        selected_student_id = st.selectbox(
            "Select Student USN",
            student_options
        )

        student_row = students[
            students["Student ID"].astype(str) == selected_student_id
        ].iloc[0]

        student_name = student_row["Student"]

        attendance = float(student_row["Attendance"])
        marks = float(student_row["Marks"])
        assignments = float(student_row["Assignments"])

        st.markdown(f"### 👤 {student_name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Attendance", f"{attendance:.1f}%")
        col2.metric("Marks", f"{marks:.1f}%")
        col3.metric("Assignments", f"{assignments:.1f}%")

    else:

        st.subheader("Enter Academic Details")

        col1, col2, col3 = st.columns(3)

        attendance = col1.slider(
            "Attendance (%)",
            0.0,
            100.0,
            75.0
        )

        marks = col2.slider(
            "Marks (%)",
            0.0,
            100.0,
            65.0
        )

        assignments = col3.slider(
            "Assignments (%)",
            0.0,
            100.0,
            70.0
        )

        student_name = "Manual Student"

    # Prediction
    predicted_risk, probabilities, high_risk_probability = predict_student(
        selected_model,
        attendance,
        marks,
        assignments
    )

    risk_score = high_risk_probability * 100

    st.divider()

    # Result
    st.subheader("Prediction Result")

    if predicted_risk == "Low":
        result_color = "#16A34A"
        result_icon = "🟢"

    elif predicted_risk == "Moderate":
        result_color = "#F59E0B"
        result_icon = "🟡"

    elif predicted_risk == "High":
        result_color = "#F97316"
        result_icon = "🟠"

    else:
        result_color = "#DC2626"
        result_icon = "🔴"

    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background:{result_color}18;
            border:2px solid {result_color};
            text-align:center;
        ">
            <h2 style="color:{result_color};">
                {result_icon} {predicted_risk} Risk
            </h2>
            <h3>Predicted Academic Risk Category</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated High/Critical Probability",
        f"{risk_score:.2f}%"
    )

    col2.metric(
        "Selected ML Model",
        selected_model_name
    )

    st.divider()

    # Probability visualization
    st.subheader("Risk Probability Distribution")

    probability_data = pd.DataFrame({
        "Risk Level": list(probabilities.keys()),
        "Probability": [
            float(value) * 100
            for value in probabilities.values()
        ]
    })

    fig_probability = px.bar(
        probability_data,
        x="Risk Level",
        y="Probability",
        color="Risk Level",
        text_auto=".2f",
        color_discrete_map=risk_colors
    )

    fig_probability.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="Risk Category",
        yaxis_range=[0, 100]
    )

    fig_probability = apply_chart_theme(fig_probability)

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

    # Model comparison
    st.divider()

    st.subheader("Machine Learning Model Comparison")

    comparison_df = get_model_comparison(evaluation_results)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    best_model_name, best_f1 = get_best_model(evaluation_results)

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Performing Model",
        best_model_name
    )

    col2.metric(
        "Best F1 Score",
        f"{best_f1 * 100:.2f}%"
    )

    # Model comparison chart
    comparison_chart = px.bar(
        comparison_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        barmode="group",
        title="Model Performance Comparison"
    )

    comparison_chart = apply_chart_theme(comparison_chart)

    st.plotly_chart(
        comparison_chart,
        use_container_width=True
    )

    # Feature importance
    st.divider()

    st.subheader("Academic Feature Importance")

    importance_data = get_feature_importance(selected_model)

    if importance_data:

        importance_df = pd.DataFrame(
            importance_data.items(),
            columns=["Feature", "Importance"]
        )

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=True
        )

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Factors Influencing Risk Prediction"
        )

        fig_importance = apply_chart_theme(fig_importance)

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

    # Explanation
    st.divider()

    st.subheader("🧠 Prediction Interpretation")

    st.write(
        f"""
        The selected machine learning model is **{selected_model_name}**.

        Based on the student's academic information:

        - **Attendance:** {attendance:.1f}%
        - **Marks:** {marks:.1f}%
        - **Assignments:** {assignments:.1f}%

        The model predicts the student belongs to the
        **{predicted_risk} Risk** category.

        The estimated High/Critical probability is
        **{risk_score:.2f}%**.
        """
    )

    st.info(
        "This system currently uses synthetic training data for demonstration. "
        "Predictions are experimental and should not be used for actual "
        "academic decisions."
    )

