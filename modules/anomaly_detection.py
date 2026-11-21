import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def render_anomaly_detection(students):

    st.title("🚨 AI Anomaly Detection")
    st.caption(
        "Detect unusual academic patterns using unsupervised "
        "machine learning and Isolation Forest."
    )

    st.divider()

    # --------------------------------------------------
    # 1. VALIDATE DATA
    # --------------------------------------------------

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
            "Missing columns: " + ", ".join(missing_columns)
        )
        return

    df = students.copy()

    features = [
        "Attendance",
        "Marks",
        "Assignments"
    ]

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=features).copy()

    if len(df) < 5:
        st.warning(
            "At least 5 students with valid academic data "
            "are recommended for anomaly detection."
        )
        return

    # --------------------------------------------------
    # 2. CONFIGURATION
    # --------------------------------------------------

    st.subheader("⚙️ Detection Settings")

    contamination = st.slider(
        "Expected proportion of unusual students",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.05,
        help=(
            "Controls how many records the algorithm may identify "
            "as unusual. This is an estimate, not a confirmed "
            "percentage of students with problems."
        )
    )

    # --------------------------------------------------
    # 3. PREPARE FEATURES
    # --------------------------------------------------

    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 4. ISOLATION FOREST
    # --------------------------------------------------

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    decision_scores = model.decision_function(X_scaled)

    df["Anomaly Status"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    # Higher score means more unusual.
    df["Anomaly Score"] = -decision_scores

    # Relative rank, 0-100
    df["Unusualness Percentile"] = (
        df["Anomaly Score"].rank(pct=True) * 100
    ).round(1)

    # --------------------------------------------------
    # 5. OVERVIEW
    # --------------------------------------------------

    st.subheader("📊 Anomaly Overview")

    anomaly_count = int(
        (df["Anomaly Status"] == "Anomaly").sum()
    )

    normal_count = int(
        (df["Anomaly Status"] == "Normal").sum()
    )

    anomaly_percentage = (
        anomaly_count / len(df) * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Students Analyzed", len(df))
    col2.metric("Unusual Records", anomaly_count)
    col3.metric("Normal Records", normal_count)
    col4.metric("Unusual Records (%)", f"{anomaly_percentage:.1f}%")

    st.info(
        "An anomaly is an unusual data pattern, not proof that "
        "a student is failing or has a problem. Review the records "
        "before taking any action."
    )

    st.divider()

    # --------------------------------------------------
    # 6. ANOMALY DISTRIBUTION
    # --------------------------------------------------

    st.subheader("📌 Normal vs Unusual Records")

    status_counts = (
        df["Anomaly Status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Students"
    ]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Students",
        hole=0.45,
        color="Status",
        color_discrete_map={
            "Normal": "#16A34A",
            "Anomaly": "#DC2626"
        }
    )

    fig.update_layout(height=400)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # 7. ANOMALY SCATTER PLOT
    # --------------------------------------------------

    st.subheader("📈 Academic Pattern Detection")

    fig = px.scatter(
        df,
        x="Attendance",
        y="Marks",
        color="Anomaly Status",
        size="Assignments",
        hover_name="Student",
        hover_data=[
            "Student ID",
            "Attendance",
            "Marks",
            "Assignments",
            "Anomaly Score",
            "Unusualness Percentile"
        ],
        color_discrete_map={
            "Normal": "#16A34A",
            "Anomaly": "#DC2626"
        },
        title="Unusual Students vs Normal Student Patterns"
    )

    fig.update_layout(
        height=550,
        xaxis_title="Attendance (%)",
        yaxis_title="Marks (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # 8. FEATURE COMPARISON
    # --------------------------------------------------

    st.subheader("📊 Academic Feature Comparison")

    feature_choice = st.selectbox(
        "Select academic parameter",
        features
    )

    fig = px.box(
        df,
        x="Anomaly Status",
        y=feature_choice,
        color="Anomaly Status",
        color_discrete_map={
            "Normal": "#16A34A",
            "Anomaly": "#DC2626"
        },
        points="all",
        title=f"{feature_choice} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # 9. ANOMALOUS STUDENTS
    # --------------------------------------------------

    st.subheader("🚩 Unusual Student Records")

    anomalous_students = df[
        df["Anomaly Status"] == "Anomaly"
    ].sort_values(
        "Anomaly Score",
        ascending=False
    )

    if not anomalous_students.empty:

        st.dataframe(
            anomalous_students[
                [
                    "Student ID",
                    "Student",
                    "Attendance",
                    "Marks",
                    "Assignments",
                    "Anomaly Score",
                    "Unusualness Percentile"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No unusual records were detected in this analysis."
        )

    st.divider()

    # --------------------------------------------------
    # 10. STUDENT INVESTIGATION
    # --------------------------------------------------

    st.subheader("🔎 Student Investigation")

    selected_student = st.selectbox(
        "Select student",
        df["Student ID"].astype(str).tolist(),
        key="anomaly_student_search"
    )

    student_row = df[
        df["Student ID"].astype(str) == selected_student
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Attendance",
        f"{student_row['Attendance']:.1f}%"
    )

    col2.metric(
        "Marks",
        f"{student_row['Marks']:.1f}%"
    )

    col3.metric(
        "Assignments",
        f"{student_row['Assignments']:.1f}%"
    )

    col4.metric(
        "Detection",
        student_row["Anomaly Status"]
    )

    st.metric(
        "Unusualness Percentile",
        f"{student_row['Unusualness Percentile']:.1f}%"
    )

    if student_row["Anomaly Status"] == "Anomaly":

        st.warning(
            "The machine learning model detected an unusual combination "
            "of academic data for this student. Review the individual "
            "parameters before drawing conclusions."
        )

    else:

        st.success(
            "This student's academic data falls within the patterns "
            "identified as normal by the current model."
        )

    # --------------------------------------------------
    # 11. PARAMETER REVIEW
    # --------------------------------------------------

    st.subheader("📋 Individual Parameter Review")

    student_features = pd.DataFrame({
        "Parameter": features,
        "Student Value": [
            student_row["Attendance"],
            student_row["Marks"],
            student_row["Assignments"]
        ],
        "Class Average": [
            df["Attendance"].mean(),
            df["Marks"].mean(),
            df["Assignments"].mean()
        ]
    })

    student_features["Difference from Class Average"] = (
        student_features["Student Value"]
        - student_features["Class Average"]
    ).round(2)

    st.dataframe(
        student_features.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # 12. DOWNLOAD REPORT
    # --------------------------------------------------

    st.subheader("📥 Export Anomaly Report")

    report = df[
        [
            "Student ID",
            "Student",
            "Attendance",
            "Marks",
            "Assignments",
            "Anomaly Status",
            "Anomaly Score",
            "Unusualness Percentile"
        ]
    ].copy()

    st.download_button(
        "Download Anomaly Detection Report",
        data=report.to_csv(index=False),
        file_name="student_anomaly_report.csv",
        mime="text/csv",
        use_container_width=True
    )
