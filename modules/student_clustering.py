import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def render_student_clustering(students):

    st.title("🧠 Student Clustering")
    st.caption(
        "Machine learning-based grouping of students "
        "according to their academic performance patterns."
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
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        return

    df = students.copy()

    for col in ["Attendance", "Marks", "Assignments"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(
        subset=["Attendance", "Marks", "Assignments"]
    ).copy()

    if len(df) < 2:
        st.warning(
            "At least two students with valid academic data "
            "are required for clustering."
        )
        return

    # --------------------------------------------------
    # 2. SELECT FEATURES
    # --------------------------------------------------

    features = [
        "Attendance",
        "Marks",
        "Assignments"
    ]

    X = df[features]

    # Standardize features so each feature has equal influence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 3. CLUSTER SETTINGS
    # --------------------------------------------------

    st.subheader("⚙️ Clustering Configuration")

    maximum_clusters = min(5, len(df))

    selected_clusters = st.slider(
        "Number of Student Groups",
        min_value=2,
        max_value=maximum_clusters,
        value=min(3, maximum_clusters),
        help="Choose how many groups the machine learning model should create."
    )

    # --------------------------------------------------
    # 4. APPLY K-MEANS
    # --------------------------------------------------

    model = KMeans(
        n_clusters=selected_clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X_scaled)

    # --------------------------------------------------
    # 5. GENERATE CLUSTER PROFILES
    # --------------------------------------------------

    cluster_profiles = (
        df.groupby("Cluster")[features]
        .mean()
        .round(2)
        .reset_index()
    )

    # Sort clusters based on average marks
    cluster_profiles = cluster_profiles.sort_values(
        "Marks"
    ).reset_index(drop=True)

    cluster_names = {}

    for position, row in cluster_profiles.iterrows():

        if position == 0:
            name = "Needs Improvement"

        elif position == len(cluster_profiles) - 1:
            name = "High Performers"

        else:
            name = f"Group {position + 1}"

        cluster_names[int(row["Cluster"])] = name

    df["Cluster Name"] = df["Cluster"].map(
        cluster_names
    )

    cluster_profiles["Cluster Name"] = (
        cluster_profiles["Cluster"].map(cluster_names)
    )

    # --------------------------------------------------
    # 6. OVERVIEW METRICS
    # --------------------------------------------------

    st.subheader("📊 Clustering Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Students Analyzed",
        len(df)
    )

    col2.metric(
        "Groups Identified",
        selected_clusters
    )

    col3.metric(
        "Average Attendance",
        f"{df['Attendance'].mean():.1f}%"
    )

    col4.metric(
        "Average Marks",
        f"{df['Marks'].mean():.1f}%"
    )

    st.divider()

    # --------------------------------------------------
    # 7. CLUSTER DISTRIBUTION
    # --------------------------------------------------

    st.subheader("👥 Student Group Distribution")

    cluster_counts = (
        df["Cluster Name"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster Name",
        "Students"
    ]

    fig = px.pie(
        cluster_counts,
        names="Cluster Name",
        values="Students",
        hole=0.45
    )

    fig.update_layout(height=420)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # 8. CLUSTER VISUALIZATION
    # --------------------------------------------------

    st.subheader("📈 Student Performance Clusters")

    fig = px.scatter(
        df,
        x="Attendance",
        y="Marks",
        color="Cluster Name",
        size="Assignments",
        hover_name="Student",
        hover_data=[
            "Student ID",
            "Attendance",
            "Marks",
            "Assignments"
        ],
        title="Student Groups Based on Attendance and Marks"
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
    # 9. CLUSTER PROFILES
    # --------------------------------------------------

    st.subheader("📋 Group Academic Profiles")

    st.dataframe(
        cluster_profiles[
            [
                "Cluster Name",
                "Attendance",
                "Marks",
                "Assignments"
            ]
        ].rename(
            columns={
                "Attendance": "Average Attendance",
                "Marks": "Average Marks",
                "Assignments": "Average Assignments"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # 10. CLUSTER INSIGHTS
    # --------------------------------------------------

    st.subheader("💡 Machine Learning Insights")

    for cluster_id, group in df.groupby("Cluster"):

        group_name = cluster_names[int(cluster_id)]

        avg_attendance = group["Attendance"].mean()
        avg_marks = group["Marks"].mean()
        avg_assignments = group["Assignments"].mean()

        st.markdown(f"### {group_name}")

        st.write(
            f"**Students in group:** {len(group)}"
        )

        st.write(
            f"**Average attendance:** {avg_attendance:.1f}% "
            f"· **Average marks:** {avg_marks:.1f}% "
            f"· **Average assignments:** {avg_assignments:.1f}%"
        )

        if avg_attendance < 60:
            st.warning(
                "This group has low average attendance. "
                "Attendance improvement may be a priority."
            )

        if avg_marks < 50:
            st.warning(
                "This group has relatively low academic performance "
                "and may benefit from additional academic support."
            )

        if avg_marks >= 75 and avg_attendance >= 80:
            st.success(
                "This group demonstrates strong academic performance "
                "and attendance."
            )

    st.divider()

    # --------------------------------------------------
    # 11. STUDENT SEARCH
    # --------------------------------------------------

    st.subheader("🔎 Find Student Group")

    student_options = df["Student ID"].astype(str).tolist()

    selected_student = st.selectbox(
        "Select Student ID",
        student_options,
        key="clustering_student_search"
    )

    student_row = df[
        df["Student ID"].astype(str) == selected_student
    ].iloc[0]

    st.markdown(
        f"### {student_row['Student']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Student Group",
        student_row["Cluster Name"]
    )

    col2.metric(
        "Attendance",
        f"{student_row['Attendance']:.1f}%"
    )

    col3.metric(
        "Marks",
        f"{student_row['Marks']:.1f}%"
    )

    col4.metric(
        "Assignments",
        f"{student_row['Assignments']:.1f}%"
    )

    # --------------------------------------------------
    # 12. STUDENT RECORDS
    # --------------------------------------------------

    st.subheader("📑 Cluster Student Records")

    st.dataframe(
        df[
            [
                "Student ID",
                "Student",
                "Attendance",
                "Marks",
                "Assignments",
                "Cluster Name"
            ]
        ].sort_values(
            ["Cluster Name", "Marks"],
            ascending=[True, False]
        ),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 13. DOWNLOAD REPORT
    # --------------------------------------------------

    st.download_button(
        "📥 Download Clustering Report",
        data=df.to_csv(index=False),
        file_name="student_clustering_report.csv",
        mime="text/csv",
        use_container_width=True
    )
