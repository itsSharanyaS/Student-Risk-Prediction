
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

RISK_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}

FEATURES = [
    "Attendance",
    "Marks",
    "Assignments"
]


def generate_training_data(n_samples=2500, random_state=42):
    """
    Generate synthetic academic data for demonstration.
    This is not real student data.
    """

    rng = np.random.default_rng(random_state)

    attendance = rng.uniform(35, 100, n_samples)
    marks = rng.uniform(20, 100, n_samples)
    assignments = rng.uniform(20, 100, n_samples)

    # Generate a synthetic academic risk score.
    risk_score = (
        (100 - attendance) * 0.40
        + (100 - marks) * 0.40
        + (100 - assignments) * 0.20
    )

    # Convert risk scores into four classes.
    risk = np.zeros(n_samples, dtype=int)

    risk[risk_score >= 25] = 1
    risk[risk_score >= 40] = 2
    risk[risk_score >= 55] = 3

    # Add small label variations for a more realistic
    # experimental training dataset.
    noise_indices = rng.choice(
        n_samples,
        size=int(n_samples * 0.06),
        replace=False
    )

    risk[noise_indices] = rng.integers(
        0, 4, size=len(noise_indices)
    )

    data = pd.DataFrame({
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments,
        "Risk": risk
    })

    return data


def train_models():
    """
    Train three ML models and evaluate them.
    """

    data = generate_training_data()

    X = data[FEATURES]
    y = data["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ])
    }

    trained_models = {}
    evaluation_results = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        trained_models[model_name] = model

        evaluation_results[model_name] = {
            "Accuracy": accuracy_score(
                y_test, predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

            "Confusion Matrix": confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1, 2, 3]
            ),

            "Predictions": predictions,

            "Actual": y_test.to_numpy()
        }

    return trained_models, evaluation_results, data


def predict_student(model, attendance, marks, assignments):
    """
    Predict the risk category for an individual student.
    """

    student_data = pd.DataFrame([{
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments
    }])

    prediction = model.predict(student_data)[0]

    probabilities = model.predict_proba(student_data)[0]

    class_probabilities = {}

    for class_index, probability in zip(
        model.classes_, probabilities
    ):
        class_probabilities[
            RISK_LABELS[int(class_index)]
        ] = float(probability)

    risk_probability = (
        class_probabilities.get("High", 0)
        + class_probabilities.get("Critical", 0)
    )

    return {
        "Risk": RISK_LABELS[int(prediction)],
        "Risk Probability": float(risk_probability),
        "Class Probabilities": class_probabilities
    }


def get_feature_importance(model):
    """
    Return feature importance for tree-based models.
    """

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

    if hasattr(model, "named_steps"):
        logistic_model = model.named_steps["model"]

        if hasattr(logistic_model, "coef_"):
            importance = np.mean(
                np.abs(logistic_model.coef_),
                axis=0
            )

            return pd.DataFrame({
                "Feature": FEATURES,
                "Importance": importance
            }).sort_values(
                "Importance",
                ascending=False
            )

    return pd.DataFrame({
        "Feature": FEATURES,
        "Importance": [0, 0, 0]
    })

