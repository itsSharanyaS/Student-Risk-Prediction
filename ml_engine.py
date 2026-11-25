
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


FEATURE_COLUMNS = ["Attendance", "Marks", "Assignments"]
RISK_LEVELS = ["Low", "Moderate", "High", "Critical"]


def generate_training_data(size=1500):
    np.random.seed(42)

    attendance = np.random.randint(30, 101, size)
    marks = np.random.randint(20, 101, size)
    assignments = np.random.randint(20, 101, size)

    score = (
        (100 - attendance) * 0.40
        + (100 - marks) * 0.40
        + (100 - assignments) * 0.20
    )

    risk = np.where(
        score >= 70, "Critical",
        np.where(score >= 55, "High",
        np.where(score >= 35, "Moderate", "Low"))
    )

    return pd.DataFrame({
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments,
        "Risk": risk
    })


def prepare_data(df=None):
    if df is None or df.empty:
        df = generate_training_data()

    df = df.copy()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.random.randint(40, 90, len(df))

    if "Risk" not in df.columns:
        score = (
            (100 - df["Attendance"]) * 0.40
            + (100 - df["Marks"]) * 0.40
            + (100 - df["Assignments"]) * 0.20
        )

        df["Risk"] = np.where(
            score >= 70, "Critical",
            np.where(score >= 55, "High",
            np.where(score >= 35, "Moderate", "Low"))
        )

    df = df.dropna(subset=["Risk"])

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS)

    return df


def train_models(df=None):
    data = prepare_data(df)

    X = data[FEATURE_COLUMNS]
    y = data["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ])
    }

    trained_models = {}
    evaluation_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        trained_models[name] = model

        evaluation_results[name] = {
            "Accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "Precision": round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "Recall": round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "F1 Score": round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            )
        }

    return trained_models, evaluation_results, X


def predict_student(model, attendance, marks, assignments):
    student_data = pd.DataFrame([{
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments
    }])

    prediction = model.predict(student_data)[0]

    probabilities = model.predict_proba(student_data)[0]

    probability_dict = {
        str(class_name): round(float(probability) * 100, 2)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    high_risk_probability = sum(
        probability_dict.get(level, 0)
        for level in ["High", "Critical"]
    )

    return prediction, probability_dict, round(high_risk_probability, 2)


def get_model_comparison(evaluation_results):
    return pd.DataFrame([
        {
            "Model": model_name,
            **metrics
        }
        for model_name, metrics in evaluation_results.items()
    ])


def get_best_model(trained_models, evaluation_results):
    best_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["F1 Score"]
    )

    return best_name, trained_models[best_name]


def get_feature_importance(model):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif hasattr(model, "named_steps"):
        classifier = model.named_steps["model"]

        if hasattr(classifier, "coef_"):
            importance = np.mean(
                np.abs(classifier.coef_),
                axis=0
            )
        else:
            importance = np.zeros(len(FEATURE_COLUMNS))

    else:
        importance = np.zeros(len(FEATURE_COLUMNS))

    importance = np.array(importance, dtype=float)

    total = importance.sum()

    if total > 0:
        importance = importance / total * 100

    return pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": np.round(importance, 2)
    }).sort_values("Importance", ascending=False)


def get_training_data_summary():
    df = generate_training_data()

    return {
        "total_samples": len(df),
        "risk_distribution": df["Risk"].value_counts().to_dict(),
        "features": FEATURE_COLUMNS
    }
