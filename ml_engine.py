
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================================
# CONFIGURATION
# ==========================================================

FEATURE_COLUMNS = [
    "Attendance",
    "Marks",
    "Assignments"
]

RISK_LEVELS = [
    "Low",
    "Moderate",
    "High",
    "Critical"
]


# ==========================================================
# GENERATE BALANCED TRAINING DATA
# ==========================================================

def generate_training_data(size=1200):

    np.random.seed(42)

    per_class = size // 4

    # Low risk
    low_attendance = np.random.uniform(85, 100, per_class)
    low_marks = np.random.uniform(75, 100, per_class)
    low_assignments = np.random.uniform(80, 100, per_class)

    # Moderate risk
    moderate_attendance = np.random.uniform(65, 90, per_class)
    moderate_marks = np.random.uniform(50, 75, per_class)
    moderate_assignments = np.random.uniform(55, 80, per_class)

    # High risk
    high_attendance = np.random.uniform(40, 70, per_class)
    high_marks = np.random.uniform(30, 60, per_class)
    high_assignments = np.random.uniform(35, 65, per_class)

    # Critical risk
    critical_attendance = np.random.uniform(15, 45, per_class)
    critical_marks = np.random.uniform(10, 40, per_class)
    critical_assignments = np.random.uniform(15, 45, per_class)

    data = pd.DataFrame({

        "Attendance": np.concatenate([
            low_attendance,
            moderate_attendance,
            high_attendance,
            critical_attendance
        ]),

        "Marks": np.concatenate([
            low_marks,
            moderate_marks,
            high_marks,
            critical_marks
        ]),

        "Assignments": np.concatenate([
            low_assignments,
            moderate_assignments,
            high_assignments,
            critical_assignments
        ]),

        "Risk Level": (
            ["Low"] * per_class +
            ["Moderate"] * per_class +
            ["High"] * per_class +
            ["Critical"] * per_class
        )
    })

    return data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_data(df=None):

    if df is None:
        df = generate_training_data()

    df = df.copy()

    required_columns = FEATURE_COLUMNS + ["Risk Level"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.dropna(subset=["Risk Level"])

    df["Risk Level"] = (
        df["Risk Level"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df = df[
        df["Risk Level"].isin(RISK_LEVELS)
    ]

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["Risk Level"].copy()

    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(X) == 0:
        raise ValueError("Training data is empty.")

    if y.nunique() < 2:
        raise ValueError(
            "At least two risk categories are required."
        )

    return X, y


# ==========================================================
# TRAIN MODELS
# ==========================================================

def train_models(df=None):

    X, y = prepare_data(df)

    # Ensure each category has enough samples
    counts = y.value_counts()

    valid_classes = counts[counts >= 2].index

    valid_rows = y.isin(valid_classes)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if y.nunique() < 2:
        raise ValueError(
            "Not enough risk categories to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    trained_models = {}
    evaluation_results = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)
    trained_models["Random Forest"] = rf

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )

    gb.fit(X_train, y_train)
    trained_models["Gradient Boosting"] = gb

    # Logistic Regression
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lr.fit(X_train, y_train)
    trained_models["Logistic Regression"] = lr

    # Evaluate models
    for name, model in trained_models.items():

        predictions = model.predict(X_test)

        evaluation_results[name] = {
            "Accuracy": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "Precision": round(
                precision_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "Recall": round(
                recall_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "F1 Score": round(
                f1_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            )
        }

    # Return exactly 3 values
    return trained_models, evaluation_results, X


# ==========================================================
# PREDICT STUDENT RISK
# ==========================================================

def predict_student(
    model,
    attendance,
    marks,
    assignments
):

    student_data = pd.DataFrame([{
        "Attendance": float(attendance),
        "Marks": float(marks),
        "Assignments": float(assignments)
    }])

    predicted_risk = model.predict(
        student_data
    )[0]

    probabilities = {
        risk: 0.0 for risk in RISK_LEVELS
    }

    if hasattr(model, "predict_proba"):

        raw_probabilities = model.predict_proba(
            student_data
        )[0]

        model_classes = model.classes_

        for class_name, probability in zip(
            model_classes,
            raw_probabilities
        ):
            probabilities[class_name] = round(
                float(probability) * 100, 2
            )

    else:
        probabilities[predicted_risk] = 100.0

    high_risk_probability = (
        probabilities["High"] +
        probabilities["Critical"]
    )

    return (
        predicted_risk,
        probabilities,
        high_risk_probability
    )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

def get_model_comparison(evaluation_results):

    if not evaluation_results:
        return pd.DataFrame()

    comparison = pd.DataFrame(
        evaluation_results
    ).T

    comparison.index.name = "Model"

    comparison = comparison.reset_index()

    return comparison.sort_values(
        "Accuracy",
        ascending=False
    ).reset_index(drop=True)


# ==========================================================
# BEST MODEL
# ==========================================================

def get_best_model(
    trained_models,
    evaluation_results
):

    if not trained_models or not evaluation_results:
        return None, None

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["Accuracy"]
    )

    return (
        best_model_name,
        trained_models.get(best_model_name)
    )


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(model):

    importance = {}

    # Tree models
    if hasattr(model, "feature_importances_"):

        values = model.feature_importances_

        for feature, value in zip(
            FEATURE_COLUMNS, values
        ):
            importance[feature] = float(value)

    # Logistic Regression
    elif isinstance(model, Pipeline):

        classifier = model.named_steps.get("classifier")

        if classifier is not None and hasattr(classifier, "coef_"):

            values = np.abs(
                classifier.coef_
            ).mean(axis=0)

            for feature, value in zip(
                FEATURE_COLUMNS, values
            ):
                importance[feature] = float(value)

    # Normalize
    total = sum(importance.values())

    if total > 0:

        importance = {
            feature: round(
                (value / total) * 100, 2
            )
            for feature, value in importance.items()
        }

    return importance
