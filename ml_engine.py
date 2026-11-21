
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# CONFIGURATION
# ==========================================================

FEATURE_COLUMNS = ["Attendance", "Marks", "Assignments"]

RISK_LEVELS = ["Low", "Moderate", "High", "Critical"]


# ==========================================================
# SYNTHETIC DATA GENERATOR
# ==========================================================

def generate_training_data(size=1200):

    np.random.seed(42)

    size = max(400, int(size))
    per_class = size // 4

    def create_group(attendance_range, marks_range, assignment_range, label):

        return pd.DataFrame({
            "Attendance": np.random.uniform(
                attendance_range[0], attendance_range[1], per_class
            ),
            "Marks": np.random.uniform(
                marks_range[0], marks_range[1], per_class
            ),
            "Assignments": np.random.uniform(
                assignment_range[0], assignment_range[1], per_class
            ),
            "Risk Level": [label] * per_class
        })

    low = create_group((85, 100), (75, 100), (80, 100), "Low")
    moderate = create_group((65, 89), (50, 74), (55, 79), "Moderate")
    high = create_group((40, 69), (30, 59), (35, 64), "High")
    critical = create_group((15, 44), (10, 39), (15, 44), "Critical")

    data = pd.concat(
        [low, moderate, high, critical],
        ignore_index=True
    )

    return data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare_data(df=None):

    if df is None or "Risk Level" not in df.columns:
        df = generate_training_data(1200)

    df = df.copy()

    required_columns = FEATURE_COLUMNS + ["Risk Level"]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    df["Risk Level"] = (
        df["Risk Level"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df = df[df["Risk Level"].isin(RISK_LEVELS)]

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["Risk Level"].copy()

    # Remove rows with no usable feature values
    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    # Fallback when real data is insufficient
    if len(X) == 0 or y.nunique() < 2:
        fallback = generate_training_data(1200)

        X = fallback[FEATURE_COLUMNS].copy()
        y = fallback["Risk Level"].copy()

    return X, y


# ==========================================================
# SAFE TRAINING / TEST SPLIT
# ==========================================================

def _safe_training_split(X, y):

    counts = y.value_counts()

    valid_classes = counts[counts >= 2].index

    keep = y.isin(valid_classes)

    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    # Fallback for insufficient classes
    if y.nunique() < 2 or y.value_counts().min() < 2:

        fallback = generate_training_data(1200)

        X = fallback[FEATURE_COLUMNS].copy()
        y = fallback["Risk Level"].copy()

    # Safe stratification
    class_counts = y.value_counts()

    if (
        y.nunique() >= 2
        and class_counts.min() >= 2
        and int(len(y) * 0.25) >= y.nunique()
    ):

        return train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )


# ==========================================================
# TRAIN MACHINE LEARNING MODELS
# ==========================================================

def train_models(df=None):

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = _safe_training_split(X, y)

    trained_models = {}
    evaluation_results = {}

    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42
    )

    random_forest.fit(X_train, y_train)

    trained_models["Random Forest"] = random_forest

    # ------------------------------------------------------
    # GRADIENT BOOSTING
    # ------------------------------------------------------

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )

    gradient_boosting.fit(X_train, y_train)

    trained_models["Gradient Boosting"] = gradient_boosting

    # ------------------------------------------------------
    # LOGISTIC REGRESSION
    # ------------------------------------------------------

    logistic_regression = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=42
        ))
    ])

    logistic_regression.fit(X_train, y_train)

    trained_models["Logistic Regression"] = logistic_regression

    # ------------------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------------------

    for name, model in trained_models.items():

        predictions = model.predict(X_test)

        evaluation_results[name] = {
            "Accuracy": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
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

    # Always return exactly 3 items
    return trained_models, evaluation_results, X


# ==========================================================
# STUDENT RISK PREDICTION
# ==========================================================

def predict_student(model, attendance, marks, assignments):

    student = pd.DataFrame([{
        "Attendance": float(attendance),
        "Marks": float(marks),
        "Assignments": float(assignments)
    }])

    predicted_risk = model.predict(student)[0]

    probabilities = {
        risk: 0.0 for risk in RISK_LEVELS
    }

    if hasattr(model, "predict_proba"):

        raw_probabilities = model.predict_proba(student)[0]

        classes = model.classes_

        # Pipeline models expose their classes through the final estimator
        if isinstance(model, Pipeline):
            classes = model.named_steps["classifier"].classes_

        for risk, probability in zip(classes, raw_probabilities):

            probabilities[risk] = round(
                float(probability) * 100, 2
            )

    else:

        probabilities[predicted_risk] = 100.0

    high_risk_probability = round(
        probabilities.get("High", 0.0)
        + probabilities.get("Critical", 0.0),
        2
    )

    # Returns probabilities in percentage format (0-100)
    return predicted_risk, probabilities, high_risk_probability


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
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)


# ==========================================================
# BEST MODEL
# ==========================================================

def get_best_model(trained_models, evaluation_results):

    if not trained_models or not evaluation_results:
        return None, None

    best_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["Accuracy"]
    )

    best_model = trained_models.get(best_name)

    # Returns (model name, actual model object)
    return best_name, best_model


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(model):

    importance = {}

    if model is None:
        return importance

    # Tree-based models
    if hasattr(model, "feature_importances_"):

        values = model.feature_importances_

        for feature, value in zip(FEATURE_COLUMNS, values):
            importance[feature] = float(value)

    # Logistic Regression Pipeline
    elif isinstance(model, Pipeline):

        classifier = model.named_steps.get("classifier")

        if classifier is not None and hasattr(classifier, "coef_"):

            values = np.abs(classifier.coef_).mean(axis=0)

            for feature, value in zip(FEATURE_COLUMNS, values):
                importance[feature] = float(value)

    total = sum(importance.values())

    if total > 0:

        importance = {
            feature: round((value / total) * 100, 2)
            for feature, value in importance.items()
        }

    return importance


# Compatibility alias
def get_feature_importances(model):
    return get_feature_importance(model)


# ==========================================================
# TRAINING DATA SUMMARY
# ==========================================================

def get_training_data_summary(df=None):

    if df is None or "Risk Level" not in df.columns:
        df = generate_training_data(1200)

    return {
        "Total Students": int(len(df)),
        "Risk Distribution": df["Risk Level"].value_counts().to_dict()
    }
