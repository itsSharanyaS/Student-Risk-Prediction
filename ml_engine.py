
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


# =========================================================
# 1. GENERATE TRAINING DATA
# =========================================================

def generate_training_data(size=1500, random_state=42):

    np.random.seed(random_state)

    attendance = np.random.uniform(35, 100, size)
    marks = np.random.uniform(20, 100, size)
    assignments = np.random.uniform(20, 100, size)

    risk_score = (
        (100 - attendance) * 0.40
        + (100 - marks) * 0.40
        + (100 - assignments) * 0.20
    )

    risk = []

    for score in risk_score:
        if score >= 70:
            risk.append("Critical")
        elif score >= 55:
            risk.append("High")
        elif score >= 35:
            risk.append("Moderate")
        else:
            risk.append("Low")

    df = pd.DataFrame({
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments,
        "Risk Level": risk
    })

    return df


# =========================================================
# 2. PREPARE DATA
# =========================================================

def prepare_data(df=None):

    if df is None or len(df) == 0:
        df = generate_training_data()

    df = df.copy()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(
        df[FEATURE_COLUMNS].median()
    )

    if "Risk Level" not in df.columns:
        risk_score = (
            (100 - df["Attendance"]) * 0.40
            + (100 - df["Marks"]) * 0.40
            + (100 - df["Assignments"]) * 0.20
        )

        df["Risk Level"] = risk_score.apply(
            lambda x: (
                "Critical" if x >= 70 else
                "High" if x >= 55 else
                "Moderate" if x >= 35 else
                "Low"
            )
        )

    return df


# =========================================================
# 3. MODEL CREATION
# =========================================================

def train_models(df=None):

    df = prepare_data(df)

    X = df[FEATURE_COLUMNS]
    y = df["Risk Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        ),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ])
    }

    trained_models = {}
    evaluation_results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        trained_models[name] = model

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

    return trained_models, evaluation_results, X


# =========================================================
# 4. MODEL COMPARISON
# =========================================================

def get_model_comparison(evaluation_results):

    comparison = pd.DataFrame(evaluation_results).T

    comparison.index.name = "Model"

    comparison = comparison.reset_index()

    comparison = comparison.sort_values(
        by="F1 Score",
        ascending=False
    )

    return comparison


# =========================================================
# 5. BEST MODEL
# =========================================================

def get_best_model(trained_models, evaluation_results):

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["F1 Score"]
    )

    best_model = trained_models[best_model_name]

    return best_model_name, best_model


# =========================================================
# 6. STUDENT PREDICTION
# =========================================================

def predict_student(model, attendance, marks, assignments):

    input_data = pd.DataFrame([{
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_dict = {}

    for i, class_name in enumerate(model.classes_):
        probability_dict[class_name] = round(
            probabilities[i] * 100, 2
        )

    high_risk_probability = (
        probability_dict.get("High", 0)
        + probability_dict.get("Critical", 0)
    )

    return prediction, probability_dict, round(
        high_risk_probability, 2
    )


# =========================================================
# 7. FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model):

    actual_model = model

    if isinstance(model, Pipeline):
        actual_model = model.named_steps["model"]

    if hasattr(actual_model, "feature_importances_"):

        importance = actual_model.feature_importances_

    elif hasattr(actual_model, "coef_"):

        importance = np.mean(
            np.abs(actual_model.coef_),
            axis=0
        )

    else:

        importance = np.zeros(len(FEATURE_COLUMNS))

    total = importance.sum()

    if total > 0:
        importance = (importance / total) * 100

    result = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importance
    })

    result = result.sort_values(
        by="Importance",
        ascending=False
    )

    result["Importance"] = result["Importance"].round(2)

    return result


# Compatibility aliases
get_feature_importances = get_feature_importance
calculate_feature_importance = get_feature_importance


# =========================================================
# 8. TRAINING DATA SUMMARY
# =========================================================

def get_training_data_summary(df=None):

    df = prepare_data(df)

    return {
        "Total Records": len(df),
        "Average Attendance": round(df["Attendance"].mean(), 2),
        "Average Marks": round(df["Marks"].mean(), 2),
        "Average Assignments": round(df["Assignments"].mean(), 2),
        "Risk Distribution": df["Risk Level"].value_counts().to_dict()
    }
