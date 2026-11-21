
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# 1. FEATURES
# =========================================================

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


# =========================================================
# 2. GENERATE SYNTHETIC TRAINING DATA
# =========================================================

def generate_training_data(size=1200):

    import numpy as np
    import pandas as pd

    np.random.seed(42)

    per_class = size // 4

    # =========================
    # LOW RISK STUDENTS
    # =========================
    low_attendance = np.random.uniform(80, 100, per_class)
    low_marks = np.random.uniform(70, 100, per_class)
    low_assignments = np.random.uniform(75, 100, per_class)

    # =========================
    # MODERATE RISK STUDENTS
    # =========================
    moderate_attendance = np.random.uniform(65, 85, per_class)
    moderate_marks = np.random.uniform(50, 75, per_class)
    moderate_assignments = np.random.uniform(55, 80, per_class)

    # =========================
    # HIGH RISK STUDENTS
    # =========================
    high_attendance = np.random.uniform(45, 70, per_class)
    high_marks = np.random.uniform(35, 60, per_class)
    high_assignments = np.random.uniform(40, 65, per_class)

    # =========================
    # CRITICAL RISK STUDENTS
    # =========================
    critical_attendance = np.random.uniform(20, 50, per_class)
    critical_marks = np.random.uniform(15, 40, per_class)
    critical_assignments = np.random.uniform(20, 45, per_class)

    # =========================
    # COMBINE DATA
    # =========================
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

    # Shuffle students
    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return data


# =========================================================
# 3. PREPARE DATA
# =========================================================

def prepare_data(df=None):

    if df is None:
        df = generate_training_data()

    data = df.copy()

    required_columns = FEATURE_COLUMNS + ["Risk Level"]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    data["Risk Level"] = data["Risk Level"].astype(str)

    data = data.dropna()

    X = data[FEATURE_COLUMNS].astype(float)

    y = data["Risk Level"]

    return X, y


# =========================================================
# 4. TRAIN MODELS
# =========================================================

def train_models(df=None):

    X, y = prepare_data(df)

    if len(X) < 20:
        raise ValueError(
            "At least 20 records are required for model training."
        )

    if y.nunique() < 2:
        raise ValueError(
            "At least two risk categories are required."
        )

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
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "Logistic Regression": Pipeline([

            ("imputer", SimpleImputer(
                strategy="median"
            )),

            ("scaler", StandardScaler()),

            ("model", LogisticRegression(
                max_iter=1500,
                class_weight="balanced"
            ))

        ])
    }

    trained_models = {}
    evaluation_results = {}

    for model_name, model in models.items():

        try:

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            trained_models[model_name] = model

            evaluation_results[model_name] = {

                "Accuracy": accuracy_score(
                    y_test,
                    predictions
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
                )
            }

        except Exception as e:

            print(
                f"Error training {model_name}: {e}"
            )

    # Return exactly 3 values because risk_prediction.py
    # expects 3 values.
    return trained_models, evaluation_results, X


# =========================================================
# 5. PREDICT A STUDENT
# =========================================================

def predict_student(
    model,
    attendance,
    marks,
    assignments
):

    student_data = pd.DataFrame({

        "Attendance": [float(attendance)],
        "Marks": [float(marks)],
        "Assignments": [float(assignments)]

    })

    predicted_risk = model.predict(
        student_data
    )[0]

    probabilities = {}

    if hasattr(model, "predict_proba"):

        raw_probabilities = model.predict_proba(
            student_data
        )[0]

        model_classes = model.classes_

        for class_name, probability in zip(
            model_classes,
            raw_probabilities
        ):

            probabilities[str(class_name)] = float(
                probability
            )

    else:

        probabilities = {
            str(predicted_risk): 1.0
        }

    # Ensure all risk levels are present
    for risk in RISK_LEVELS:

        probabilities.setdefault(
            risk,
            0.0
        )

    # High/Critical combined probability
    high_risk_probability = (
        probabilities.get("High", 0.0) +
        probabilities.get("Critical", 0.0)
    )

    return (
        str(predicted_risk),
        probabilities,
        high_risk_probability
    )


# =========================================================
# 6. MODEL COMPARISON
# =========================================================

def get_model_comparison(evaluation_results):

    comparison_data = []

    for model_name, metrics in evaluation_results.items():

        comparison_data.append({

            "Model": model_name,

            "Accuracy": round(
                metrics["Accuracy"] * 100, 2
            ),

            "Precision": round(
                metrics["Precision"] * 100, 2
            ),

            "Recall": round(
                metrics["Recall"] * 100, 2
            ),

            "F1 Score": round(
                metrics["F1 Score"] * 100, 2
            )

        })

    comparison_df = pd.DataFrame(
        comparison_data
    )

    if not comparison_df.empty:

        comparison_df = comparison_df.sort_values(
            by="F1 Score",
            ascending=False
        ).reset_index(drop=True)

    return comparison_df


# =========================================================
# 7. BEST MODEL
# =========================================================

def get_best_model(evaluation_results):

    if not evaluation_results:

        return None, 0

    best_model_name = max(
        evaluation_results,
        key=lambda name:
        evaluation_results[name]["F1 Score"]
    )

    best_f1 = evaluation_results[
        best_model_name
    ]["F1 Score"]

    return best_model_name, best_f1


# =========================================================
# 8. FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model):

    importance_values = None

    # Random Forest / Gradient Boosting
    if hasattr(model, "feature_importances_"):

        importance_values = model.feature_importances_

    # Logistic Regression
    elif isinstance(model, Pipeline):

        final_model = model.steps[-1][1]

        if hasattr(final_model, "coef_"):

            importance_values = np.mean(
                np.abs(final_model.coef_),
                axis=0
            )

    if importance_values is None:

        return {}

    importance_data = {}

    for feature, value in zip(
        FEATURE_COLUMNS,
        importance_values
    ):

        importance_data[feature] = float(value)

    # Normalize to percentage
    total = sum(importance_data.values())

    if total > 0:

        importance_data = {

            feature: value / total

            for feature, value in
            importance_data.items()

        }

    return importance_data
