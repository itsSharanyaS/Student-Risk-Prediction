
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================================================
# 1. CONFIGURATION
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
# 2. GENERATE BALANCED TRAINING DATA
# =========================================================

def generate_training_data(size=1200):

    np.random.seed(42)

    per_class = size // 4

    # -------------------------
    # LOW RISK
    # -------------------------
    low_attendance = np.random.uniform(85, 100, per_class)
    low_marks = np.random.uniform(75, 100, per_class)
    low_assignments = np.random.uniform(80, 100, per_class)

    # -------------------------
    # MODERATE RISK
    # -------------------------
    moderate_attendance = np.random.uniform(65, 90, per_class)
    moderate_marks = np.random.uniform(50, 75, per_class)
    moderate_assignments = np.random.uniform(55, 80, per_class)

    # -------------------------
    # HIGH RISK
    # -------------------------
    high_attendance = np.random.uniform(40, 70, per_class)
    high_marks = np.random.uniform(30, 60, per_class)
    high_assignments = np.random.uniform(35, 65, per_class)

    # -------------------------
    # CRITICAL RISK
    # -------------------------
    critical_attendance = np.random.uniform(15, 45, per_class)
    critical_marks = np.random.uniform(10, 40, per_class)
    critical_assignments = np.random.uniform(15, 45, per_class)

    # -------------------------
    # CREATE DATASET
    # -------------------------
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

    # Shuffle data
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

    df = df.copy()

    required_columns = FEATURE_COLUMNS + ["Risk Level"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove rows with missing target labels
    df = df.dropna(subset=["Risk Level"])

    # Clean risk labels
    df["Risk Level"] = (
        df["Risk Level"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Keep only valid risk levels
    df = df[df["Risk Level"].isin(RISK_LEVELS)]

    # Convert features to numeric
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["Risk Level"].copy()

    if len(X) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if y.nunique() < 2:
        raise ValueError(
            "At least two risk categories are required."
        )

    # Remove rows where every feature is missing
    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    return X, y


# =========================================================
# 4. TRAIN MACHINE LEARNING MODELS
# =========================================================

def train_models(df=None):

    X, y = prepare_data(df)

    # Check category counts
    class_counts = y.value_counts()

    # Remove classes with fewer than 2 samples
    valid_classes = class_counts[class_counts >= 2].index

    valid_rows = y.isin(valid_classes)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if y.nunique() < 2:
        raise ValueError(
            "Training requires at least two risk categories "
            "with two or more samples each."
        )

    # Safe stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    trained_models = {}

    evaluation_results = {}

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )

    random_forest.fit(
        X_train,
        y_train
    )

    trained_models["Random Forest"] = random_forest

    # =====================================================
    # GRADIENT BOOSTING
    # =====================================================

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )

    gradient_boosting.fit(
        X_train,
        y_train
    )

    trained_models["Gradient Boosting"] = gradient_boosting

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    logistic_regression = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    logistic_regression.fit(
        X_train,
        y_train
    )

    trained_models["Logistic Regression"] = logistic_regression

    # =====================================================
    # EVALUATE MODELS
    # =====================================================

    for model_name, model in trained_models.items():

        predictions = model.predict(X_test)

        evaluation_results[model_name] = {

            "Accuracy": round(
                accuracy_score(y_test, predictions) * 100,
                2
            ),

            "Precision": round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            ),

            "Recall": round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            ),

            "F1 Score": round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            )
        }

    # Returns exactly 3 values
    return trained_models, evaluation_results, X


# =========================================================
# 5. PREDICT STUDENT RISK
# =========================================================

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

    probabilities = {}

    # Probability prediction
    if hasattr(model, "predict_proba"):

        raw_probabilities = model.predict_proba(
            student_data
        )[0]

        model_classes = model.classes_

        for risk in RISK_LEVELS:
            probabilities[risk] = 0.0

        for class_name, probability in zip(
            model_classes,
            raw_probabilities
        ):

            probabilities[class_name] = round(
                float(probability) * 100,
                2
            )

    else:

        for risk in RISK_LEVELS:
            probabilities[risk] = 0.0

        probabilities[predicted_risk] = 100.0

    high_risk_probability = (
        probabilities.get("High", 0)
        + probabilities.get("Critical", 0)
    )

    return (
        predicted_risk,
        probabilities,
        high_risk_probability
    )


# =========================================================
# 6. MODEL COMPARISON
# =========================================================

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


# =========================================================
# 7. GET BEST MODEL
# =========================================================

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

    best_model = trained_models.get(
        best_model_name
    )

    return best_model_name, best_model


# =========================================================
# 8. FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model):

    importance = {}

    # Random Forest / Gradient Boosting
    if hasattr(model, "feature_importances_"):

        values = model.feature_importances_

        for feature, value in zip(
            FEATURE_COLUMNS,
            values
        ):
            importance[feature] = float(value)

    # Logistic Regression pipeline
    elif isinstance(model, Pipeline):

        classifier = model.named_steps.get(
            "classifier"
        )

        if classifier is not None and hasattr(
            classifier,
            "coef_"
        ):

            coefficients = np.abs(
                classifier.coef_
            )

            values = coefficients.mean(axis=0)

            for feature, value in zip(
                FEATURE_COLUMNS,
                values
            ):
                importance[feature] = float(value)

    # Normalize values to percentages
    total = sum(importance.values())

    if total > 0:

        importance = {
            feature: round(
                (value / total) * 100,
                2
            )
            for feature, value in importance.items()
        }

    return importance
