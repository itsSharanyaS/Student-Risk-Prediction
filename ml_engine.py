
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FEATURE_COLUMNS = ["Attendance", "Marks", "Assignments"]

RISK_LEVELS = ["Low", "Moderate", "High", "Critical"]


# =========================================================
# 1. GENERATE TRAINING DATA
# =========================================================

def generate_training_data(size=1200):

    rng = np.random.default_rng(42)

    attendance = rng.uniform(30, 100, size)
    marks = rng.uniform(20, 100, size)
    assignments = rng.uniform(20, 100, size)

    risk_score = (
        (100 - attendance) * 0.40
        + (100 - marks) * 0.40
        + (100 - assignments) * 0.20
    )

    risk = np.where(
        risk_score >= 70, "Critical",
        np.where(
            risk_score >= 55, "High",
            np.where(risk_score >= 35, "Moderate", "Low")
        )
    )

    return pd.DataFrame({
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments,
        "Risk Level": risk
    })


# =========================================================
# 2. PREPARE DATA
# =========================================================

def prepare_data(df=None):

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = generate_training_data()

    required = FEATURE_COLUMNS + ["Risk Level"]

    if not all(col in df.columns for col in required):
        df = generate_training_data()

    df = df[required].copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Risk Level"] = df["Risk Level"].astype(str)

    df = df.dropna(subset=FEATURE_COLUMNS + ["Risk Level"])

    if df.empty or df["Risk Level"].nunique() < 2:
        df = generate_training_data()

    return df


# =========================================================
# 3. TRAIN MODELS
# =========================================================

def train_models(df=None):

    data = prepare_data(df)

    X = data[FEATURE_COLUMNS]
    y = data["Risk Level"]

    # Remove classes with insufficient samples
    counts = y.value_counts()

    valid_classes = counts[counts >= 2].index

    data = data[data["Risk Level"].isin(valid_classes)]

    # If the data is too small, use synthetic data
    if data.empty or data["Risk Level"].nunique() < 2:
        data = generate_training_data()

    X = data[FEATURE_COLUMNS]
    y = data["Risk Level"]

    # Safe stratified split
    test_count = max(int(np.ceil(len(data) * 0.25)), y.nunique())

    if test_count >= len(data):
        data = generate_training_data()
        X = data[FEATURE_COLUMNS]
        y = data["Risk Level"]
        test_count = max(int(np.ceil(len(data) * 0.25)), y.nunique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_count,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            random_state=42
        ),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000))
        ])
    }

    trained_models = {}
    evaluation_results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        trained_models[name] = model

        evaluation_results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        }

    return trained_models, evaluation_results, X


# =========================================================
# 4. STUDENT PREDICTION
# =========================================================

def predict_student(model, attendance, marks, assignments):

    input_data = pd.DataFrame([{
        "Attendance": attendance,
        "Marks": marks,
        "Assignments": assignments
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_dict = {
        str(classes[i]): float(probabilities[i] * 100)
        for i in range(len(classes))
    }

    high_risk_probability = (
        probability_dict.get("High", 0)
        + probability_dict.get("Critical", 0)
    )

    return prediction, probability_dict, high_risk_probability


# =========================================================
# 5. MODEL COMPARISON
# =========================================================

def get_model_comparison(evaluation_results):

    if not evaluation_results:
        return pd.DataFrame()

    df = pd.DataFrame(evaluation_results).T

    return df.reset_index().rename(columns={"index": "Model"})


# =========================================================
# 6. BEST MODEL
# =========================================================

def get_best_model(trained_models, evaluation_results):

    if not trained_models or not evaluation_results:
        return None, None

    best_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["F1 Score"]
    )

    return best_name, trained_models[best_name]


# =========================================================
# 7. FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model):

    if model is None:
        return pd.DataFrame()

    importance = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif isinstance(model, Pipeline):

        classifier = model.named_steps.get("classifier")

        if classifier is not None and hasattr(classifier, "coef_"):
            importance = np.abs(classifier.coef_).mean(axis=0)

    if importance is None:
        return pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": [0, 0, 0]
        })

    importance = np.asarray(importance, dtype=float)

    total = importance.sum()

    if total > 0:
        importance = importance / total * 100

    return pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importance
    }).sort_values("Importance", ascending=False)


# =========================================================
# 8. COMPATIBILITY FUNCTION
# =========================================================

def get_feature_importances(model):
    return get_feature_importance(model)


def calculate_feature_importance(model):
    return get_feature_importance(model)


# =========================================================
# 9. TRAINING DATA SUMMARY
# =========================================================

def get_training_data_summary(df=None):

    data = prepare_data(df)

    return {
        "total_students": len(data),
        "risk_distribution": data["Risk Level"].value_counts().to_dict(),
        "average_attendance": float(data["Attendance"].mean()),
        "average_marks": float(data["Marks"].mean()),
        "average_assignments": float(data["Assignments"].mean())
    }
