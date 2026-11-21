
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# 1. PREPARE DATA
# =========================================================

def prepare_data(df):

    data = df.copy()

    target_column = "Risk Level"

    if target_column not in data.columns:
        raise ValueError("Risk Level column not found in dataset.")

    data = data.dropna(subset=[target_column])

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove ID columns if present
    id_columns = [
        col for col in X.columns
        if "id" in col.lower()
    ]

    X = X.drop(columns=id_columns, errors="ignore")

    # Encode categorical features
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False
    )

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


# =========================================================
# 2. TRAIN MODELS
# =========================================================

def train_models(df):

    X, y, label_encoder = prepare_data(df)

    if len(X) < 10:
        raise ValueError("At least 10 student records are required.")

    if len(np.unique(y)) < 2:
        raise ValueError("At least two Risk Level classes are required.")

    class_counts = pd.Series(y).value_counts()

    stratify_option = (
        y if class_counts.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_option
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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
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
                )
            }

        except Exception as e:

            print(f"Model error - {model_name}: {e}")

    return trained_models, evaluation_results, X.columns.tolist(), label_encoder


# =========================================================
# 3. PREDICT STUDENT
# =========================================================

def predict_student(
    student_data,
    model,
    feature_columns,
    label_encoder
):

    student_df = pd.DataFrame([student_data])

    # Remove ID columns
    id_columns = [
        col for col in student_df.columns
        if "id" in col.lower()
    ]

    student_df = student_df.drop(
        columns=id_columns,
        errors="ignore"
    )

    # Encode categorical values
    categorical_columns = student_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    student_df = pd.get_dummies(
        student_df,
        columns=categorical_columns,
        drop_first=False
    )

    # Align input with training features
    student_df = student_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(student_df)

    predicted_label = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]

    return predicted_label


# =========================================================
# 4. MODEL COMPARISON
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

    comparison_df = pd.DataFrame(comparison_data)

    if not comparison_df.empty:

        comparison_df = comparison_df.sort_values(
            by="F1 Score",
            ascending=False
        ).reset_index(drop=True)

    return comparison_df


# =========================================================
# 5. BEST MODEL
# =========================================================

def get_best_model(evaluation_results):

    if not evaluation_results:
        return None, 0

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["F1 Score"]
    )

    best_f1 = evaluation_results[best_model_name]["F1 Score"]

    return best_model_name, best_f1
