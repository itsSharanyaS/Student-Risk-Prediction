
def get_model_comparison(evaluation_results):
    """
    Convert model evaluation results into a comparison table.
    """

    comparison_data = []

    for model_name, metrics in evaluation_results.items():

        comparison_data.append({
            "Model": model_name,
            "Accuracy": round(metrics["Accuracy"] * 100, 2),
            "Precision": round(metrics["Precision"] * 100, 2),
            "Recall": round(metrics["Recall"] * 100, 2),
            "F1 Score": round(metrics["F1 Score"] * 100, 2)
        })

    comparison_df = pd.DataFrame(comparison_data)

    comparison_df = comparison_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    return comparison_df


def get_best_model(evaluation_results):
    """
    Identify the best-performing model based on F1 score.
    """

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["F1 Score"]
    )

    best_f1 = evaluation_results[best_model_name]["F1 Score"]

    return best_model_name, best_f1
