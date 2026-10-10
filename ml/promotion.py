import mlflow


MODEL_NAME = "customer-churn-model"
MIN_F1_SCORE = 0.80


def get_model_f1(client, model_version):
    run = client.get_run(model_version.run_id)

    if "f1_score" not in run.data.metrics:
        raise ValueError(
            f"F1 score not found for run {model_version.run_id}"
        )

    return run.data.metrics["f1_score"]


def promote_challenger():
    client = mlflow.MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion",
    )

    challenger = client.get_model_version_by_alias(
        MODEL_NAME,
        "challenger",
    )

    champion_f1 = get_model_f1(client, champion)
    challenger_f1 = get_model_f1(client, challenger)

    print("=== Model Promotion Evaluation ===")
    print(f"Champion version  : {champion.version}")
    print(f"Champion F1       : {champion_f1:.4f}")
    print(f"Challenger version: {challenger.version}")
    print(f"Challenger F1     : {challenger_f1:.4f}")
    print(f"Minimum F1        : {MIN_F1_SCORE:.4f}")

    print("\n=== Quality Gate ===")

    if challenger_f1 < MIN_F1_SCORE:
        print("Challenger Quality Gate: FAILED")
        print("Decision: KEEP CURRENT CHAMPION")
        return False

    print("Challenger Quality Gate: PASSED")

    print("\n=== Champion vs Challenger ===")

    if challenger_f1 > champion_f1:
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            challenger.version,
        )

        print(
            f"Decision: PROMOTE VERSION {challenger.version}"
        )
        print(
            f"Champion alias moved from version "
            f"{champion.version} to version {challenger.version}"
        )

        return True

    print("Decision: KEEP CURRENT CHAMPION")
    print(
        "Reason: Challenger is not better than the current champion."
    )

    return False


if __name__ == "__main__":
    promote_challenger()
