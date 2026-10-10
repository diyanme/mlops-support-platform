import sys


MIN_F1_SCORE = 0.80


def evaluate_model(f1_score: float) -> bool:
    print("=== Model Quality Gate ===")
    print(f"F1 Score      : {f1_score:.4f}")
    print(f"Minimum F1    : {MIN_F1_SCORE:.4f}")

    if f1_score >= MIN_F1_SCORE:
        print("Quality Gate  : PASSED")
        return True

    print("Quality Gate  : FAILED")
    return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ml/evaluate.py <f1_score>")
        sys.exit(1)

    try:
        f1_score = float(sys.argv[1])
    except ValueError:
        print("F1 score must be a number.")
        sys.exit(1)

    if not evaluate_model(f1_score):
        sys.exit(1)
