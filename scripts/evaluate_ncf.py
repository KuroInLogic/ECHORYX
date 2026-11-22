
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.evaluation.ncf_evaluation import evaluate_ncf_model


if __name__ == "__main__":

    results = evaluate_ncf_model(
        k=10,
        batch_size=8192,
        max_users=None
    )

    print("\nNEUMF FULL EVALUATION")
    print("=" * 40)

    for metric, value in results.items():
        print(f"{metric}: {value}")
