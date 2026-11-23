
from services.evaluation.hybrid_evaluation import evaluate_hybrid_model

experiments = [
    {
        "name": "CF Heavy",
        "weights": {
            "user_cf": 0.60,
            "ncf": 0.25,
            "popularity": 0.15
        }
    },
    {
        "name": "NCF Heavy",
        "weights": {
            "user_cf": 0.25,
            "ncf": 0.60,
            "popularity": 0.15
        }
    },
    {
        "name": "Popularity Heavy",
        "weights": {
            "user_cf": 0.30,
            "ncf": 0.30,
            "popularity": 0.40
        }
    },
    {
        "name": "Equal Weights",
        "weights": {
            "user_cf": 1/3,
            "ncf": 1/3,
            "popularity": 1/3
        }
    },
    {
        "name": "Current Hybrid",
        "weights": {
            "user_cf": 0.40,
            "ncf": 0.40,
            "popularity": 0.20
        }
    }
]

results = []

for experiment in experiments:
    print("\n" + "=" * 50)
    print("EXPERIMENT:", experiment["name"])

    metrics = evaluate_hybrid_model(
        k=10,
        max_users=None,
        weights=experiment["weights"]
    )

    results.append({
        "Model": experiment["name"],
        "Weights": str(experiment["weights"]),
        "Precision@10": metrics["precision_at_k"],
        "Recall@10": metrics["recall_at_k"],
        "NDCG@10": metrics["ndcg_at_k"]
    })

print("\nFINAL WEIGHT COMPARISON")
print("=" * 80)

import pandas as pd

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))

comparison.to_csv(
    "hybrid_weight_results.csv",
    index=False
)
