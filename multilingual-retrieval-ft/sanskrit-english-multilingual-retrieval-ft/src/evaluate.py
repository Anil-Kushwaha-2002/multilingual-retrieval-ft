"""
Evaluation utilities for multilingual semantic retrieval.

Metrics:
- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG@K

This script evaluates the fine-tuned E5 model on the
held-out test split.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import prepare_data


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    query_embeddings,
    passage_embeddings,
    k
):
    """
    Calculate Recall@K.

    Assumption:
        query i corresponds to passage i.
    """

    similarity_matrix = np.matmul(
        query_embeddings,
        passage_embeddings.T
    )

    top_k_indices = np.argsort(
        -similarity_matrix,
        axis=1
    )[:, :k]

    correct = 0

    for i in range(len(query_embeddings)):

        if i in top_k_indices[i]:
            correct += 1

    return correct / len(query_embeddings)


# ============================================================
# MRR
# ============================================================

def mean_reciprocal_rank(
    query_embeddings,
    passage_embeddings
):
    """
    Calculate Mean Reciprocal Rank.
    """

    similarity_matrix = np.matmul(
        query_embeddings,
        passage_embeddings.T
    )

    rankings = np.argsort(
        -similarity_matrix,
        axis=1
    )

    reciprocal_ranks = []

    for i in range(len(query_embeddings)):

        positions = np.where(
            rankings[i] == i
        )[0]

        if len(positions) > 0:

            rank = positions[0] + 1

            reciprocal_ranks.append(
                1.0 / rank
            )

        else:

            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(
    query_embeddings,
    passage_embeddings,
    k
):
    """
    Calculate nDCG@K.

    Each query has one known relevant passage.
    """

    similarity_matrix = np.matmul(
        query_embeddings,
        passage_embeddings.T
    )

    rankings = np.argsort(
        -similarity_matrix,
        axis=1
    )[:, :k]

    ndcg_scores = []

    for i in range(len(query_embeddings)):

        dcg = 0.0

        for position, index in enumerate(rankings[i]):

            if index == i:

                dcg = 1.0 / np.log2(position + 2)
                break

        # With one relevant document, ideal DCG = 1.
        ideal_dcg = 1.0

        ndcg_scores.append(
            dcg / ideal_dcg
        )

    return np.mean(ndcg_scores)


# ============================================================
# COMPLETE EVALUATION
# ============================================================

def evaluate_retrieval(
    query_embeddings,
    passage_embeddings
):
    """
    Calculate all retrieval metrics.
    """

    results = {

        "Recall@1": recall_at_k(
            query_embeddings,
            passage_embeddings,
            1
        ),

        "Recall@3": recall_at_k(
            query_embeddings,
            passage_embeddings,
            3
        ),

        "Recall@5": recall_at_k(
            query_embeddings,
            passage_embeddings,
            5
        ),

        "Recall@10": recall_at_k(
            query_embeddings,
            passage_embeddings,
            10
        ),

        "MRR": mean_reciprocal_rank(
            query_embeddings,
            passage_embeddings
        ),

        "nDCG@5": ndcg_at_k(
            query_embeddings,
            passage_embeddings,
            5
        ),

        "nDCG@10": ndcg_at_k(
            query_embeddings,
            passage_embeddings,
            10
        )
    }

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\nRetrieval Evaluation")
    print("=" * 50)

    for metric, score in results.items():

        print(
            f"{metric:<15}: {score:.4f}"
        )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("SANSKRIT-ENGLISH RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_dataset, validation_dataset, test_dataset = prepare_data()

    print(
        f"Test examples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Load fine-tuned model
    # --------------------------------------------------------

    model_path = "./models/e5-sanskrit-english"

    print(
        f"\nLoading model: {model_path}"
    )

    model = SentenceTransformer(model_path)

    # --------------------------------------------------------
    # Extract queries and passages
    # --------------------------------------------------------

    queries = test_dataset["query_en"]
    passages = test_dataset["passage_sa"]

    print(
        "\nEncoding test queries..."
    )

    query_embeddings = model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(
        "\nEncoding test passages..."
    )

    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\nCalculating retrieval metrics..."
    )

    results = evaluate_retrieval(
        query_embeddings,
        passage_embeddings
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(results)

    print(
        "\nEvaluation completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

