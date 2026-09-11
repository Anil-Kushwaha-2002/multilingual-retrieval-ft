# INTERACTIVE RETRIEVAL
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from .data import prepare_data

"""
Semantic retrieval for Sanskrit-English passages.

Provides:
- Query embedding
- Cosine similarity
- Top-K retrieval
- English -> Sanskrit search
- Sanskrit -> English search
- RAG retrieval
- Interactive CLI
"""

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from .data import prepare_data


# ============================================================
# EMBEDDING
# ============================================================

def encode_texts(
    model,
    texts,
    batch_size=32
):
    """
    Encode text into normalized embeddings.
    """

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings


# ============================================================
# TOP-K RETRIEVAL
# ============================================================

def retrieve(
    query_embedding,
    passage_embeddings,
    top_k=5
):
    """
    Retrieve top-K passages using cosine similarity.

    Because embeddings are normalized,
    dot product = cosine similarity.
    """

    scores = np.dot(
        passage_embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        -scores
    )[:top_k]

    return (
        top_indices,
        scores[top_indices]
    )


# ============================================================
# ENGLISH -> SANSKRIT
# ============================================================

def search_sanskrit(
    query,
    model,
    passage_embeddings,
    dataframe,
    top_k=5
):
    """
    English query -> Sanskrit passages.
    """

    query_text = (
        "query: "
        + query.strip()
    )

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    indices, scores = retrieve(
        query_embedding,
        passage_embeddings,
        top_k=top_k
    )

    results = []

    for rank, (index, score) in enumerate(
        zip(indices, scores),
        start=1
    ):

        row = dataframe.iloc[index]

        results.append({

            "rank": rank,

            "score": float(score),

            "query": row["query_en"],

            "sanskrit": row["passage_sa"]

        })

    return results


# ============================================================
# SANSKRIT -> ENGLISH
# ============================================================

def search_english(
    sanskrit_query,
    model,
    english_embeddings,
    dataframe,
    top_k=5
):
    """
    Sanskrit query -> English passages.

    This function expects an English passage
    embedding matrix.
    """

    query_text = (
        "query: "
        + sanskrit_query.strip()
    )

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    indices, scores = retrieve(
        query_embedding,
        english_embeddings,
        top_k=top_k
    )

    results = []

    for rank, (index, score) in enumerate(
        zip(indices, scores),
        start=1
    ):

        row = dataframe.iloc[index]

        results.append({

            "rank": rank,

            "score": float(score),

            "query": row["query_en"],

            "sanskrit": row["passage_sa"]

        })

    return results


# ============================================================
# RAG RETRIEVER
# ============================================================

def rag_retrieve(
    question,
    model,
    passage_embeddings,
    dataframe,
    top_k=3
):
    """
    Retrieve context for a RAG pipeline.
    """

    results = search_sanskrit(
        question,
        model,
        passage_embeddings,
        dataframe,
        top_k=top_k
    )

    context = []

    for result in results:

        context.append({

            "rank": result["rank"],

            "score": result["score"],

            "sanskrit": result["sanskrit"],

            "query": result["query"]

        })

    return context


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SANSKRIT-ENGLISH SEMANTIC RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_df, validation_df, test_df = prepare_data()

    print(
        f"Train passages      : {len(train_df)}"
    )

    print(
        f"Validation passages : {len(validation_df)}"
    )

    print(
        f"Test passages       : {len(test_df)}"
    )

    # Combine datasets for interactive search
    dataframe = pd.concat(
        [
            train_df,
            validation_df,
            test_df
        ],
        ignore_index=True
    )

    print(
        f"Total passages      : {len(dataframe)}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "query_en",
        "passage_sa"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
            + "\nAvailable columns: "
            + str(list(dataframe.columns))
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_path = (
        "./models/e5-sanskrit-english"
    )

    print(
        f"\nLoading model: {model_path}"
    )

    model = SentenceTransformer(
        model_path
    )

    # --------------------------------------------------------
    # Prepare Sanskrit passages
    # --------------------------------------------------------

    passages = [
        "passage: " + str(text).strip()
        for text in dataframe["passage_sa"]
    ]

    # --------------------------------------------------------
    # Encode passages
    # --------------------------------------------------------

    print(
        "\nCreating Sanskrit passage embeddings..."
    )

    passage_embeddings = encode_texts(
        model,
        passages,
        batch_size=32
    )

    print(
        f"\nEmbedding shape: "
        f"{passage_embeddings.shape}"
    )

    print("\nModel ready.")

    # --------------------------------------------------------
    # Interactive search
    # --------------------------------------------------------

    while True:

        print("\n" + "=" * 70)

        query = input(
            "\nEnter English question "
            "(or type 'exit' to quit): "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nExiting retrieval."
            )

            break

        if not query:

            print(
                "Please enter a question."
            )

            continue

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        results = search_sanskrit(
            query=query,
            model=model,
            passage_embeddings=passage_embeddings,
            dataframe=dataframe,
            top_k=5
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            "\nTop 5 Retrieved Sanskrit Passages"
        )

        print("=" * 70)

        for result in results:

            print(
                f"\nRank       : "
                f"{result['rank']}"
            )

            print(
                f"Similarity : "
                f"{result['score']:.4f}"
            )

            print(
                "\nOriginal English Query:"
            )

            print(
                result["query"]
            )

            print(
                "\nSanskrit Passage:"
            )

            print(
                result["sanskrit"]
            )

            print(
                "-" * 70
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()