"""
Fine-tune multilingual-e5-small for Sanskrit-English retrieval.

Training strategy:
- Positive English-Sanskrit pairs
- Multiple Negatives Ranking Loss
- In-batch negatives
- SentenceTransformers
"""

import os
from .data import prepare_data
from src.data import prepare_data

from torch.utils.data import DataLoader

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)

from .data import prepare_data


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-small"

OUTPUT_DIR = (
    "./models/e5-sanskrit-english"
)

BATCH_SIZE = 16

EPOCHS = 3

SEED = 42


# ============================================================
# CREATE TRAINING EXAMPLES
# ============================================================

def create_training_examples(df):
    """
    Convert dataframe rows into SentenceTransformer examples.
    """

    examples = []

    for _, row in df.iterrows():

        query = row["query_en"]

        passage = row["passage_sa"]

        examples.append(
            InputExample(
                texts=[
                    query,
                    passage
                ]
            )
        )

    return examples


# ============================================================
# CREATE VALIDATION EVALUATOR
# ============================================================

def create_validation_evaluator(
    val_df
):
    """
    Create InformationRetrievalEvaluator.
    """

    queries = {
        str(i): query
        for i, query in enumerate(
            val_df["query_en"]
        )
    }

    corpus = {
        str(i): passage
        for i, passage in enumerate(
            val_df["passage_sa"]
        )
    }

    relevant_docs = {
        str(i): {str(i)}
        for i in range(
            len(val_df)
        )
    }

    evaluator = (
        evaluation
        .InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="validation"
        )
    )

    return evaluator


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("=" * 70)
    print("SANSKRIT-ENGLISH EMBEDDING FINE-TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, val_df, test_df = (
        prepare_data()
    )

    # --------------------------------------------------------
    # Create examples
    # --------------------------------------------------------

    train_examples = (
        create_training_examples(
            train_df
        )
    )

    print(
        "\nTraining examples:",
        len(train_examples)
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Number of batches:",
        len(train_dataloader)
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading model:",
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    train_loss = (
        losses
        .MultipleNegativesRankingLoss(
            model=model
        )
    )

    print(
        "\nLoss:",
        train_loss.__class__.__name__
    )

    # --------------------------------------------------------
    # Validation evaluator
    # --------------------------------------------------------

    evaluator = (
        create_validation_evaluator(
            val_df
        )
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    warmup_steps = int(
        len(train_dataloader)
        * EPOCHS
        * 0.1
    )

    print(
        "Warmup steps:",
        warmup_steps
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Fine-tune
    # --------------------------------------------------------

    print("\nStarting training...")

    model.fit(
        train_objectives=[
            (
                train_dataloader,
                train_loss
            )
        ],

        evaluator=evaluator,

        epochs=EPOCHS,

        warmup_steps=warmup_steps,

        output_path=OUTPUT_DIR,

        show_progress_bar=True,

        use_amp=True
    )

    print(
        "\nTraining completed."
    )

    print(
        "Model saved at:",
        OUTPUT_DIR
    )

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()