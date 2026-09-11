"""
Dataset preparation for Sanskrit-English multilingual retrieval.

Responsibilities:
- Load Bhagavad Gita dataset
- Extract Sanskrit and English text
- Clean records
- Create chapter-level train/validation/test splits
- Format text for E5 models
"""

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "Voider22/bhagavad-gita-verses-sanskrit-translations"


def load_gita_dataset(dataset_name=DATASET_NAME):
    """
    Load the Bhagavad Gita dataset from Hugging Face.

    Returns:
        pandas.DataFrame
    """

    dataset = load_dataset(dataset_name)

    df = dataset["train"].to_pandas()

    return df


def extract_english_translation(row):
    """
    Extract English translation from available translation fields.

    The dataset contains translation/commentary dictionaries.
    The 'et' field is used for English translation.
    """

    preferred_columns = [
        "siva",
        "purohit",
        "chinmay",
        "adi",
        "gambir",
        "prabhu"
    ]

    for column in preferred_columns:

        if column not in row:
            continue

        value = row[column]

        if isinstance(value, dict):

            if "et" in value:

                text = value["et"]

                if (
                    isinstance(text, str)
                    and len(text.strip()) > 20
                ):
                    return text.strip()

    return None


def clean_dataset(df):
    """
    Clean and prepare the raw dataset.
    """

    df = df.copy()

    # Extract English translation
    df["english"] = df.apply(
        extract_english_translation,
        axis=1
    )

    # Remove missing values
    df = df[
        df["slok"].notna()
        &
        df["english"].notna()
    ].copy()

    # Convert to strings
    df["slok"] = (
        df["slok"]
        .astype(str)
        .str.strip()
    )

    df["english"] = (
        df["english"]
        .astype(str)
        .str.strip()
    )

    # Remove very short records
    df = df[
        df["slok"].str.len() > 20
    ]

    df = df[
        df["english"].str.len() > 20
    ]

    # Remove duplicate records
    if "_id" in df.columns:
        df = df.drop_duplicates(
            subset=["_id"]
        )

    df = df.reset_index(drop=True)

    return df


def add_e5_prefixes(df):
    """
    Add E5 retrieval prefixes.

    E5 models use:
        query: ...
        passage: ...
    """

    df = df.copy()

    df["query_en"] = (
        "query: "
        + df["english"].str.strip()
    )

    df["passage_sa"] = (
        "passage: "
        + df["slok"].str.strip()
    )

    return df


def chapter_split(
    df,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42
):
    """
    Split data by chapter instead of individual rows.

    This helps reduce leakage between neighboring verses.
    """

    chapters = sorted(
        df["chapter"].unique()
    )

    n_chapters = len(chapters)

    train_end = int(
        n_chapters * train_ratio
    )

    val_end = int(
        n_chapters
        * (train_ratio + val_ratio)
    )

    train_chapters = chapters[
        :train_end
    ]

    val_chapters = chapters[
        train_end:val_end
    ]

    test_chapters = chapters[
        val_end:
    ]

    train_df = df[
        df["chapter"].isin(
            train_chapters
        )
    ].copy()

    val_df = df[
        df["chapter"].isin(
            val_chapters
        )
    ].copy()

    test_df = df[
        df["chapter"].isin(
            test_chapters
        )
    ].copy()

    return (
        train_df,
        val_df,
        test_df
    )


def prepare_data():
    """
    Complete data preparation pipeline.

    Returns:
        train_df
        val_df
        test_df
    """

    print("Loading dataset...")

    df = load_gita_dataset()

    print(
        "Raw dataset size:",
        len(df)
    )

    print("Cleaning dataset...")

    df = clean_dataset(df)

    print(
        "Clean dataset size:",
        len(df)
    )

    print("Adding E5 prefixes...")

    df = add_e5_prefixes(df)

    print("Creating chapter-level split...")

    train_df, val_df, test_df = chapter_split(
        df
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    return (
        train_df,
        val_df,
        test_df
    )


if __name__ == "__main__":

    train_df, val_df, test_df = (
        prepare_data()
    )

    print("\nSample training record:")

    print(
        train_df[
            [
                "query_en",
                "passage_sa"
            ]
        ].head(1)
    )