# Sanskrit-English Multilingual Retrieval

A domain-adapted multilingual semantic retrieval system for Sanskrit-English retrieval using contrastive fine-tuning of `intfloat/multilingual-e5-small`.

## Problem

The objective is to retrieve semantically relevant Sanskrit passages when given an English query, and support the reverse direction as well.

Example:

English query:

"What does the Bhagavad Gita say about karma?"

The system retrieves relevant Sanskrit verses.

## Architecture

English/Sanskrit Query
        |
        v
Multilingual E5 Encoder
        |
        v
384-dimensional embedding
        |
        v
Cosine Similarity
        |
        v
Top-K Retrieval
        |
        v
Relevant Sanskrit/English passages

## Workflow
```
multilingual-retrieval-ft/
│
├── README.md
├── REPORT.md
├── requirements.txt
│
├── notebooks/
│   └── Sanskrit_English_Multilingual_Retrieval.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── retrieval.py
│
├── results/
│   ├── retrieval_metrics.csv
│   ├── base_vs_finetuned.csv
│   ├── failure_analysis.csv
│   └── sample_outputs.csv
│
├── models/
│   └── README.md
│
└── data/
    └── README.md
```
### important
data.py       → dataset loading + cleaning + splitting
train.py      → model + training
evaluate.py   → Recall@K + MRR + nDCG
retrieval.py  → semantic search / Top-K retrieval

## How to run it

From the project root:

```
pip install -r requirements.txt
```

Then:

Test data preparation, Train & Retrieval

```
python -m src.data
python -m src.train
python -m src.retrieval
```

## Model

Base model:

intfloat/multilingual-e5-small

The model is fine-tuned using Multiple Negatives Ranking Loss.

## Dataset

Bhagavad Gita Sanskrit-English aligned dataset.

The dataset contains Sanskrit verses, transliteration and English translations.

## Training

Training pairs:

(query, positive passage)

The training uses in-batch negatives through Multiple Negatives Ranking Loss.

## Evaluation

Metrics:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR
- nDCG@5
- nDCG@10

## Experiments

1. Base model
2. Fine-tuned model
3. Cross-lingual retrieval
4. Transliteration analysis
5. Hard-negative analysis
6. Failure analysis

## Compute

Experiments were designed for Google Colab T4/L4 GPU.

## Limitations

- Dataset is relatively small.
- Exact one-to-one retrieval evaluation can be overly strict.
- A query may have multiple semantically valid passages.
- Sanskrit morphology can create lexical variation.
- Single-verse retrieval may miss context spanning multiple verses.
- Translation differences can affect semantic alignment.

## Future Work

- Larger Sanskrit-English parallel corpus
- Better hard-negative mining
- Chapter/context-aware chunking
- Hybrid BM25 + dense retrieval
- Cross-encoder reranking
- RAG generation
- Human evaluation