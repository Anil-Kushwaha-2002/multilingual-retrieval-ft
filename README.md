# multilingual-retrieval-ft
Fine-tuned multilingual embedding model for Sanskrit-English semantic retrieval using contrastive learning, hard-negative mining, and retrieval evaluation.

# Sanskrit-English Multilingual Retrieval

## 1. Executive Summary

This project implements a multilingual semantic retrieval system for Sanskrit-English retrieval. The system fine-tunes `intfloat/multilingual-e5-small` on aligned Sanskrit-English Bhagavad Gita verse pairs using contrastive learning.

The objective is not to train a multilingual embedding model from scratch, but to adapt an existing multilingual retrieval model to a Sanskrit-English domain while keeping the computation practical for a Google Colab T4/L4 environment.

The final system supports:

* English → Sanskrit retrieval
* Sanskrit → English retrieval
* Transliteration analysis
* Top-K semantic retrieval
* Retrieval metrics including Recall@K, MRR and nDCG
* Failure analysis
* A mini RAG-style retrieval pipeline

## 2. Problem Definition

Given an English semantic query, the system should retrieve the most relevant Sanskrit passage.

Example:
Query:
"What does the Bhagavad Gita say about karma?"

Expected behavior:

Retrieve Sanskrit verses discussing karma, action, duty and related concepts.

The reverse direction is also evaluated:

Sanskrit verse → relevant English translation.

## 3. Model Selection

I selected `intfloat/multilingual-e5-small`.

The main reasons were:

1. Multilingual representation capability.
2. Retrieval-oriented pretraining.
3. Small computational footprint.
4. 384-dimensional embeddings.
5. Sentence-Transformers compatibility.
6. Practical suitability for T4/L4 GPU environments.

Instead of training from scratch, domain adaptation was performed using contrastive fine-tuning.

## 4. Dataset

The experiment uses a Bhagavad Gita Sanskrit-English aligned dataset containing approximately 700 verses.

Each record contains:

* verse ID
* chapter
* verse number
* Sanskrit text
* transliteration
* English translations/commentaries

The dataset was converted into aligned query-passage pairs.

Example:
Query:
English translation
Positive passage:
Corresponding Sanskrit verse

## 5. Data Preparation

The preprocessing pipeline performs:

1. Missing-value removal.
2. Duplicate removal.
3. Sanskrit text normalization.
4. English translation extraction.
5. Removal of extremely short records.
6. E5 query/passage prefix formatting.
7. Chapter-level train/validation/test splitting.

The chapter-level split was chosen to reduce data leakage and make the test set more representative of retrieval on unseen chapters.

## 6. Training Strategy

The model was fine-tuned using Multiple Negatives Ranking Loss.
For each training batch:

Query 1 → Positive Passage 1
Query 2 → Positive Passage 2
Query 3 → Positive Passage 3

The positive passage associated with each query is treated as the positive target. Other passages in the batch act as negatives.
This approach provides multiple negative comparisons without requiring a separate manually constructed negative for every training example.

## 7. Negative Sampling

The first experiment uses in-batch negatives.

A second experiment explores hard-negative mining by using the base embedding model to identify highly similar but incorrect passages.

Hard negatives are useful because random negatives are often too easy.

For example:

Query:

"What does the Gita say about karma?"

Positive:

A verse primarily describing karma.

Hard negative:

Another verse discussing duty or action but not representing the intended semantic target.

## 8. Evaluation

The following metrics were used:

* Recall@1
* Recall@3
* Recall@5
* Recall@10
* Mean Reciprocal Rank
* nDCG@5
* nDCG@10

Recall@K measures whether the correct aligned passage appears in the top K results.

MRR measures how highly the correct passage is ranked.

nDCG measures ranking quality while giving more importance to higher-ranked results.

## 9. Baseline vs Fine-Tuned Model

The base model was evaluated before fine-tuning.

The fine-tuned model was evaluated using exactly the same test set.

| Metric    | Base Model | Fine-Tuned Model |
| --------- | ---------: | ---------------: |
| Recall@1  |   [INSERT] |         [INSERT] |
| Recall@3  |   [INSERT] |         [INSERT] |
| Recall@5  |   [INSERT] |         [INSERT] |
| Recall@10 |   [INSERT] |         [INSERT] |
| MRR       |   [INSERT] |         [INSERT] |
| nDCG@5    |   [INSERT] |         [INSERT] |
| nDCG@10   |   [INSERT] |         [INSERT] |

The final values are generated directly by the evaluation notebook rather than manually estimated.

## 10. Cross-Lingual Alignment

A key objective was to evaluate whether the model can align English and Sanskrit representations.
The system uses English queries and Sanskrit passages in the primary retrieval experiment.
This tests whether semantically equivalent content in two different languages is mapped into nearby embedding regions.
The experiment also evaluates Sanskrit script versus IAST transliteration.

## 11. Transliteration Mismatch

Sanskrit can appear in multiple forms:

* Devanagari
* IAST transliteration
* Romanized text without diacritics

These forms may produce different tokenization patterns.

The experiment therefore includes transliteration analysis to determine whether the model maintains semantic similarity across script representations.

## 12. Retrieval Pipeline

The final retrieval pipeline is:
``
User Query
↓
E5 query prefix
↓
Multilingual embedding model
↓
384-dimensional normalized vector
↓
Cosine similarity against indexed passages
↓
Top-K ranking
↓
Relevant Sanskrit passages
``
This design can directly serve as the retrieval component of a RAG system.

## 13. Mini RAG Design

A minimal RAG architecture is:
``
Question
↓
Embedding Retriever
↓
Top-K Sanskrit passages
↓
Context construction
↓
LLM
↓
Final answer
``
The current project focuses primarily on the retrieval component because retrieval quality is the central objective.

## 14. Failure Analysis

Several potential failure modes were investigated.

### Semantic overlap

Multiple verses can discuss related concepts such as karma, dharma, duty and action.
Therefore, the model may retrieve a semantically related but not exactly aligned verse.

### Translation variation

Different English translations can express the same Sanskrit meaning using substantially different wording.

### Context dependency

Some verses cannot be interpreted accurately without neighboring verses.
A one-verse retrieval system therefore has a structural limitation.

### Sanskrit morphology

Sanskrit is morphologically rich, so the same semantic concept can appear in different inflected forms.

### Transliteration mismatch

IAST and Devanagari representations can create tokenization differences.

## 15. Practical Tradeoffs

### Why not train from scratch?

The dataset is too small to train a high-quality multilingual embedding model from scratch.
Using a pretrained multilingual encoder allows the experiment to focus on domain adaptation.

### Why a small model?

The assignment targets T4/L4 GPU environments and a 1–2 day development window.
A smaller model provides faster iteration and easier deployment.

### Why contrastive learning?

The dataset naturally provides aligned positive pairs.
Contrastive learning directly optimizes the representation space for retrieval.

### Why chapter-level split?

A random split can produce overly optimistic results because neighboring verses may contain highly similar content.
Chapter-level splitting provides a stronger test of generalization.

## 16. Limitations

The current implementation has several limitations:

1. The dataset is small.
2. Exact one-to-one retrieval is stricter than real semantic relevance.
3. Some queries may have multiple correct passages.
4. Context spanning multiple verses is not explicitly modeled.
5. Hard-negative mining is relatively simple.
6. No human relevance judgments were collected.
7. The experiment focuses on Bhagavad Gita rather than broad Sanskrit literature.

## 17. Future Improvements

Possible improvements include:

* Larger Sanskrit-English parallel corpora.
* AI4Bharat/BPCC data.
* Upanishad data.
* More diverse Sanskrit sources.
* Better hard-negative mining.
* Query generation.
* Multi-positive training.
* Chapter-aware chunking.
* Hybrid BM25 + dense retrieval.
* Cross-encoder reranking.
* Human evaluation.
* RAG generation with a local LLM.
* Quantized inference.

## 18. Final Conclusion

The experiment demonstrates that a pretrained multilingual embedding model can be adapted to a specialized Sanskrit-English retrieval task using a relatively small aligned dataset and practical contrastive fine-tuning.

The most important result is not only the final retrieval score, but the complete engineering workflow:
``
dataset preparation → baseline evaluation → contrastive fine-tuning → retrieval evaluation → error analysis → deployment-oriented retrieval demo.
``
The project prioritizes reproducibility, efficient use of compute, rigorous evaluation and analysis of multilingual retrieval failure modes.
