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
| Recall@1  |    0.8462  |         0.9923   |
| Recall@3  |    0.9308  |         0.9923   |
| Recall@5  |    0.9538  |         0.9923   |
| Recall@10 |    0.9615  |         0.9923   |
| MRR       |    0.8948  |         0.9928   |
| nDCG@5    |    0.9075  |         0.9923   |
| nDCG@10   |    0.9097  |         0.9923   |

The final values are generated directly by the evaluation notebook.

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
```
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
```
This design can directly serve as the retrieval component of a RAG system.

## 13. Mini RAG Design

A minimal RAG architecture is:

```
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
```

The current project focuses primarily on the retrieval component because retrieval quality is the central objective of the assignment.

## 14. Failure Analysis

Several potential failure modes were investigated.

```
1. Semantic overlap
2. Translation ambiguity
3. Very short verse
4. Sanskrit morphology
5. Transliteration mismatch
6. Similar philosophical concepts
7. Multi-verse context required
8. English translation variation
9. Long passage truncation
10. Insufficient training examples
```
```
| Failure Type          | Example                   | Explanation                             |
| --------------------- | ------------------------- | --------------------------------------- |
| Semantic overlap      | Karma vs duty             | Both concepts appear in multiple verses |
| Context dependency    | Multi-verse discussion    | Single verse insufficient               |
| Translation variation | Different English wording | Same meaning expressed differently      |
| Sanskrit morphology   | Inflected forms           | Surface forms differ                    |
| Transliteration       | IAST vs Devanagari        | Different token patterns                |
```

### 1. Semantic overlap

* Multiple verses can discuss related concepts such as karma, dharma, duty and action.
* Therefore, the model may retrieve a semantically related but not exactly aligned verse.

### 2. Translation variation

* Different English translations can express the same Sanskrit meaning using substantially different wording.

### 3. Context dependency

* Some verses cannot be interpreted accurately without neighboring verses.
* A one-verse retrieval system therefore has a structural limitation.

### 4. Sanskrit morphology

* Sanskrit is morphologically rich, so the same semantic concept can appear in different inflected forms.

### 5. Transliteration mismatch

* IAST and Devanagari representations can create tokenization differences.

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

dataset preparation → baseline evaluation → contrastive fine-tuning → retrieval evaluation → error analysis → deployment-oriented retrieval demo.

The project prioritizes reproducibility, efficient use of compute, rigorous evaluation and analysis of multilingual retrieval failure modes.


# Result - 
## 1. Test English Queries
====================================================================================================
QUERY: What does the Bhagavad Gita say about karma?
====================================================================================================
   rank     score       id  chapter  verse  \
0     1  0.106059  BG18.78       18     78   
1     2  0.099859  BG17.21       17     21   
2     3  0.099556  BG18.50       18     50   

                                            sanskrit  \
0  यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...   
1  यत्तु प्रत्युपकारार्थं फलमुद्दिश्य वा पुनः |\n...   
2  सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...   

                                             english  
0  18.78 Wherever is Krishna, the Lord of Yoga; w...  
1  17.21 And, that gift which is given with a vie...  
2  18.50 Learn from Me in brief, O Arjuna, how he...  
rank	score	id	chapter	verse	sanskrit
0	1	0.106059	BG18.78	18	78	यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...
1	2	0.099859	BG17.21	17	21	यत्तु प्रत्युपकारार्थं फलमुद्दिश्य वा पुनः |\n...
2	3	0.099556	BG18.50	18	50	सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...


====================================================================================================
QUERY: What does Krishna say about controlling the mind?
====================================================================================================
   rank     score       id  chapter  verse  \
0     1  0.175252  BG18.78       18     78   
1     2  0.147535   BG17.1       17      1   
2     3  0.144721  BG18.75       18     75   

                                            sanskrit  \
0  यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...   
1  अर्जुन उवाच |\nये शास्त्रविधिमुत्सृज्य यजन्ते ...   
2  व्यासप्रसादाच्छ्रुतवानेतद्गुह्यमहं परम् |\nयोग...   

                                             english  
0  18.78 Wherever is Krishna, the Lord of Yoga; w...  
1  17.1 Arjuna said  Those who, setting aside the...  
2  18.75 Through the grace of Vyasa I have heard ...  
rank	score	id	chapter	verse	sanskrit
0	1	0.175252	BG18.78	18	78	यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...
1	2	0.147535	BG17.1	17	1	अर्जुन उवाच |\nये शास्त्रविधिमुत्सृज्य यजन्ते ...
2	3	0.144721	BG18.75	18	75	व्यासप्रसादाच्छ्रुतवानेतद्गुह्यमहं परम् |\nयोग...


====================================================================================================
QUERY: What is the meaning of performing one's duty?
====================================================================================================
   rank     score       id  chapter  verse  \
0     1  0.217283  BG18.60       18     60   
1     2  0.192890  BG18.18       18     18   
2     3  0.183018  BG18.45       18     45   

                                            sanskrit  \
0  स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...   
1  ज्ञानं ज्ञेयं परिज्ञाता त्रिविधा कर्मचोदना |\n...   
2  स्वे स्वे कर्मण्यभिरतः संसिद्धिं लभते नरः |\nस...   

                                             english  
0  18.60 O Arjuna, bound by thy own Karma (action...  
1  18.18 Knowledge, the knowable and the knower f...  
2  5.10 He who does actions, offering them to Bra...  
rank	score	id	chapter	verse	sanskrit
0	1	0.217283	BG18.60	18	60	स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...
1	2	0.192890	BG18.18	18	18	ज्ञानं ज्ञेयं परिज्ञाता त्रिविधा कर्मचोदना |\n...
2	3	0.183018	BG18.45	18	45	स्वे स्वे कर्मण्यभिरतः संसिद्धिं लभते नरः |\nस...


====================================================================================================
QUERY: What does the Gita say about meditation?
====================================================================================================
   rank     score       id  chapter  verse  \
0     1  0.114925  BG18.33       18     33   
1     2  0.088002  BG18.60       18     60   
2     3  0.085845  BG18.50       18     50   

                                            sanskrit  \
0  धृत्या यया धारयते मनःप्राणेन्द्रियक्रियाः |\nय...   
1  स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...   
2  सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...   

                                             english  
0  18.33 The unwavering firmness by which, throug...  
1  18.60 O Arjuna, bound by thy own Karma (action...  
2  18.50 Learn from Me in brief, O Arjuna, how he...  
rank	score	id	chapter	verse	sanskrit
0	1	0.114925	BG18.33	18	33	धृत्या यया धारयते मनःप्राणेन्द्रियक्रियाः |\nय...
1	2	0.088002	BG18.60	18	60	स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...
2	3	0.085845	BG18.50	18	50	सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...


====================================================================================================
QUERY: What does Krishna say about the soul?
====================================================================================================
   rank     score       id  chapter  verse  \
0     1  0.198116  BG18.78       18     78   
1     2  0.175024  BG18.75       18     75   
2     3  0.159698   BG17.1       17      1   

                                            sanskrit  \
0  यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...   
1  व्यासप्रसादाच्छ्रुतवानेतद्गुह्यमहं परम् |\nयोग...   
2  अर्जुन उवाच |\nये शास्त्रविधिमुत्सृज्य यजन्ते ...   

                                             english  
0  18.78 Wherever is Krishna, the Lord of Yoga; w...  
1  18.75 Through the grace of Vyasa I have heard ...  
2  17.1 Arjuna said  Those who, setting aside the...  
rank	score	id	chapter	verse	sanskrit
0	1	0.198116	BG18.78	18	78	यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...
1	2	0.175024	BG18.75	18	75	व्यासप्रसादाच्छ्रुतवानेतद्गुह्यमहं परम् |\nयोग...
2	3	0.159698	BG17.1	17	1	अर्जुन उवाच |\nये शास्त्रविधिमुत्सृज्य यजन्ते ..


# 2. SANSKRIT → ENGLISH DEMO

SANSKRIT QUERY:
श्रीभगवानुवाच |
अभयं सत्त्वसंशुद्धिर्ज्ञानयोगव्यवस्थितिः |
दानं दमश्च यज्ञश्च स्वाध्यायस्तप आर्जवम् ||१६-१||
rank	score	id	chapter	verse	english
0	1	0.651015	BG16.1	16	1	16.1 The Blessed Lord said Fearlessness, puri...
1	2	0.474475	BG16.17	16	17	16.17 Self-conceited, stubborn, filled with th...
2	3	0.472985	BG16.16	16	16	16.16 Bewildered by many a fancy, entangled in...
3	4	0.458365	BG16.6	16	6	16.6 There are two types of beings in this wor...
4	5	0.448011	BG18.61	18	61	18.61 The Lord dwells in the hearts of all bei..


# 3. INSPECT HARD NEGATIVES

QUERY:
1.1 Dhritarashtra said  What did my people and the sons of Pandu do when they had assembled
together eager for battle on the holy plain of Kurukshetra, O Sanjaya.

POSITIVE:
धृतराष्ट्र उवाच |
धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः |
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ||१-१||

HARD NEGATIVE:
धृष्टकेतुश्चेकितानः काशिराजश्च वीर्यवान् |
पुरुजित्कुन्तिभोजश्च शैब्यश्च नरपुंगवः ||१-५||
====================================================================================================
QUERY:
1.2. Sanjaya said  Having seen the army of the Pandavas drawn up in battle-array,
King Duryodhana then approached his teacher (Drona) and spoke these words.

POSITIVE:
सञ्जय उवाच |
दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा |
आचार्यमुपसंगम्य राजा वचनमब्रवीत् ||१-२||

HARD NEGATIVE:
सञ्जय उवाच |
तं तथा कृपयाविष्टमश्रुपूर्णाकुलेक्षणम् |
विषीदन्तमिदं वाक्यमुवाच मधुसूदनः ||२-१||


# 4. RAG Demo

QUESTION:
What does the Bhagavad Gita say about karma?

RETRIEVED CONTEXT:
====================================================================================================
Rank: 1
Score: 0.1061
Verse: 18.78

SANSKRIT:
यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |
तत्र श्रीर्विजयो भूतिर्ध्रुवा नीतिर्मतिर्मम ||१८-७८||

ENGLISH:
18.78 Wherever is Krishna, the Lord of Yoga; wherever is Arjuna, the wielder of the bow; there are prosperity, victory, happiness and firm policy; such is my conviction.
====================================================================================================
Rank: 2
Score: 0.0999
Verse: 17.21

SANSKRIT:
यत्तु प्रत्युपकारार्थं फलमुद्दिश्य वा पुनः |
दीयते च परिक्लिष्टं तद्दानं राजसं स्मृतम् ||१७-२१||

ENGLISH:
17.21 And, that gift which is given with a view to receive something in return, or looking for a reward, or reluctantly, is held to be Rajasic.


# 5. SAMPLE OUTPUTS 
   rank     score       id  chapter  verse  \
0     1  0.106059  BG18.78       18     78
1     2  0.099859  BG17.21       17     21
2     3  0.099556  BG18.50       18     50   
3     4  0.074624  BG18.42       18     42   
4     5  0.071543  BG17.20       17     20   

                                            sanskrit  \
0  यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...   
1  यत्तु प्रत्युपकारार्थं फलमुद्दिश्य वा पुनः |\n...   
2  सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...   
3  शमो दमस्तपः शौचं क्षान्तिरार्जवमेव च |\nज्ञानं...   
4  दातव्यमिति यद्दानं दीयतेऽनुपकारिणे |\nदेशे काल...   

                                             english  
0  18.78 Wherever is Krishna, the Lord of Yoga; w...  
1  17.21 And, that gift which is given with a vie...  
2  18.50 Learn from Me in brief, O Arjuna, how he...  
3  18.42 Serenity, self-restraint, austerity, pur...  
4  17.20 That gift which is given to one who does...  
   rank     score       id  chapter  verse  \
0     1  0.175252  BG18.78       18     78   
1     2  0.147535   BG17.1       17      1   
2     3  0.144721  BG18.75       18     75   
3     4  0.106315  BG18.33       18     33   
4     5  0.098465  BG18.50       18     50   

                                            sanskrit  \
0  यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...   
1  अर्जुन उवाच |\nये शास्त्रविधिमुत्सृज्य यजन्ते ...   
2  व्यासप्रसादाच्छ्रुतवानेतद्गुह्यमहं परम् |\nयोग...   
3  धृत्या यया धारयते मनःप्राणेन्द्रियक्रियाः |\nय...   
4  सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...   

                                             english  
0  18.78 Wherever is Krishna, the Lord of Yoga; w...  
1  17.1 Arjuna said  Those who, setting aside the...  
2  18.75 Through the grace of Vyasa I have heard ...  
3  18.33 The unwavering firmness by which, throug...  
4  18.50 Learn from Me in brief, O Arjuna, how he...  
   rank     score       id  chapter  verse  \
0     1  0.173169  BG18.33       18     33   
1     2  0.125488  BG18.38       18     38   
2     3  0.096318  BG18.42       18     42   
3     4  0.094362  BG18.78       18     78   
4     5  0.088615  BG18.60       18     60   

                                            sanskrit  \
0  धृत्या यया धारयते मनःप्राणेन्द्रियक्रियाः |\nय...   
1  विषयेन्द्रियसंयोगाद्यत्तदग्रेऽमृतोपमम् |\nपरिण...   
2  शमो दमस्तपः शौचं क्षान्तिरार्जवमेव च |\nज्ञानं...   
3  यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...   
4  स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...   

                                             english  
0  18.33 The unwavering firmness by which, throug...  
1  18.38 That happiness which arises from the con...  
2  18.42 Serenity, self-restraint, austerity, pur...  
3  18.78 Wherever is Krishna, the Lord of Yoga; w...  
4  18.60 O Arjuna, bound by thy own Karma (action...  
   rank     score       id  chapter  verse  \
0     1  0.114925  BG18.33       18     33   
1     2  0.088002  BG18.60       18     60   
2     3  0.085845  BG18.50       18     50   
3     4  0.071902  BG18.38       18     38   
4     5  0.063714  BG18.18       18     18   

                                            sanskrit  \
0  धृत्या यया धारयते मनःप्राणेन्द्रियक्रियाः |\nय...   
1  स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...   
2  सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...   
3  विषयेन्द्रियसंयोगाद्यत्तदग्रेऽमृतोपमम् |\nपरिण...   
4  ज्ञानं ज्ञेयं परिज्ञाता त्रिविधा कर्मचोदना |\n...   

                                             english  
0  18.33 The unwavering firmness by which, throug...  
1  18.60 O Arjuna, bound by thy own Karma (action...  
2  18.50 Learn from Me in brief, O Arjuna, how he...  
3  18.38 That happiness which arises from the con...  
4  18.18 Knowledge, the knowable and the knower f...  
   rank     score       id  chapter  verse  \
0     1  0.206473  BG18.60       18     60   
1     2  0.198648  BG18.18       18     18   
2     3  0.181101  BG18.45       18     45   
3     4  0.152809  BG18.24       18     24   
4     5  0.148460  BG18.33       18     33   

                                            sanskrit  \
0  स्वभावजेन कौन्तेय निबद्धः स्वेन कर्मणा |\nकर्त...   
1  ज्ञानं ज्ञेयं परिज्ञाता त्रिविधा कर्मचोदना |\n...   
2  स्वे स्वे कर्मण्यभिरतः संसिद्धिं लभते नरः |\nस...   
3  यत्तु कामेप्सुना कर्म साहंकारेण वा पुनः |\nक्र...   
4  धृत्या यया धारयते मनःप्राणेन्द्रियक्रियाः |\nय...   

                                             english  
0  18.60 O Arjuna, bound by thy own Karma (action...  
1  18.18 Knowledge, the knowable and the knower f...  
2  5.10 He who does actions, offering them to Bra...  
3  18.24 But that action which is done by one lon...  
4  18.33 The unwavering firmness by which, throug...  


    Query	                                      Rank	score	   id	  chapter verse	sanskrit	                    English
0	What does the Bhagavad Gita say about karma?	1	0.106059	BG18.78	18	78	यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः |\n...	18.78 Wherever is Krishna, the Lord of Yoga; w...
1	What does the Bhagavad Gita say about karma?	2	0.099859	BG17.21	17	21	यत्तु प्रत्युपकारार्थं फलमुद्दिश्य वा पुनः |\n...	17.21 And, that gift which is given with a vie...
2	What does the Bhagavad Gita say about karma?	3	0.099556	BG18.50	18	50	सिद्धिं प्राप्तो यथा ब्रह्म तथाप्नोति निबोध मे...	18.50 Learn from Me in brief, O Arjuna, how he...
3	What does the Bhagavad Gita say about karma?	4	0.074624	BG18.42	18	42	शमो दमस्तपः शौचं क्षान्तिरार्जवमेव च |\nज्ञानं...	18.42 Serenity, self-restraint, austerity, pur...




