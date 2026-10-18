AI-Powered Indian Tourism Recommender
Production Runtime Artifact Bundle
===================================

PURPOSE
-------
This bundle contains the minimum project-generated artifacts
required by the production recommendation application.

CORE PIPELINE
-------------
User Query
  -> City Detection / Query Parsing
  -> TF-IDF + SBERT/FAISS Hybrid Retrieval
  -> Reciprocal Rank Fusion
  -> Strict City Constraint
  -> Cross-Encoder Reranking
  -> Conservative Recommendation Layer
  -> Explainability
  -> Grounded RAG Context
  -> API LLM or Deterministic Fallback

EXTERNAL MODELS
---------------
The following public pretrained models are intentionally not
stored inside this artifact bundle:

1. sentence-transformers/all-MiniLM-L6-v2
   - Used to encode incoming semantic search queries.
   - Expected embedding dimension: 384.

2. cross-encoder/ms-marco-MiniLM-L-6-v2
   - Used to rerank the Hybrid Top-10 candidate set.

They should be downloaded/cached by the production environment.

NOT REQUIRED AT RUNTIME
-----------------------
- Raw/deduplicated 1.46M review corpus
- Raw semantic embedding matrix
- Project-trained DistilBERT sentiment model
- Full review-level sentiment prediction outputs
- Training checkpoints
- Evaluation benchmark files

Sentiment has already been aggregated into the final place-level
features used by the deployed recommendation system.

IMPORTANT
---------
The raw semantic embedding matrix is not required because the
FAISS index already contains the vectors needed for retrieval.

Do not infer missing structured tourism metadata. Sparse metadata
is optional supporting evidence only.

The deployed reranking configuration has been validated at a
Cross-Encoder candidate depth of 10.
