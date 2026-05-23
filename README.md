# CORD-19 Semantic Search System

> Master's Capstone · University of Sydney · March 2026

End-to-end semantic search system over a COVID-19 research corpus — 
built entirely from scratch.

## Overview

Semantic search pipeline that indexes 10 CORD-19 research papers into 
a 384-dimension vector store and returns clinically coherent ranked 
results across multiple query domains. Includes NER extraction, 
knowledge graph generation, and an interactive search interface.

## Results

### Pipeline Output
- ✅ 10 papers processed end-to-end
- ✅ Vector index: 10 documents, 384 dimensions
- ✅ Knowledge graph: 5 nodes saved as interactive HTML
- ✅ Complete analysis saved to `outputs/results/complete_analysis.json`

### Semantic Search Performance

| Query | Top Result | Score |
|---|---|---|
| COVID-19 symptoms and diagnosis | A Novel Approach for SARS-CoV-2 | 0.554 |
| Coronavirus symptoms | A Novel Approach for SARS-CoV-2 | 0.571 |
| Antiviral drug treatments | Xanthine-based phosphonates study | 0.489 |
| Vaccine development research | Evolutionary Medicine IV | 0.405 |
| Hospital safety protocols | Caring for persons in detention | 0.276 |

### Models Used
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (103 layers)
- **Vector Index:** 384-dimension FAISS index
- **NER:** Custom extractor over CORD-19 abstracts

## What I Built

### `run_pipeline.py` — Main Pipeline
- Loads and processes 10 CORD-19 papers
- Runs NER extraction on each paper
- Builds knowledge graph and saves as interactive HTML
- Builds 384-dimension vector search index
- Runs demo searches across 4 clinical query domains
- Saves complete analysis to JSON

### `test_search.py` — Interactive Search Interface
- Loads pre-built vector index
- Accepts free-text queries interactively
- Returns top 5 ranked results with similarity scores
- Shows entity and relation counts per result

### `src/` — Core Modules
- `vector_search.py` — FAISS-based semantic search
- `ner_extractor.py` — Named entity extraction
- `relation_extractor.py` — Relation extraction
- `knowledge_graph.py` — Graph construction and HTML export

## Repository Structure

```
├── run_pipeline.py           ← main pipeline runner
├── test_search.py            ← interactive search interface
├── src/
│   ├── vector_search.py      ← FAISS semantic search
│   ├── ner_extractor.py      ← NER extraction
│   ├── relation_extractor.py ← relation extraction
│   └── knowledge_graph.py   ← graph builder + HTML export
├── data/
│   └── papers/               ← CORD-19 paper JSON files
└── outputs/
    └── results/
        ├── complete_analysis.json  ← full pipeline output
        └── knowledge_graph.html   ← interactive graph
```

## Tech Stack

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search:** FAISS
- **NER:** Custom NLP extraction pipeline
- **Knowledge Graph:** NetworkX + Pyvis (interactive HTML)
- **Dataset:** CORD-19 (COVID-19 Open Research Dataset)
- **Language:** Python 3.13

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Interactive search
python test_search.py
```

---

*Part of a broader semantically-augmented RAG research platform built 
at University of Sydney. Full system includes injection stage comparison 
engine, MMD knowledge graph explorer, and NER/KG validation suite.*
