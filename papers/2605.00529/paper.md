---
arxiv_id: 2605.00529
title: "Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation"
published: 2026-04-30
primary_category: unknown
hf_upvotes: 3
popularity_score: 10
suggested_stage: inference
suggested_series: Second Brain
fast_verdict: spark-feasible
relevance_score: 0.7
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.00529
pdf_url: https://arxiv.org/pdf/2605.00529
hf_paper_url: https://huggingface.co/papers/2605.00529
---

# Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation

**Verdict:** spark-feasible · **Series:** Second Brain · **Stage:** inference · **Relevance:** 0.70 · **Popularity:** 10/100

> Hierarchical cross-document Tree-RAG — Second Brain F-arc fit; pgvector + NIM-Embed already in place.

## Abstract

Retrieval-augmented generation (RAG) enhances large language models with external knowledge, and tree-based RAG organizes documents into hierarchical indexes to support queries at multiple granularities. However, existing Tree-RAG methods designed for single-document retrieval face critical challenges in scaling to cross-document multi-hop questions: (1) poor distribution adaptability, where k-means clustering introduces noise due to rigid distribution assumptions; (2) structural isolation, as tree indexes lack explicit cross-document connections; and (3) coarse abstraction, which obscures fine-grained details. To address these limitations, we propose Ψ-RAG, a tree-RAG framework with two key components. First, a hierarchical abstract tree index built through an iterative "merging and collapse" process that adapts to data distributions without a priori assumption. Second, a multi-granular retrieval agent that intelligently interacts with the knowledge base with reorganized queries and an agent-powered hybrid retriever. Ψ-RAG supports diverse tasks from token-level question answering to document-level summarization. On cross-document multi-hop QA benchmarks, it outperforms RAPTOR by 25.9% and HippoRAG 2 by 7.4% in average F1 score. Code is available at https://github.com/Newiz430/Psi-RAG.

## Why this matters for ai-field-notes

- **Topic tags:** rag, retrieval, tree-rag, multi-document, embeddings
- **NVIDIA stack:** NemoRetriever, pgvector, NIM
- **Fast verdict rationale:** Hierarchical cross-document Tree-RAG — Second Brain F-arc fit; pgvector + NIM-Embed already in place.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.00529)
- [PDF](https://arxiv.org/pdf/2605.00529)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.00529)

