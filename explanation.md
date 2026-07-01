# WikiMind Architecture Decision: Why Search-Scoped Hybrid RAG?

This document explains the architectural reasoning behind WikiMind's retrieval
design. It captures the key design questions that shaped the pipeline and why
the final architecture uses **Tavily web search for article scoping** combined
with a **local Hybrid RAG pipeline for chunk-level retrieval**.

---

## The Core Problem

WikiMind is a question-answering system backed by Wikipedia. The fundamental
challenge is: **Wikipedia is enormous.** The English Wikipedia alone contains
over 6.8 million articles. When ingested and chunked into a vector database,
this produces tens of millions of text chunks.

When a user asks a question like *"What is the population of Tokyo?"*, a naive
vector search across the entire corpus returns chunks from Tokyo, Osaka, Japan
demographics, List of largest cities, Metropolitan areas, and dozens of
tangentially related articles. The LLM then receives a context window full of
noisy, loosely related chunks and either:

- Produces a hallucinated answer by stitching together unrelated facts.
- Fails the grounding check and loops back for another retrieval attempt.
- Loops indefinitely without converging on a focused, correct answer.

This is the **needle-in-a-haystack problem** applied to retrieval-augmented
generation.

---

## Three Candidate Architectures

### Option A: Pure Vector-Search RAG (Original Design)

```
User Query -> Query Expansion -> Hybrid Search (Dense + Sparse + RRF)
           -> Reranker -> Grade Documents -> Generate -> Self-RAG Checks
```

**How it works:** Embed the query, search the entire Qdrant collection,
fuse results with Reciprocal Rank Fusion, rerank with a cross-encoder, grade
relevance, and generate.

**Why it fails at scale:** With millions of chunks, even hybrid search
(dense + sparse) returns too many loosely related results. The reranker helps,
but it is working with candidates that are already contaminated by cross-article
noise. The document grading step then makes N serial LLM calls (one per
document), adding massive latency. If grading marks everything as irrelevant,
the pipeline falls back to web search or loops.

**Verdict:** Works well for small corpora (a few hundred articles). Breaks down
at Wikipedia scale.

---

### Option B: Tavily-Only (No RAG Pipeline)

```
User Query -> Tavily Web Search -> LLM generates answer from snippets
```

**How it works:** Send the query directly to Tavily (or any search API),
receive web search snippets, pass them to an LLM, and generate an answer.
Tavily can scope results to `en.wikipedia.org` and even return raw page
content.

**Why this is tempting:** It works. For most factual questions, web search
engines are already excellent at identifying the right Wikipedia article. The
returned snippets contain enough context for an LLM to answer. This approach
can be implemented in about 20 lines of code.

**Why it is insufficient for a production system:**

| Concern | Tavily-Only Limitation |
|---------|----------------------|
| **Cost at scale** | Every query costs an API call (~$0.01/search, 1,000 free/month). A production system serving thousands of users would incur significant costs just for retrieval. |
| **Latency for repeated queries** | Every query requires a network round-trip to Tavily, even if the same question was asked 5 minutes ago. No caching layer. |
| **Offline operation** | Impossible. If Tavily is down or rate-limited, the entire system fails. |
| **Snippet depth** | Tavily returns ~500-character snippets per result. For complex, multi-hop questions that require synthesizing information from multiple sections of an article, snippets are often insufficient. |
| **Control** | You have no control over how content is chunked, what metadata is attached, or how freshness is managed. You get whatever Tavily returns. |
| **Observability** | No trace of what was retrieved, how it was ranked, or why a particular answer was generated. |
| **Safety** | No guardrails layer. The LLM can hallucinate or generate unsafe content without any structural checks. |
| **Engineering depth** | There is no retrieval engineering to demonstrate. The entire "pipeline" is a single API call. |

**Verdict:** Works for prototypes and simple Q&A. Lacks the depth, control,
and cost efficiency required for a production-grade or portfolio-quality system.

---

### Option C: Search-Scoped Hybrid RAG (Chosen Architecture)

```
User Query -> Query Expansion -> Tavily (scoped to en.wikipedia.org)
           -> Identify 2-3 relevant article titles
           -> Scoped Hybrid Search (Qdrant, filtered to those articles only)
           -> Reranker -> Batched Document Grading -> Generate
           -> Hallucination Check -> Answer Quality Check -> Return
```

**How it works:** Use Tavily for what it does best -- identifying which
Wikipedia articles are relevant to the query. Then use the local Hybrid RAG
pipeline for what it does best -- precise, chunk-level retrieval within those
articles using dense embeddings, sparse BM25, and cross-encoder reranking.

**Why this is the right architecture:**

1. **Tavily solves the scoping problem.** A single, cheap API call identifies
   that "What is the population of Tokyo?" should look at the "Tokyo" article.
   This eliminates cross-article noise at the source.

2. **Qdrant solves the extraction problem.** Once scoped to the right article,
   the hybrid search (dense + sparse + RRF) finds the exact chunks that contain
   the answer. The cross-encoder reranker then picks the top 5 most relevant
   chunks. This level of precision is impossible with web search snippets alone.

3. **The two systems are complementary, not redundant:**
   - Tavily tells you *where* to look (article-level discovery).
   - The RAG pipeline extracts *exactly what* to answer with (chunk-level retrieval).

4. **The full engineering stack is preserved:**
   - Dual-layer semantic caching (L1 exact-match + L2 vector similarity).
   - Agentic CRAG/Self-RAG loops with hallucination and answer quality checks.
   - NeMo Guardrails for input/output safety.
   - Langfuse observability for full trace instrumentation.
   - Cross-encoder reranking for precision.
   - Reciprocal Rank Fusion for combining dense and sparse retrieval signals.

**Verdict:** Combines the strengths of web search (article discovery) with the
strengths of local RAG (precise extraction, caching, safety, observability).
This is the architecture that balances production readiness with engineering
depth.

---

## Tradeoff Summary

| Capability | Pure RAG | Tavily-Only | Search-Scoped RAG |
|-----------|----------|-------------|-------------------|
| Article identification accuracy | Low (noisy at scale) | High | High (uses Tavily) |
| Chunk-level precision | High (if right article) | Low (snippets only) | High (scoped search) |
| Latency (first query) | Medium | Medium | Medium |
| Latency (repeated query) | Low (cached) | Medium (no cache) | Low (cached) |
| Cost at scale | Low (local retrieval) | High (per-query API) | Low (1 Tavily call + local retrieval) |
| Offline capability | Full | None | Partial (fallback to unscoped RAG) |
| Observability | Full (Langfuse) | None | Full (Langfuse) |
| Safety guardrails | Full (NeMo) | None | Full (NeMo) |
| Engineering portfolio value | High | Low | High |

---

## Key Technical Decisions

### Why Tavily and not the Wikipedia API directly?

The MediaWiki API can search Wikipedia by title, but its full-text search is
primitive compared to modern search engines. Tavily leverages Google-grade
search relevance to identify articles, handles disambiguation automatically
(e.g., "Python" returns the programming language, not the snake, based on
query context), and returns structured results with relevance scores.

### Why not just use Tavily's raw content as the generation context?

Tavily can return raw page content with `include_raw_content=True`, but a
full Wikipedia article can be 50,000+ tokens. Passing the entire article to
an LLM is wasteful and exceeds most context windows. The RAG pipeline's
chunking, hybrid search, and reranking extract only the 5 most relevant
~512-token chunks, keeping the context focused and the generation grounded.

### Why keep the CRAG/Self-RAG agentic loops?

Even with scoped retrieval, the LLM can still hallucinate or produce answers
that do not address the question. The hallucination check verifies that the
generation is grounded in the retrieved chunks. The answer quality check
verifies that it actually answers the user's question. These loops provide a
measurable quality guarantee that a single-pass generation cannot.

### Why split retry counters instead of using a shared counter?

The original design used a single `retry_count` field incremented by both
the hallucination checker and the answer quality checker. This caused
cross-contamination: a hallucination retry consumed budget that the answer
quality checker needed, and vice versa. Splitting into `hallucination_retries`
and `answer_retries` ensures each check type has its own independent budget.

---

## Architecture Diagram

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Query Expansion  |
                    | (Multi-Query,    |
                    |  HyDE, StepBack) |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Tavily Search    |
                    | (scoped to       |
                    |  en.wikipedia)   |
                    +--------+---------+
                             |
                    Identified Article Titles
                             |
                    +--------v---------+
                    | Qdrant Hybrid    |
                    | Search (filtered |
                    | by article title)|
                    +----+--------+----+
                         |        |
                    Dense+BM25  RRF Fusion
                         |        |
                    +----v--------v----+
                    | FlashRank        |
                    | Cross-Encoder    |
                    | Reranker         |
                    +--------+---------+
                             |
                    Top 5 Relevant Chunks
                             |
                    +--------v---------+
                    | Batched Document |
                    | Grading (1 call) |
                    +--------+---------+
                             |
                    +--------v---------+
                    | NeMo Guardrails  |
                    | + LLM Generation |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v---------+         +--------v---------+
     | Hallucination    |         | Answer Quality   |
     | Check            |         | Check            |
     | (retries <= 2)   |         | (retries <= 2)   |
     +--------+---------+         +--------+---------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v---------+
                   | L1/L2 Cache      |
                   | Store + Return   |
                   +------------------+
```
