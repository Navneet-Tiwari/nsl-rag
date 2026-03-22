# Known Limitations — NSL-RAG v0.1.0

## 1. Manual Lattice Construction
The knowledge lattice is currently constructed manually.
Automatic construction via Formal Concept Analysis is
planned for v0.2.0. This limits scalability to domains
where a lattice can be defined by hand.

## 2. Synthetic Dataset
The benchmark uses a synthetic e-commerce microservices
dataset designed specifically to demonstrate multi-hop
reasoning. Results on real-world datasets may differ.

## 3. Hardcoded Benchmark Tags
The retrieval benchmark uses pre-defined intent tags
to bypass LLM rate limits during evaluation.
Full end-to-end benchmark with Gemini intent extraction
is pending — results will be published in v0.2.0.

## 4. LLM Generation Metrics Pending
Token efficiency and answer accuracy comparisons with
LLM generation are not yet included due to API rate
limits on the free tier. These will be added in v0.2.0.

## 5. Single Domain
The current implementation targets DevOps incident
reasoning. Generalisation to other domains requires
domain-specific lattice construction.

## 6. Proof of Concept Status
This is a research preview — not production software.
It demonstrates the NSL-RAG architecture and its
formal properties. Production hardening is planned
for v0.3.0.

## 7. Comparison Against Graph-Based RAG Approaches
NSL-RAG has not yet been formally benchmarked against
graph-based RAG approaches including Microsoft GraphRAG,
HippoRAG, and RAPTOR. The architectural argument for
NSL-RAG's advantage over these systems is:

- GraphRAG retrieves subgraphs — no formal ordering,
  no bounded retrieval proof
- HippoRAG uses Personalized PageRank — probabilistic,
  not symbolically constrained
- RAPTOR uses tree summarisation — no causal reasoning

Formal empirical benchmarks against these systems are
planned for v0.2.0 and will be included in the
accompanying research paper.