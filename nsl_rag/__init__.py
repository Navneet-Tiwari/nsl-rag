"""
nsl_rag
-------
NSL-RAG: Neuro-Symbolic Lattice Retrieval-Augmented Generation.

Formally bounded, causally traced retrieval for DevOps reasoning.

Quick start:
    from nsl_rag import NSLRAGPipeline

    pipeline = NSLRAGPipeline()
    response = pipeline.query("Why is the payment service failing?")
    print(response.answer)
    print(response.trace)
"""

from nsl_rag.pipeline import NSLRAGPipeline

__version__ = "0.1.0"
__all__ = ["NSLRAGPipeline"]
