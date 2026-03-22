"""
formatter.py
------------
OutputFormatter — structures LLM response into standard NSL-RAG output.
Final step in the generation pipeline.

Takes raw Gemini response text and produces a clean,
structured NSLRAGResponse object with answer, trace,
confidence score, and any flags.

Usage:
    from nsl_rag.generation.formatter import OutputFormatter

    formatter = OutputFormatter()
    response  = formatter.format(
        raw_response, audit_report, retrieval_result
    )
    print(response.answer)
    print(response.confidence)
    print(response.trace)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from nsl_rag.core.logger import get_logger
from nsl_rag.reasoning.auditor import AuditReport
from nsl_rag.retrieval.navigator import RetrievalResult

log = get_logger(__name__)


# ── NSL-RAG Response ──────────────────────────────────────────────────────────


@dataclass
class NSLRAGResponse:
    """
    The final structured output of the NSL-RAG pipeline.
    This is what gets returned to the user.

    Fields:
        answer:               The causal explanation answer
        root_cause:           Primary root cause in one sentence
        confidence:           HIGH / MEDIUM / LOW — from LLM
        confidence_score:     Numeric 0.0-1.0 — derived from confidence
        affected_components:  List of affected system components
        trace:                Full reasoning trace — lattice path taken
        flags:                Contradiction flags from Auditor
        retrieved_nodes:      Node IDs that were retrieved
        query:                Original user query
        query_type:           Type of query — causal_trace etc
        facts_used:           Number of facts sent to LLM
        generated_at:         Timestamp
    """

    answer: str
    root_cause: str
    confidence: str
    confidence_score: float
    affected_components: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    retrieved_nodes: list[str] = field(default_factory=list)
    query: str = ""
    query_type: str = ""
    facts_used: int = 0
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return {
            "answer": self.answer,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "affected_components": self.affected_components,
            "trace": self.trace,
            "flags": self.flags,
            "retrieved_nodes": self.retrieved_nodes,
            "query": self.query,
            "query_type": self.query_type,
            "facts_used": self.facts_used,
            "generated_at": self.generated_at.isoformat(),
        }

    def display(self) -> str:
        """
        Human-readable display format.
        Used for demo output and paper figures.
        """
        lines = [
            "=" * 60,
            f"QUERY: {self.query}",
            "=" * 60,
            f"ANSWER:\n{self.answer}",
            "",
            f"ROOT CAUSE: {self.root_cause}",
            "",
            f"CONFIDENCE: {self.confidence} ({self.confidence_score:.2f})",
            "",
            f"AFFECTED: {', '.join(self.affected_components)}",
        ]

        if self.flags:
            lines.append("")
            lines.append("⚠ FLAGS:")
            for flag in self.flags:
                lines.append(f"  {flag}")

        lines.append("")
        lines.append("REASONING TRACE:")
        for step in self.trace:
            lines.append(f"  → {step}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Output Formatter ──────────────────────────────────────────────────────────


class OutputFormatter:
    """
    Structures raw LLM response into NSLRAGResponse.

    Parses the structured format requested in the generation prompt:
        ANSWER: ...
        ROOT_CAUSE: ...
        CONFIDENCE: HIGH / MEDIUM / LOW
        AFFECTED_COMPONENTS: ...

    Falls back gracefully if LLM doesn't follow the format exactly.
    """

    # Confidence label to numeric score mapping
    CONFIDENCE_SCORES = {
        "HIGH": 0.90,
        "MEDIUM": 0.65,
        "LOW": 0.35,
    }

    def __init__(self) -> None:
        log.debug("OutputFormatter initialised")

    # ── Primary Interface ─────────────────────────────────────────────────────

    def format(
        self,
        raw_response: str,
        audit_report: AuditReport,
        retrieval_result: RetrievalResult,
    ) -> NSLRAGResponse:
        """
        Format raw LLM response into structured NSLRAGResponse.

        Args:
            raw_response:     Raw text from Generator.
            audit_report:     Audit report for flags and fact count.
            retrieval_result: Retrieval result for trace and metadata.

        Returns:
            Fully structured NSLRAGResponse.
        """
        log.info("Formatting LLM response — %d chars", len(raw_response))

        # Parse structured fields from LLM response
        answer = self._extract_field(raw_response, "ANSWER")
        root_cause = self._extract_field(raw_response, "ROOT_CAUSE")
        confidence_label = (
            self._extract_field(raw_response, "CONFIDENCE").upper().strip()
        )
        affected_raw = self._extract_field(raw_response, "AFFECTED_COMPONENTS")

        # Clean and validate fields
        answer = answer or raw_response
        root_cause = root_cause or "Could not determine root cause"

        confidence_label = self._clean_confidence(confidence_label)
        confidence_score = self.CONFIDENCE_SCORES.get(confidence_label, 0.5)

        affected_components = self._parse_components(affected_raw)

        # Build reasoning trace
        trace = self._build_trace(retrieval_result, audit_report)

        # Collect flags
        flags = audit_report.to_context_flags()

        response = NSLRAGResponse(
            answer=answer,
            root_cause=root_cause,
            confidence=confidence_label,
            confidence_score=confidence_score,
            affected_components=affected_components,
            trace=trace,
            flags=flags,
            retrieved_nodes=retrieval_result.node_ids,
            query=retrieval_result.query,
            query_type=retrieval_result.intent.query_type.value,
            facts_used=len(audit_report.all_facts),
            generated_at=datetime.utcnow(),
        )

        log.info(
            "Response formatted — confidence: %s (%.2f), "
            "components: %d, trace: %d steps",
            response.confidence,
            response.confidence_score,
            len(response.affected_components),
            len(response.trace),
        )

        return response

    # ── Parsing Helpers ───────────────────────────────────────────────────────

    def _extract_field(self, text: str, field_name: str) -> str:
        """
        Extract a labelled field from structured LLM response.
        Handles single-line and multi-line field values.
        Robust to missing trailing newlines.
        """
        pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]{{2,}}:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # Fallback — try simple line-by-line search
        for line in text.split("\n"):
            if line.upper().startswith(f"{field_name}:"):
                value = line[len(field_name) + 1 :].strip()
                if value:
                    return value

        return ""

    def _clean_confidence(self, raw: str) -> str:
        """
        Normalize confidence label to HIGH / MEDIUM / LOW.
        Handles variations like 'High', 'high confidence', etc.
        """
        raw_upper = raw.upper()
        if "HIGH" in raw_upper:
            return "HIGH"
        elif "MEDIUM" in raw_upper or "MED" in raw_upper:
            return "MEDIUM"
        elif "LOW" in raw_upper:
            return "LOW"
        return "MEDIUM"

    def _parse_components(self, raw: str) -> list[str]:
        """
        Parse comma-separated affected components list.
        Cleans and deduplicates entries.
        """
        if not raw:
            return []
        components = [c.strip() for c in raw.split(",")]
        components = [c for c in components if c]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in components:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        return unique

    def _build_trace(
        self,
        retrieval_result: RetrievalResult,
        audit_report: AuditReport,
    ) -> list[str]:
        """
        Build the complete reasoning trace.
        Combines retrieval trace with fact validation summary.
        This is the audit trail that makes NSL-RAG explainable.
        """
        trace = []

        # Step 1 — Intent
        intent = retrieval_result.intent
        trace.append(
            f"Intent extracted: tags={intent.tags}, "
            f"type={intent.query_type.value}, "
            f"entity={intent.target_entity}"
        )

        # Step 2 — Retrieval
        trace.append(
            f"Lattice traversal retrieved " f"{len(retrieval_result.nodes)} nodes"
        )
        for node in retrieval_result.nodes:
            trace.append(f"  ↳ {node.to_trace_entry()}")

        # Step 3 — Dependency chain
        if retrieval_result.dependency_chain:
            chain = " → ".join(n.node_id for n in retrieval_result.dependency_chain)
            trace.append(f"Dependency chain: {chain}")

        # Step 4 — Reasoning
        trace.append(
            f"Prosecutor proposed facts, " f"Judge validated {audit_report.summary}"
        )

        # Step 5 — Audit
        if audit_report.is_clean:
            trace.append("Auditor: clean — no contradictions detected")
        else:
            trace.append(
                f"Auditor: {len(audit_report.contradictions)} "
                f"contradiction(s) flagged"
            )

        return trace
