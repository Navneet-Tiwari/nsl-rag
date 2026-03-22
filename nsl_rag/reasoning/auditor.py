"""
auditor.py
----------
Auditor — detects contradictions across validated facts.
Third and final stage of the Prosecutor / Judge / Auditor pipeline.

The Auditor cross-validates all Judge-approved facts looking for
logical inconsistencies before they reach the LLM generator.
It does not reject facts — it flags them with contradiction markers
so the LLM and downstream consumers can make informed decisions.

Usage:
    from nsl_rag.reasoning.auditor import Auditor

    auditor = Auditor()
    report = auditor.audit(validated_facts)
"""

from dataclasses import dataclass, field

from nsl_rag.config.config_loader import config
from nsl_rag.core.exceptions import UnresolvableContradictionError
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import FactStatus
from nsl_rag.reasoning.prosecutor import ProposedFact

log = get_logger(__name__)


# ── Audit Report ──────────────────────────────────────────────────────────────


@dataclass
class ContradictionFlag:
    """
    Records a detected contradiction between two facts.
    """

    fact_a_id: str
    fact_b_id: str
    reason: str
    severity: str = "warning"  # warning | critical


@dataclass
class AuditReport:
    """
    Complete audit result from the Auditor.
    Passed to the Generator alongside validated facts.

    Fields:
        clean_facts:     Facts with no contradictions — safe to use
        flagged_facts:   Facts involved in contradictions — use with caution
        contradictions:  List of detected contradiction pairs
        is_clean:        True if no contradictions detected
        summary:         Human readable audit summary
    """

    clean_facts: list[ProposedFact] = field(default_factory=list)
    flagged_facts: list[ProposedFact] = field(default_factory=list)
    contradictions: list[ContradictionFlag] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.contradictions) == 0

    @property
    def all_facts(self) -> list[ProposedFact]:
        """All facts — clean and flagged combined."""
        return self.clean_facts + self.flagged_facts

    @property
    def summary(self) -> str:
        if self.is_clean:
            return f"Clean audit — {len(self.clean_facts)} facts, " f"no contradictions"
        return (
            f"Audit flagged {len(self.contradictions)} contradiction(s) "
            f"— {len(self.clean_facts)} clean, "
            f"{len(self.flagged_facts)} flagged"
        )

    def to_context_flags(self) -> list[str]:
        """
        Formats contradiction flags for LLM context.
        Tells the LLM what inconsistencies were detected.
        """
        if self.is_clean:
            return []
        flags = []
        for c in self.contradictions:
            flags.append(
                f"CONTRADICTION DETECTED: {c.reason} "
                f"(facts {c.fact_a_id} and {c.fact_b_id})"
            )
        return flags


# ── Auditor ───────────────────────────────────────────────────────────────────


class Auditor:
    """
    Cross-validates facts for logical contradictions.

    Contradiction types detected:
    1. Temporal contradictions — two events at impossible times
    2. Status contradictions — node described as both up and down
    3. Ownership contradictions — node has two different owners
    4. Version contradictions — two different versions claimed
    5. Confidence contradictions — same claim with very different confidence
    """

    def __init__(self) -> None:
        self._threshold = config.reasoning.auditor.contradiction_threshold
        log.debug("Auditor initialised — threshold: %.2f", self._threshold)

    # ── Primary Interface ─────────────────────────────────────────────────────

    def audit(
        self,
        facts: list[ProposedFact],
    ) -> AuditReport:
        """
        Audit all validated facts for contradictions.

        Args:
            facts: Judge-validated facts to audit.

        Returns:
            AuditReport with clean facts, flagged facts,
            and contradiction details.
        """
        log.info("Auditor auditing %d facts", len(facts))

        contradictions: list[ContradictionFlag] = []
        flagged_ids: set[str] = set()

        # Run all contradiction checks
        checks = [
            self._check_temporal_contradictions,
            self._check_status_contradictions,
            self._check_ownership_contradictions,
            self._check_version_contradictions,
            self._check_confidence_contradictions,
        ]

        for check in checks:
            found = check(facts)
            contradictions.extend(found)
            for c in found:
                flagged_ids.add(c.fact_a_id)
                flagged_ids.add(c.fact_b_id)

        # Split into clean and flagged
        clean_facts = [f for f in facts if f.fact_id not in flagged_ids]
        flagged_facts = [f for f in facts if f.fact_id in flagged_ids]

        report = AuditReport(
            clean_facts=clean_facts,
            flagged_facts=flagged_facts,
            contradictions=contradictions,
        )

        if report.is_clean:
            log.info("Audit complete — clean | %d facts", len(clean_facts))
        else:
            log.warning(
                "Audit complete — %d contradiction(s) detected | "
                "%d clean, %d flagged",
                len(contradictions),
                len(clean_facts),
                len(flagged_facts),
            )
            for c in contradictions:
                log.warning("  [%s] %s", c.severity.upper(), c.reason)

        return report

    # ── Contradiction Checks ──────────────────────────────────────────────────

    def _check_temporal_contradictions(
        self,
        facts: list[ProposedFact],
    ) -> list[ContradictionFlag]:
        """
        Detect temporal contradictions — events with impossible ordering.
        Example: Service crashed before it was deployed.
        """
        contradictions = []
        temporal_facts = [
            f
            for f in facts
            if f.metadata.get("extraction_type") == "temporal"
            and "timestamp" in f.metadata
        ]

        # Group temporal facts by source node
        by_node: dict[str, list[ProposedFact]] = {}
        for fact in temporal_facts:
            node_id = fact.source_id
            by_node.setdefault(node_id, []).append(fact)

        # Check for crash before deployment
        for node_id, node_facts in by_node.items():
            deployed = next(
                (f for f in node_facts if "deployed" in f.claim.lower()), None
            )
            crashed = next(
                (f for f in node_facts if "crashed" in f.claim.lower()), None
            )

            if deployed and crashed:
                deployed_time = deployed.metadata.get("timestamp", "")
                crashed_time = crashed.metadata.get("timestamp", "")

                # Simple string comparison works for HH:MM format
                if crashed_time and deployed_time:
                    if crashed_time < deployed_time:
                        contradictions.append(
                            ContradictionFlag(
                                fact_a_id=deployed.fact_id,
                                fact_b_id=crashed.fact_id,
                                reason=(
                                    f"{node_id} crashed ({crashed_time}) "
                                    f"before it was deployed ({deployed_time})"
                                ),
                                severity="critical",
                            )
                        )

        return contradictions

    def _check_status_contradictions(
        self,
        facts: list[ProposedFact],
    ) -> list[ContradictionFlag]:
        """
        Detect status contradictions — node described as both healthy and failed.
        """
        contradictions = []

        positive_terms = {"healthy", "running", "up", "available", "online"}
        negative_terms = {
            "failing",
            "failed",
            "down",
            "crashed",
            "unavailable",
            "error",
            "503",
            "timeout",
        }

        # Group facts by source node
        by_node: dict[str, list[ProposedFact]] = {}
        for fact in facts:
            by_node.setdefault(fact.source_id, []).append(fact)

        for node_id, node_facts in by_node.items():
            has_positive = any(
                any(term in f.claim.lower() for term in positive_terms)
                for f in node_facts
            )
            has_negative = any(
                any(term in f.claim.lower() for term in negative_terms)
                for f in node_facts
            )

            if has_positive and has_negative:
                positive_fact = next(
                    f
                    for f in node_facts
                    if any(t in f.claim.lower() for t in positive_terms)
                )
                negative_fact = next(
                    f
                    for f in node_facts
                    if any(t in f.claim.lower() for t in negative_terms)
                )
                # Only flag if they are genuinely different facts
                if positive_fact.fact_id != negative_fact.fact_id:
                    contradictions.append(
                        ContradictionFlag(
                            fact_a_id=positive_fact.fact_id,
                            fact_b_id=negative_fact.fact_id,
                            reason=(
                                f"{node_id} described as both " f"healthy and failing"
                            ),
                            severity="warning",
                        )
                    )

        return contradictions

    def _check_ownership_contradictions(
        self,
        facts: list[ProposedFact],
    ) -> list[ContradictionFlag]:
        """
        Detect ownership contradictions — two different owners for same node.
        """
        contradictions = []

        ownership_facts = [
            f for f in facts if f.metadata.get("metadata_key") == "owner"
        ]

        by_node: dict[str, list[ProposedFact]] = {}
        for fact in ownership_facts:
            by_node.setdefault(fact.source_id, []).append(fact)

        for node_id, node_facts in by_node.items():
            if len(node_facts) > 1:
                owners = set(f.claim for f in node_facts)
                if len(owners) > 1:
                    contradictions.append(
                        ContradictionFlag(
                            fact_a_id=node_facts[0].fact_id,
                            fact_b_id=node_facts[1].fact_id,
                            reason=(f"{node_id} has conflicting ownership claims"),
                            severity="warning",
                        )
                    )

        return contradictions

    def _check_version_contradictions(
        self,
        facts: list[ProposedFact],
    ) -> list[ContradictionFlag]:
        """
        Detect version contradictions — two different versions for same node.
        """
        contradictions = []

        version_facts = [
            f for f in facts if f.metadata.get("metadata_key") == "version"
        ]

        by_node: dict[str, list[ProposedFact]] = {}
        for fact in version_facts:
            by_node.setdefault(fact.source_id, []).append(fact)

        for node_id, node_facts in by_node.items():
            if len(node_facts) > 1:
                versions = set(f.claim for f in node_facts)
                if len(versions) > 1:
                    contradictions.append(
                        ContradictionFlag(
                            fact_a_id=node_facts[0].fact_id,
                            fact_b_id=node_facts[1].fact_id,
                            reason=(f"{node_id} has conflicting version claims"),
                            severity="warning",
                        )
                    )

        return contradictions

    def _check_confidence_contradictions(
        self,
        facts: list[ProposedFact],
    ) -> list[ContradictionFlag]:
        """
        Detect confidence contradictions — same source node has facts
        with very different confidence scores.
        This can indicate stale or unreliable data.
        """
        contradictions = []

        by_node: dict[str, list[ProposedFact]] = {}
        for fact in facts:
            by_node.setdefault(fact.source_id, []).append(fact)

        for node_id, node_facts in by_node.items():
            if len(node_facts) < 2:
                continue
            confidences = [f.confidence for f in node_facts]
            max_conf = max(confidences)
            min_conf = min(confidences)

            if max_conf - min_conf >= self._threshold:
                high_fact = next(f for f in node_facts if f.confidence == max_conf)
                low_fact = next(f for f in node_facts if f.confidence == min_conf)
                contradictions.append(
                    ContradictionFlag(
                        fact_a_id=high_fact.fact_id,
                        fact_b_id=low_fact.fact_id,
                        reason=(
                            f"{node_id} has large confidence spread: "
                            f"{max_conf:.2f} vs {min_conf:.2f}"
                        ),
                        severity="warning",
                    )
                )

        return contradictions
