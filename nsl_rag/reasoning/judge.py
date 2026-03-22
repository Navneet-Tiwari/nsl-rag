"""
judge.py
--------
Judge — validates proposed facts against lattice constraints.
Second stage of the Prosecutor / Judge / Auditor pipeline.

The Judge applies symbolic rules to each proposed fact and
assigns a status: CONFIDENT, UNCERTAIN, or INVALID.
Only CONFIDENT and UNCERTAIN facts reach the LLM.
INVALID facts are discarded.

Usage:
    from nsl_rag.reasoning.judge import Judge

    judge = Judge(index)
    validated = judge.validate(proposed_facts)
"""

from nsl_rag.config.config_loader import config
from nsl_rag.core.exceptions import AllFactsRejectedError
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import FactStatus, NodeType
from nsl_rag.lattice.index import LatticeIndex
from nsl_rag.reasoning.prosecutor import ProposedFact

log = get_logger(__name__)


class Judge:
    """
    Validates proposed facts using symbolic lattice constraints.

    Validation rules applied to every fact:
    1. Source node must exist in lattice
    2. Source node confidence must meet threshold
    3. Node type must be appropriate for the claim type
    4. Dependency claims must reference existing nodes
    5. Temporal facts must have valid timestamps

    Strict mode (from config):
        True  → INVALID facts are rejected entirely
        False → INVALID facts are downgraded to UNCERTAIN
    """

    def __init__(self, index: LatticeIndex) -> None:
        self._index = index
        self._threshold = config.lattice.confidence_threshold
        self._strict_mode = config.reasoning.judge.strict_mode
        log.debug(
            "Judge initialised — threshold: %.2f, strict: %s",
            self._threshold,
            self._strict_mode,
        )

    # ── Primary Interface ─────────────────────────────────────────────────────

    def validate(
        self,
        facts: list[ProposedFact],
    ) -> list[ProposedFact]:
        """
        Validate all proposed facts.
        Returns only facts that passed validation.

        Args:
            facts: List of ProposedFacts from Prosecutor.

        Returns:
            List of facts with status set to CONFIDENT or UNCERTAIN.
            INVALID facts excluded if strict_mode is True.

        Raises:
            AllFactsRejectedError: If no facts pass validation.
        """
        log.info("Judge validating %d proposed facts", len(facts))

        validated: list[ProposedFact] = []
        counts = {
            FactStatus.CONFIDENT: 0,
            FactStatus.UNCERTAIN: 0,
            FactStatus.INVALID: 0,
        }

        for fact in facts:
            result = self._validate_fact(fact)
            counts[result.status] = counts.get(result.status, 0) + 1

            if result.status == FactStatus.INVALID:
                if not self._strict_mode:
                    # Downgrade to uncertain instead of rejecting
                    object.__setattr__(result, "status", FactStatus.UNCERTAIN)
                    validated.append(result)
                    log.debug(
                        "Fact %s downgraded INVALID → UNCERTAIN " "(strict_mode=False)",
                        fact.fact_id,
                    )
                else:
                    log.debug(
                        "Fact %s rejected as INVALID (strict_mode=True)", fact.fact_id
                    )
            else:
                validated.append(result)

        log.info(
            "Judge results — confident: %d, uncertain: %d, invalid: %d",
            counts[FactStatus.CONFIDENT],
            counts[FactStatus.UNCERTAIN],
            counts[FactStatus.INVALID],
        )

        if not validated:
            raise AllFactsRejectedError(
                "Judge rejected all proposed facts",
                details={
                    "total_facts": len(facts),
                    "strict_mode": self._strict_mode,
                    "confident_count": counts[FactStatus.CONFIDENT],
                    "uncertain_count": counts[FactStatus.UNCERTAIN],
                    "invalid_count": counts[FactStatus.INVALID],
                },
            )

        return validated

    # ── Validation Rules ──────────────────────────────────────────────────────

    def _validate_fact(self, fact: ProposedFact) -> ProposedFact:
        """
        Apply all validation rules to a single fact.
        Returns the fact with status set.
        """
        # Rule 1 — Source node must exist in lattice
        if not self._index.exists(fact.source_id):
            return self._set_status(
                fact,
                FactStatus.INVALID,
                f"Source node not in lattice: {fact.source_id}",
            )

        # Rule 2 — Confidence threshold
        if fact.confidence < self._threshold:
            return self._set_status(
                fact,
                FactStatus.UNCERTAIN,
                f"Below confidence threshold: "
                f"{fact.confidence} < {self._threshold}",
            )

        # Rule 3 — Empty claim
        if not fact.claim or len(fact.claim.strip()) < 10:
            return self._set_status(
                fact, FactStatus.INVALID, "Claim is empty or too short"
            )

        # Rule 4 — Dependency facts must reference existing nodes
        extraction_type = fact.metadata.get("extraction_type", "")
        if extraction_type in ("dependency_downstream", "dependency_upstream"):
            status = self._validate_dependency_fact(fact)
            if status != FactStatus.CONFIDENT:
                return self._set_status(
                    fact, status, "Dependency references unknown node"
                )

        # Rule 5 — Temporal facts must have valid timestamps
        if extraction_type == "temporal":
            if "timestamp" not in fact.metadata:
                return self._set_status(
                    fact, FactStatus.UNCERTAIN, "Temporal fact missing timestamp"
                )

        # Rule 6 — Deployment facts are always marked uncertain
        # They describe past events — relevant but not ground truth
        if fact.source_type == NodeType.DEPLOYMENT:
            return self._set_status(
                fact,
                FactStatus.UNCERTAIN,
                "Deployment facts are marked uncertain by policy",
            )

        # All rules passed
        return self._set_status(
            fact, FactStatus.CONFIDENT, "All validation rules passed"
        )

    def _validate_dependency_fact(
        self,
        fact: ProposedFact,
    ) -> FactStatus:
        """
        Validate that a dependency fact references nodes
        that actually exist in the lattice.
        """
        node = self._index.get(fact.source_id)

        for child_id in node.children:
            if not self._index.exists(child_id):
                log.debug("Dependency fact references missing child: %s", child_id)
                return FactStatus.UNCERTAIN

        for parent_id in node.parents:
            if not self._index.exists(parent_id):
                log.debug("Dependency fact references missing parent: %s", parent_id)
                return FactStatus.UNCERTAIN

        return FactStatus.CONFIDENT

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(
        self,
        fact: ProposedFact,
        status: FactStatus,
        reason: str,
    ) -> ProposedFact:
        """
        Set fact status and log the decision.
        Uses object.__setattr__ because ProposedFact is a dataclass
        but not frozen — direct assignment works here.
        """
        fact.status = status
        log.debug("Fact %s → %s | %s", fact.fact_id, status.value, reason)
        return fact

    # ── Summary ───────────────────────────────────────────────────────────────

    def summarise(self, facts: list[ProposedFact]) -> dict:
        """
        Returns a summary of validation results.
        Used for logging and benchmark metrics.
        """
        counts: dict[str, int] = {}
        for fact in facts:
            key = fact.status.value
            counts[key] = counts.get(key, 0) + 1

        return {
            "total": len(facts),
            "confident": counts.get("confident", 0),
            "uncertain": counts.get("uncertain", 0),
            "invalid": counts.get("invalid", 0),
        }
