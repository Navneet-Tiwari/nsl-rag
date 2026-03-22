"""
generator.py
------------
Generator — sends validated context to Gemini and produces answers.
Second and final Gemini call in the NSL-RAG pipeline.

Receives audit report from Auditor.
Sends only validated facts to Gemini.
Returns raw LLM response for formatting.

Usage:
    from nsl_rag.generation.generator import Generator

    generator = Generator()
    response = generator.generate(query, audit_report, retrieval_result)
"""

import time

from google import genai
from google.genai import types as genai_types

from nsl_rag.config.config_loader import config
from nsl_rag.core.exceptions import GenerationError, RateLimitError, EmptyResponseError
from nsl_rag.core.logger import get_logger
from nsl_rag.reasoning.auditor import AuditReport
from nsl_rag.retrieval.navigator import RetrievalResult

log = get_logger(__name__)


class Generator:
    """
    Generates answers from validated, audited context using Gemini.

    Responsibilities:
    - Build a focused generation prompt from audit report
    - Call Gemini with only validated facts — never raw chunks
    - Handle rate limiting with exponential backoff
    - Return raw LLM text for OutputFormatter to structure

    This is intentionally simple — it does one thing.
    Prompt engineering lives here.
    Output structuring lives in formatter.py.
    """

    def __init__(self) -> None:
        self._setup_gemini()
        self._max_retries = config.llm.max_retries
        self._retry_delay = config.llm.retry_delay_seconds
        self._max_nodes = config.generation.max_context_nodes
        self._call_count = 0
        log.debug(
            "Generator initialised — model: %s, max_context_nodes: %d",
            config.llm.model,
            self._max_nodes,
        )

    def _setup_gemini(self) -> None:
        """Configure Gemini API client."""
        self._client = genai.Client(api_key=config.api_key)
        log.debug("Generator Gemini client configured")

    # ── Primary Interface ─────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        audit_report: AuditReport,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Generate an answer from validated context.

        Args:
            query:            Original natural language query.
            audit_report:     Audited facts from the reasoning pipeline.
            retrieval_result: Retrieval result containing trace info.

        Returns:
            Raw LLM response text.

        Raises:
            GenerationError: If generation fails after all retries.
            EmptyResponseError: If LLM returns empty response.
        """
        log.info("Generator generating answer for: %r", query)

        prompt = self._build_prompt(query, audit_report, retrieval_result)

        log.debug(
            "Prompt built — %d facts in context, %d contradiction flags",
            len(audit_report.all_facts),
            len(audit_report.contradictions),
        )

        return self._call_gemini(prompt)

    # ── Prompt Engineering ────────────────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        audit_report: AuditReport,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Build a focused generation prompt.

        Design principles:
        - Give Gemini only validated facts — no noise
        - Make the causal reasoning task explicit
        - Include contradiction flags so LLM can acknowledge uncertainty
        - Request structured response for formatter to parse
        - Keep prompt focused — token efficiency matters
        """
        # Build context from clean facts first, then flagged
        context_lines = []

        # Clean facts — high confidence
        for fact in audit_report.clean_facts[: self._max_nodes]:
            context_lines.append(fact.to_context_string())

        # Flagged facts — include with warning
        for fact in audit_report.flagged_facts:
            context_lines.append(f"[UNCERTAIN] {fact.to_context_string()}")

        context = "\n".join(context_lines)

        # Build contradiction warnings
        contradiction_section = ""
        if not audit_report.is_clean:
            flags = "\n".join(audit_report.to_context_flags())
            contradiction_section = (
                f"\nIMPORTANT — Contradictions detected:\n{flags}\n"
                f"Acknowledge these in your answer if relevant.\n"
            )

        # Build retrieval trace summary
        trace_summary = ""
        if retrieval_result.intent.target_entity:
            trace_summary = (
                f"\nSystem under investigation: "
                f"{retrieval_result.intent.target_entity}\n"
                f"Query type: {retrieval_result.intent.query_type.value}\n"
            )

        prompt = (
            f"You are a DevOps incident analysis assistant.\n"
            f"Analyze the following system facts and answer the query.\n"
            f"Provide a causal explanation — not just a description.\n"
            f"Be specific about dependency chains and failure paths.\n"
            f"{trace_summary}"
            f"\n--- VALIDATED SYSTEM FACTS ---\n"
            f"{context}\n"
            f"--- END FACTS ---\n"
            f"{contradiction_section}"
            f"\nQuery: {query}\n"
            f"\nRespond in this exact format:\n"
            f"ANSWER: [Your causal explanation here]\n"
            f"ROOT_CAUSE: [The primary root cause in one sentence]\n"
            f"CONFIDENCE: [HIGH / MEDIUM / LOW]\n"
            f"AFFECTED_COMPONENTS: [comma separated list]\n"
        )

        return prompt

    # ── Gemini Call ───────────────────────────────────────────────────────────

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini with exponential backoff on rate limit.
        """
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("Gemini generation attempt %d/%d", attempt, self._max_retries)

                response = self._client.models.generate_content(
                    model=config.llm.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=config.llm.temperature,
                        max_output_tokens=config.llm.max_tokens,
                    ),
                )

                self._call_count += 1

                if not response.text or not response.text.strip():
                    raise EmptyResponseError(
                        "Gemini returned empty response", details={"attempt": attempt}
                    )

                log.info(
                    "Generation successful (call #%d) — %d chars",
                    self._call_count,
                    len(response.text),
                )

                return response.text.strip()

            except EmptyResponseError:
                raise

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(
                    k in error_str
                    for k in ("quota", "rate", "429", "resource exhausted")
                )

                if is_rate_limit:
                    if attempt < self._max_retries:
                        log.warning(
                            "Rate limit — attempt %d/%d — waiting %ds",
                            attempt,
                            self._max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise RateLimitError(
                            "Generation rate limit exceeded",
                            details={"attempts": self._max_retries},
                        ) from e
                else:
                    raise GenerationError(
                        f"Generation failed: {e}", details={"attempt": attempt}
                    ) from e

        raise GenerationError("Generation failed after all retries")

    def get_stats(self) -> dict:
        """Returns generation usage stats."""
        return {"gemini_generation_calls": self._call_count}
