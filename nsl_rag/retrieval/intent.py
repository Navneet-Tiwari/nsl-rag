"""
intent.py
---------
IntentExtractor — neural layer for NSL-RAG.
Converts natural language queries into structured symbolic tags
that the lattice traversal engine can work with.

Primary path: Gemini handles all natural language.
Cost controls: query normalisation + caching + minimal prompt.
Safety net: keyword fallback only when API completely unavailable.

Usage:
    from nsl_rag.retrieval.intent import IntentExtractor

    extractor = IntentExtractor()
    intent = extractor.extract("Why is the payment service failing?")
    print(intent.tags)          # ["payment", "service", "critical"]
    print(intent.query_type)    # QueryType.CAUSAL_TRACE
    print(intent.target_entity) # "payment_service"
"""

import json
import re
import time
from functools import lru_cache

from google import genai
from google.genai import types as genai_types

from nsl_rag.config.config_loader import config
from nsl_rag.core.exceptions import IntentExtractionError, RateLimitError
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import QueryType

log = get_logger(__name__)


# ── Intent Result ─────────────────────────────────────────────────────────────


class IntentResult:
    """
    Structured result from intent extraction.
    Passed to LatticeNavigator for retrieval.
    """

    def __init__(
        self,
        tags: list[str],
        query_type: QueryType,
        target_entity: str | None,
        raw_query: str,
        from_cache: bool = False,
        from_fallback: bool = False,
    ) -> None:
        self.tags = tags
        self.query_type = query_type
        self.target_entity = target_entity
        self.raw_query = raw_query
        self.from_cache = from_cache
        self.from_fallback = from_fallback

    def __repr__(self) -> str:
        source = (
            "cache"
            if self.from_cache
            else "fallback" if self.from_fallback else "gemini"
        )
        return (
            f"IntentResult("
            f"tags={self.tags}, "
            f"type={self.query_type.value}, "
            f"entity={self.target_entity!r}, "
            f"source={source!r})"
        )


# ── Intent Extractor ──────────────────────────────────────────────────────────


class IntentExtractor:
    """
    Extracts structured intent from natural language queries.

    Architecture:
        1. Normalise query — lowercase, strip noise
        2. Check cache — return free if hit
        3. Call Gemini with minimal prompt — primary path
        4. Cache result for future calls
        5. Keyword fallback — safety net only, logged as WARNING

    Cost controls:
        - Query normalisation collapses near-duplicate queries
        - LRU cache eliminates repeat Gemini calls
        - Minimal prompt keeps input tokens under 150
        - Exponential backoff reduces wasted retry tokens
    """

    KNOWN_ENTITIES = [
        "api_gateway",
        "order_service",
        "payment_service",
        "user_service",
        "fraud_detection",
        "inventory_service",
        "warehouse_service",
        "notification_service",
        "email_service",
        "payment_db",
        "orders_db",
        "user_db",
        "flash_sale_deployment",
        "fraud_model_deployment",
        "email_deployment",
    ]

    VALID_TAGS = [
        "payment",
        "orders",
        "users",
        "service",
        "database",
        "deployment",
        "critical",
        "fraud",
        "ml",
        "notification",
        "email",
        "inventory",
        "warehouse",
        "fulfillment",
        "gateway",
        "infrastructure",
        "authentication",
        "traffic",
        "root",
        "system",
        "ecommerce",
    ]

    # Tags that are too broad to be useful for retrieval
    # Gemini sometimes adds these — we filter them out
    NOISE_TAGS = {"root", "system", "ecommerce"}

    def __init__(self) -> None:
        self._setup_gemini()
        self._max_retries = config.llm.max_retries
        self._retry_delay = config.llm.retry_delay_seconds
        self._max_tags = config.retrieval.max_tags_per_query
        self._cache_enabled = config.retrieval.intent_cache_enabled
        self._cache: dict[str, IntentResult] = {}
        self._cache_hits = 0
        self._gemini_calls = 0
        self._fallback_calls = 0
        log.debug(
            "IntentExtractor initialised — model: %s, cache: %s",
            config.llm.model,
            self._cache_enabled,
        )

    # def _setup_gemini(self) -> None:
    #     """Configure Gemini API client."""
    #     genai.configure(api_key=config.api_key)
    #     self._model = genai.GenerativeModel(
    #         model_name=config.llm.model,
    #         generation_config={
    #             "temperature": config.llm.temperature,
    #             "max_output_tokens": config.llm.max_tokens,
    #         },
    #     )
    #     log.debug("Gemini client configured")
    def _setup_gemini(self) -> None:
        """Configure Gemini API client."""
        self._client = genai.Client(api_key=config.api_key)
        log.debug("Gemini client configured — model: %s", config.llm.model)

    # ── Primary Interface ─────────────────────────────────────────────────────

    def extract(self, query: str) -> IntentResult:
        """
        Extract structured intent from a natural language query.

        Args:
            query: Natural language query — any phrasing, any user.

        Returns:
            IntentResult with tags, query_type, and target_entity.

        Raises:
            IntentExtractionError: If all extraction methods fail.
        """
        log.info("Extracting intent: %r", query)

        # Step 1 — Normalise
        normalised = self._normalise(query)
        log.debug("Normalised query: %r", normalised)

        # Step 2 — Cache check
        if self._cache_enabled and normalised in self._cache:
            self._cache_hits += 1
            cached = self._cache[normalised]
            log.info(
                "Cache hit (%d total) — returning cached intent: %s",
                self._cache_hits,
                cached,
            )
            return IntentResult(
                tags=cached.tags,
                query_type=cached.query_type,
                target_entity=cached.target_entity,
                raw_query=query,
                from_cache=True,
            )

        # Step 3 — Gemini extraction
        try:
            result = self._extract_with_gemini(query, normalised)
            if self._cache_enabled:
                self._cache[normalised] = result
            return result

        except RateLimitError as e:
            log.warning("Rate limit exhausted: %s", e)
            log.warning("DEGRADED STATE — falling back to keyword extraction")
            return self._extract_with_fallback(query)

        except Exception as e:
            log.warning("Gemini failed: %s", e)
            log.warning("DEGRADED STATE — falling back to keyword extraction")
            return self._extract_with_fallback(query)

    # ── Query Normalisation ───────────────────────────────────────────────────

    def _normalise(self, query: str) -> str:
        """
        Normalise a query for cache key generation.
        Collapses near-duplicate queries into the same cache key.

        Operations:
        - Lowercase
        - Strip leading/trailing whitespace
        - Remove punctuation
        - Remove common stop words that don't affect intent
        - Collapse multiple spaces
        """
        normalised = query.lower().strip()
        normalised = re.sub(r"[^\w\s]", "", normalised)
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "what",
            "why",
            "how",
            "which",
            "that",
            "this",
        }
        tokens = [w for w in normalised.split() if w not in stop_words]
        return " ".join(tokens)

    # ── Gemini Extraction ─────────────────────────────────────────────────────

    def _extract_with_gemini(
        self,
        raw_query: str,
        normalised: str,
    ) -> IntentResult:
        """
        Use Gemini to extract structured intent.
        Uses minimal prompt to keep input tokens under 150.
        Exponential backoff on rate limit.
        """
        prompt = self._build_minimal_prompt(raw_query)
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug(
                    "Gemini call %d/%d — query: %r",
                    attempt,
                    self._max_retries,
                    raw_query,
                )
                # response = self._model.generate_content(prompt)
                # self._gemini_calls += 1
                # result = self._parse_response(response.text, raw_query)
                ## -------- Updated for new Gemini API -------
                response = self._client.models.generate_content(
                    model=config.llm.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=config.llm.temperature,
                        max_output_tokens=config.llm.max_tokens,
                    ),
                )
                self._gemini_calls += 1
                result = self._parse_response(response.text, raw_query)

                ##--------
                log.info(
                    "Gemini extraction successful (call #%d): %s",
                    self._gemini_calls,
                    result,
                )
                return result

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
                        delay *= 2  # exponential backoff
                    else:
                        raise RateLimitError(
                            "Rate limit exceeded after all retries",
                            details={
                                "attempts": self._max_retries,
                                "query": raw_query,
                            },
                        )
                else:
                    raise IntentExtractionError(
                        f"Gemini error: {e}", details={"query": raw_query}
                    )

        raise IntentExtractionError("Gemini failed after all retries")

    def _build_minimal_prompt(self, query: str) -> str:
        """
        Build a minimal, token-efficient prompt for Gemini.
        Target: under 150 input tokens.
        Gemini already understands DevOps concepts —
        we only need to provide our specific entity IDs and tag vocabulary.
        """
        entities = ", ".join(self.KNOWN_ENTITIES)
        tags = ", ".join(self.VALID_TAGS)

        return (
            f"DevOps system entities: {entities}\n"
            f"Valid tags: {tags}\n"
            f"Query types: causal_trace, dependency_map, "
            f"status_check, impact_analysis, general\n\n"
            f'Query: "{query}"\n\n'
            f"Return JSON only — no markdown, no explanation:\n"
            f'{{"tags": [...], "query_type": "...", '
            f'"target_entity": "entity_id or null"}}\n'
            f"Max {self._max_tags} tags. "
            f"Tags must be from valid tags list only."
        )

    def _parse_response(
        self,
        response_text: str,
        raw_query: str,
    ) -> IntentResult:
        """
        Parse Gemini JSON response into IntentResult.
        Handles markdown fences and common formatting issues.
        """
        clean = response_text.strip()

        # Strip markdown fences if present
        if "```" in clean:
            clean = re.sub(r"```(?:json)?\n?", "", clean).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON from response if wrapped in text
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise IntentExtractionError(
                    "Cannot parse Gemini response as JSON",
                    details={"response": response_text},
                )

        # Validate tags against known vocabulary
        tags = [
            t.lower().strip()
            for t in data.get("tags", [])
            if t.lower().strip() in self.VALID_TAGS
            and t.lower().strip() not in self.NOISE_TAGS
        ][: self._max_tags]

        # Parse query type
        query_type_str = data.get("query_type", "general")
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.GENERAL

        # Validate target entity
        target_entity = data.get("target_entity")
        if target_entity and target_entity not in self.KNOWN_ENTITIES:
            log.debug("Unknown entity %r from Gemini — setting to None", target_entity)
            target_entity = None

        if not tags:
            raise IntentExtractionError(
                "Gemini returned no valid tags after validation",
                details={"response": response_text, "query": raw_query},
            )

        return IntentResult(
            tags=tags,
            query_type=query_type,
            target_entity=target_entity,
            raw_query=raw_query,
        )

    # ── Keyword Fallback ──────────────────────────────────────────────────────

    def _extract_with_fallback(self, query: str) -> IntentResult:
        """
        Keyword-based fallback — SAFETY NET ONLY.
        Only called when Gemini API is completely unavailable.
        Always logged as WARNING — this is a degraded state.

        Limitations:
        - Only works for queries containing exact tag keywords
        - Will miss natural language variations
        - Not suitable as a primary extraction method
        """
        log.warning("FALLBACK EXTRACTION — results may be degraded: %r", query)
        self._fallback_calls += 1
        query_lower = query.lower()

        tags = [tag for tag in self.VALID_TAGS if tag in query_lower][: self._max_tags]

        if any(
            w in query_lower
            for w in ["why", "cause", "fail", "down", "error", "broken"]
        ):
            query_type = QueryType.CAUSAL_TRACE
        elif any(w in query_lower for w in ["depend", "need", "require", "use"]):
            query_type = QueryType.DEPENDENCY_MAP
        elif any(
            w in query_lower for w in ["status", "healthy", "running", "up", "alive"]
        ):
            query_type = QueryType.STATUS_CHECK
        elif any(
            w in query_lower for w in ["impact", "affect", "downstream", "consequence"]
        ):
            query_type = QueryType.IMPACT_ANALYSIS
        else:
            query_type = QueryType.GENERAL

        target_entity = next(
            (e for e in self.KNOWN_ENTITIES if e.replace("_", " ") in query_lower), None
        )

        if not tags:
            raise IntentExtractionError(
                "Fallback extraction found no tags — cannot proceed",
                details={"query": query},
            )

        return IntentResult(
            tags=tags,
            query_type=query_type,
            target_entity=target_entity,
            raw_query=query,
            from_fallback=True,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Returns usage statistics for cost monitoring.
        Log this after benchmarks to understand actual Gemini usage.
        """
        total = self._gemini_calls + self._cache_hits + self._fallback_calls
        return {
            "total_extractions": total,
            "gemini_calls": self._gemini_calls,
            "cache_hits": self._cache_hits,
            "fallback_calls": self._fallback_calls,
            "cache_hit_rate": (
                round(self._cache_hits / total, 3) if total > 0 else 0.0
            ),
        }
