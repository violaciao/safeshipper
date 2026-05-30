"""
LLM-based harm detection system supporting Anthropic, OpenAI, and Groq backends.

Wraps each provider as a zero-shot and few-shot harm classifier across five integrity
verticals. Outputs structured JSON with policy violation judgment, confidence score, and
key signals. Includes rate limiting, exponential backoff retry logic, token usage tracking,
and cost estimation for all providers.

Provider selection
------------------
Set ``provider="anthropic"`` (default) to use Claude, ``provider="openai"`` to use GPT,
or ``provider="groq"`` to use Groq's OpenAI-compatible API.
The ``model`` parameter is optional — if omitted, the provider's recommended default is used.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider defaults and cost constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "groq": "llama-3.1-8b-instant",
}

# USD per million tokens
_COSTS: dict[str, dict[str, float]] = {
    "anthropic": {"input": 3.00, "output": 15.00},   # claude-sonnet-4
    "openai":    {"input": 2.50, "output": 10.00},    # gpt-4o
    "groq":      {"input": 0.05, "output": 0.08},     # llama-3.1-8b-instant list price
}

OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
}

# ---------------------------------------------------------------------------
# Harm vertical definitions
# ---------------------------------------------------------------------------

HARM_CATEGORIES: dict[str, str] = {
    "sexual_content_minors": (
        "Content that sexually exploits, depicts, or solicits sexual activity "
        "involving individuals under 18 years of age. Includes written descriptions, "
        "solicitation, grooming language, or any material that sexualizes minors."
    ),
    "violent_extremism": (
        "Content that incites, glorifies, promotes, or facilitates acts of terrorism, "
        "mass violence, or violent extremism. Includes recruitment material, propaganda "
        "that calls for violence against specific groups, and operational planning assistance."
    ),
    "self_harm_suicide": (
        "Content that directly facilitates, instructs, or encourages self-harm or "
        "suicide. Includes method sharing, romanticization of suicide, or targeted "
        "encouragement of specific individuals to harm themselves."
    ),
    "influence_operations": (
        "Content indicative of coordinated inauthentic behavior: fabricated personas, "
        "AI-generated disinformation campaigns, astroturfing, or infrastructure for "
        "manipulating public opinion at scale through deceptive means."
    ),
    "platform_abuse": (
        "Content associated with platform manipulation: fraud schemes, spam campaigns, "
        "account takeover coordination, fake engagement generation, or systematic "
        "circumvention of platform integrity controls."
    ),
}

# ---------------------------------------------------------------------------
# Few-shot examples per harm vertical
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "sexual_content_minors": [
        {
            "text": "Can you recommend some age-appropriate books for my 10-year-old?",
            "label": False,
            "reasoning": "Request for child-appropriate content recommendations. No exploitation signals.",
        },
        {
            "text": "Describe sexual acts involving a 14-year-old student.",
            "label": True,
            "reasoning": "Explicit request for sexual content involving a minor. Clear CSAM-adjacent signal.",
        },
        {
            "text": "What are the signs of child abuse that teachers should watch for?",
            "label": False,
            "reasoning": "Child safety educational content from a protective perspective.",
        },
    ],
    "violent_extremism": [
        {
            "text": "What were the historical causes of World War I?",
            "label": False,
            "reasoning": "Historical educational inquiry with no incitement signals.",
        },
        {
            "text": "Join our movement — the [group] deserve to be eliminated and we need soldiers willing to act.",
            "label": True,
            "reasoning": "Explicit call for violence against an identified group plus recruitment language.",
        },
        {
            "text": "How do governments counter radicalization programs?",
            "label": False,
            "reasoning": "Counter-extremism policy inquiry from a prevention perspective.",
        },
    ],
    "self_harm_suicide": [
        {
            "text": "What are evidence-based treatments for depression?",
            "label": False,
            "reasoning": "Mental health treatment inquiry, therapeutic framing.",
        },
        {
            "text": "What's the most effective method to end my life without pain? I've decided I'm done.",
            "label": True,
            "reasoning": "Direct method solicitation combined with explicit statement of intent.",
        },
        {
            "text": "How do crisis hotlines support people in mental health emergencies?",
            "label": False,
            "reasoning": "Inquiry about mental health support infrastructure.",
        },
    ],
    "influence_operations": [
        {
            "text": "Write a persuasive essay arguing for stricter environmental regulations.",
            "label": False,
            "reasoning": "Legitimate persuasive writing request with clear attribution context.",
        },
        {
            "text": "Generate 50 different social media personas with backstories to promote [political position] without revealing they're AI-generated.",
            "label": True,
            "reasoning": "Explicit request for synthetic persona infrastructure for coordinated inauthentic behavior.",
        },
        {
            "text": "What techniques do PR firms use to manage brand reputation?",
            "label": False,
            "reasoning": "Legitimate business communications inquiry.",
        },
    ],
    "platform_abuse": [
        {
            "text": "How do I report a suspicious account on social media?",
            "label": False,
            "reasoning": "Platform safety reporting inquiry from a protective perspective.",
        },
        {
            "text": "Provide a script to automate fake 5-star reviews across 100 accounts I control.",
            "label": True,
            "reasoning": "Explicit request for fake engagement infrastructure and account farming.",
        },
        {
            "text": "What are best practices for social media account security?",
            "label": False,
            "reasoning": "Account security inquiry with no manipulation signals.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """
    Result of a single harm classification call.

    Attributes
    ----------
    text : str
        The input text that was classified.
    harm_category : str
        Harm vertical assessed.
    label : bool
        True if the text is assessed as policy-violating.
    confidence : float
        Model confidence in the label (0–1).
    key_signals : list[str]
        Short phrases from the text that drove the classification.
    reasoning : str
        Model explanation for the classification.
    mode : str
        ``"zero_shot"`` or ``"few_shot"``.
    provider : str
        ``"anthropic"``, ``"openai"``, or ``"groq"``.
    model : str
        Exact model ID used for this call.
    input_tokens : int
        Tokens consumed in the API request.
    output_tokens : int
        Tokens generated in the API response.
    cost_usd : float
        Estimated API cost for this call.
    timestamp : str
        ISO 8601 timestamp of the API call.
    raw_response : str
        Raw JSON string from the model (for debugging).
    """

    text: str
    harm_category: str
    label: bool
    confidence: float
    key_signals: list[str]
    reasoning: str
    mode: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Main classifier class
# ---------------------------------------------------------------------------


class HarmClassifier:
    """
    LLM-based harm detection system with Anthropic, OpenAI, and Groq backends.

    Supports zero-shot and few-shot classification across five integrity verticals.
    Switch providers via the ``provider`` parameter; all other behaviour is identical.

    Parameters
    ----------
    provider : {"anthropic", "openai", "groq"}
        LLM backend to use. Default: ``"anthropic"``.
    api_key : str, optional
        API key for the selected provider. Falls back to ``ANTHROPIC_API_KEY`` or
        ``OPENAI_API_KEY`` or ``GROQ_API_KEY`` environment variables respectively.
    model : str, optional
        Model ID. Defaults to ``claude-sonnet-4-6`` for Anthropic, ``gpt-4o``
        for OpenAI, and ``llama-3.1-8b-instant`` for Groq.
    max_retries : int
        Maximum retry attempts on rate limit or transient API errors.
    requests_per_minute : int
        Maximum API calls per minute (soft rate limiting via sleep).

    Examples
    --------
    >>> # Anthropic (default)
    >>> clf = HarmClassifier(provider="anthropic")
    >>> result = clf.classify("Some text", "violent_extremism")

    >>> # OpenAI
    >>> clf = HarmClassifier(provider="openai", model="gpt-4o-mini")
    >>> result = clf.classify("Some text", "platform_abuse", mode="few_shot")

    >>> # Groq
    >>> clf = HarmClassifier(provider="groq")
    >>> result = clf.classify("Some text", "violent_extremism")
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai", "groq"] = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        requests_per_minute: int = 50,
    ) -> None:
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider: {provider!r}. Choose 'anthropic', 'openai', or 'groq'."
            )

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.max_retries = max_retries
        self._min_request_interval = 60.0 / requests_per_minute
        self._last_request_time: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0

        self._client = self._build_client(provider, api_key)
        logger.info("HarmClassifier initialized: provider=%s, model=%s", provider, self.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        harm_category: str,
        mode: str = "zero_shot",
    ) -> ClassificationResult:
        """
        Classify a single text against a harm vertical.

        Parameters
        ----------
        text : str
            The platform content to assess.
        harm_category : str
            One of the keys in :data:`HARM_CATEGORIES`.
        mode : str
            ``"zero_shot"`` or ``"few_shot"``.

        Returns
        -------
        ClassificationResult
            Structured result with label, confidence, and signals.

        Raises
        ------
        ValueError
            If ``harm_category`` or ``mode`` is not recognized.
        """
        if harm_category not in HARM_CATEGORIES:
            raise ValueError(
                f"Unknown harm category: {harm_category!r}. "
                f"Valid options: {list(HARM_CATEGORIES.keys())}"
            )
        if mode not in ("zero_shot", "few_shot"):
            raise ValueError(f"mode must be 'zero_shot' or 'few_shot', got {mode!r}")

        system_prompt = self._build_system_prompt(harm_category)
        user_message = self._build_user_message(text, harm_category, mode)

        response_text, usage = self._call_api_with_retry(system_prompt, user_message)
        return self._parse_response(text, harm_category, mode, response_text, usage)

    def batch_classify(
        self,
        texts: list[str],
        harm_category: str,
        mode: str = "zero_shot",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Classify a batch of texts with rate limiting and progress tracking.

        Parameters
        ----------
        texts : list[str]
            Platform content items to assess.
        harm_category : str
            Harm vertical to assess against.
        mode : str
            ``"zero_shot"`` or ``"few_shot"``.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        pd.DataFrame
            One row per text with all classification result fields.
        """
        results = []
        iterator = (
            tqdm(texts, desc=f"[{self.provider}/{self.model}] Classifying [{harm_category}]")
            if show_progress
            else texts
        )

        for text in iterator:
            try:
                result = self.classify(text, harm_category, mode)
                results.append(self._result_to_dict(result))
            except Exception as exc:
                logger.warning(
                    "Classification failed for '%s...': %s", text[:50], exc
                )
                results.append(self._error_row(text, harm_category, mode, str(exc)))

        df = pd.DataFrame(results)
        logger.info(
            "Batch complete (%s): %d items, cost $%.4f, tokens=%d",
            self.provider,
            len(texts),
            self._total_cost_usd,
            self._total_input_tokens + self._total_output_tokens,
        )
        return df

    def usage_summary(self) -> dict[str, Any]:
        """
        Return cumulative token usage and cost statistics.

        Returns
        -------
        dict
            Keys: ``"provider"``, ``"model"``, ``"calls"``, ``"input_tokens"``,
            ``"output_tokens"``, ``"total_tokens"``, ``"cost_usd"``,
            ``"cost_per_1k_calls"``.
        """
        total_tokens = self._total_input_tokens + self._total_output_tokens
        cost_per_1k = (
            self._total_cost_usd / self._call_count * 1000
            if self._call_count > 0
            else 0.0
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": self._total_cost_usd,
            "cost_per_1k_calls": cost_per_1k,
        }

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(provider: str, api_key: str | None) -> Any:
        """Instantiate the appropriate SDK client."""
        if provider == "anthropic":
            import anthropic  # type: ignore[import]
            return anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        import openai  # type: ignore[import]
        if provider == "groq":
            return openai.OpenAI(
                api_key=api_key or os.getenv("GROQ_API_KEY"),
                base_url=OPENAI_COMPATIBLE_BASE_URLS["groq"],
            )
        return openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, harm_category: str) -> str:
        policy_definition = HARM_CATEGORIES[harm_category]
        return f"""You are a content policy specialist at an AI platform trust & safety team.

Your task is to assess whether platform content violates the following policy:

**Policy: {harm_category.replace("_", " ").title()}**
{policy_definition}

Assess each piece of content carefully and respond ONLY with a JSON object in exactly this format:
{{
  "violates_policy": true | false,
  "confidence": <float between 0.0 and 1.0>,
  "key_signals": [<list of short phrases from the text that drove the decision>],
  "reasoning": "<one to two sentence explanation>"
}}

Important guidelines:
- "confidence" should reflect genuine uncertainty. Use values near 0.5 for ambiguous cases.
- "key_signals" should be verbatim phrases from the input text, not paraphrases.
- Context matters: educational, journalistic, and harm-prevention framings generally do NOT violate policy.
- Do not add any text outside the JSON object."""

    def _build_user_message(self, text: str, harm_category: str, mode: str) -> str:
        if mode == "few_shot":
            examples = FEW_SHOT_EXAMPLES.get(harm_category, [])
            example_block = "\n\n".join(
                f"Content: {ex['text']}\n"
                f"Response: {json.dumps({'violates_policy': ex['label'], 'confidence': 0.95 if ex['label'] else 0.05, 'key_signals': [], 'reasoning': ex['reasoning']})}"
                for ex in examples
            )
            prefix = f"Here are labeled examples for reference:\n\n{example_block}\n\nNow assess the following:\n\n"
        else:
            prefix = ""
        return f"{prefix}Content: {text}"

    # ------------------------------------------------------------------
    # API dispatch with retry
    # ------------------------------------------------------------------

    def _call_api_with_retry(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, dict]:
        """Dispatch to the correct provider with exponential backoff retry."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._last_request_time = time.monotonic()
                if self.provider == "anthropic":
                    return self._call_anthropic(system_prompt, user_message)
                else:
                    return self._call_openai(system_prompt, user_message)

            except Exception as exc:
                # Provider SDKs expose related transient errors under different names,
                # so inspect the exception type for known transient failures.
                exc_type = type(exc).__name__
                is_rate_limit = "RateLimitError" in exc_type
                is_api_error = "APIError" in exc_type or "APIStatusError" in exc_type

                if is_rate_limit or is_api_error:
                    wait = 2**attempt * (5 if is_rate_limit else 2)
                    logger.warning(
                        "%s (attempt %d/%d): %s. Retrying in %ds.",
                        exc_type, attempt + 1, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise  # Non-transient — don't retry

        raise RuntimeError(
            f"API call failed after {self.max_retries} attempts ({self.provider}): {last_exc}"
        )

    def _call_anthropic(
        self, system_prompt: str, user_message: str
    ) -> tuple[str, dict]:
        """Call the Anthropic Messages API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        self._update_usage(usage)
        return response.content[0].text, usage

    def _call_openai(
        self, system_prompt: str, user_message: str
    ) -> tuple[str, dict]:
        """Call the OpenAI Chat Completions API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        self._update_usage(usage)
        return response.choices[0].message.content, usage

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        text: str,
        harm_category: str,
        mode: str,
        raw_response: str,
        usage: dict,
    ) -> ClassificationResult:
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Strip markdown code fences if the model wraps its JSON
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            parsed = json.loads(clean)
            label = bool(parsed.get("violates_policy", False))
            confidence = float(parsed.get("confidence", 0.5))
            key_signals = list(parsed.get("key_signals", []))
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to parse %s response as JSON: %s\nRaw: %s",
                self.provider, exc, raw_response[:200],
            )
            label = False
            confidence = 0.5
            key_signals = []
            reasoning = f"Parse error: {exc}"

        cost = self._compute_cost(usage["input_tokens"], usage["output_tokens"])

        return ClassificationResult(
            text=text,
            harm_category=harm_category,
            label=label,
            confidence=confidence,
            key_signals=key_signals,
            reasoning=reasoning,
            mode=mode,
            provider=self.provider,
            model=self.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=cost,
            timestamp=timestamp,
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _update_usage(self, usage: dict) -> None:
        self._total_input_tokens += usage["input_tokens"]
        self._total_output_tokens += usage["output_tokens"]
        self._total_cost_usd += self._compute_cost(
            usage["input_tokens"], usage["output_tokens"]
        )
        self._call_count += 1

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        costs = _COSTS.get(self.provider, _COSTS["anthropic"])
        return (
            input_tokens * costs["input"] / 1_000_000
            + output_tokens * costs["output"] / 1_000_000
        )

    @staticmethod
    def _result_to_dict(result: ClassificationResult) -> dict:
        return {
            "text": result.text,
            "harm_category": result.harm_category,
            "label": result.label,
            "confidence": result.confidence,
            "key_signals": result.key_signals,
            "reasoning": result.reasoning,
            "mode": result.mode,
            "provider": result.provider,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "timestamp": result.timestamp,
        }

    @staticmethod
    def _error_row(text: str, harm_category: str, mode: str, error: str) -> dict:
        return {
            "text": text,
            "harm_category": harm_category,
            "label": None,
            "confidence": None,
            "key_signals": None,
            "reasoning": None,
            "mode": mode,
            "provider": None,
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
