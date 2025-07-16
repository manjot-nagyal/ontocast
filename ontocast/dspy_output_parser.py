"""DSPy-based replacement for LangChain's PydanticOutputParser.

This module provides a drop-in replacement that matches the (tiny) surface
API used inside OntoCast (`get_format_instructions` + `parse`). It avoids
the LangChain dependency for output parsing and instead relies on `dspy-ai`.

Key behaviour:
* `get_format_instructions` – returns a prompt snippet instructing the LLM to
  output **valid JSON** conforming to the target Pydantic schema. We embed the
  schema itself so the model has an exact contract.
* `parse` – attempts to load the text as JSON and then validate via Pydantic.
  If JSON decoding fails or validation fails, we invoke DSPy to *self-repair*:
  we pass the malformed text + schema to the LLM and ask it to return a fixed
  JSON object. The repaired result is then validated again.

The class is intentionally lightweight and does not try to replicate every
knob of LangChain's original parser. It is sufficient for OntoCast, which
only uses the two public methods mentioned above.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Type, TypeVar

import dspy
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DspyOutputParser:  # noqa: N801 – match LC naming for drop-in replacement
    """DSPy-based parser compatible with the old LangChain API."""

    def __init__(self, pydantic_object: Type[T]):
        self._schema: Type[T] = pydantic_object
        _ensure_dspy_configured()

    # ---------------------------------------------------------------------
    # Public API – mirrors LangChain's PydanticOutputParser
    # ---------------------------------------------------------------------
    def get_format_instructions(self) -> str:  # pragma: no cover – trivial
        """Return prompt instructions embedding the JSON schema.

        OntoCast concatenates this string to its template so the LLM knows the
        output contract.
        """
        schema_json = json.dumps(self._schema.model_json_schema(), indent=2)
        return (
            "You MUST respond with **valid JSON** matching **exactly** this "
            "schema (no extra keys, no markdown, no comments):\n" + schema_json
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove ``` and language tags that often wrap JSON blocks."""
        fence_pattern = re.compile(r"^```[a-zA-Z]*\n|```$", re.MULTILINE)
        cleaned = fence_pattern.sub("", raw).strip()
        return cleaned

    def parse(self, text: str) -> T:
        """Parse text into the Pydantic model, repairing via DSPy if needed."""
        text = self._strip_fences(text)
        try:
            data = json.loads(text)
            return self._schema.model_validate(data)
        except Exception as e:
            # Log the original error and attempt self-repair with DSPy
            logging.warning(f"Initial JSON parse failed: {e}")
            logging.debug(f"Raw text (first 500 chars): {text[:500]}...")
            repaired_text = _repair_with_dspy(text, self._schema)
            if not repaired_text:
                raise RuntimeError("DSPy repair failed, returned empty result.")

            repaired_text = self._strip_fences(
                json.dumps(repaired_text)
            )  # ensure string
            repaired_json = json.loads(repaired_text)
            return self._schema.model_validate(repaired_json)


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------


def _ensure_dspy_configured() -> None:  # pragma: no cover – side-effecty
    """Configure DSPy with Gemini backend once per process."""
    if getattr(dspy.settings, "_ontocast_configured", False):
        return

    # DSPy relies on LiteLLM under the hood, which picks up *_API_KEY env vars.
    llm_model = os.getenv("LLM_MODEL_NAME")
    llm_provider = "gemini"
    llm_api_key = os.getenv("LLM_API_KEY")

    dspy.configure(
        lm=dspy.LM(
            llm_provider + "/" + llm_model, api_key=llm_api_key, max_tokens=30000
        )
    )
    # Mark so we don't re-configure on every instantiation
    setattr(dspy.settings, "_ontocast_configured", True)


def _repair_with_dspy(bad_text: str, schema: Type[T]) -> dict:
    """Ask DSPy/Gemini to produce valid JSON from malformed text."""

    class RepairModule(dspy.Signature):  # type: ignore[misc]
        """Dynamic signature: freeform -> JSON"""

        bad_output: str = dspy.InputField(desc="The malformed model output.")
        fixed_json: str = dspy.OutputField(
            desc="Valid JSON that conforms exactly to the provided schema."
        )

    # Embed schema so the model has the contract.
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    prompt = (
        """The following text should be valid JSON but is not or does not match
        the schema. Fix it so that it *strictly* conforms to the schema. You
        MUST add any missing keys with sensible default values so the JSON
        exactly matches the schema. If a numerical field is unknown, output
        0.0."""
        + "\n\nSchema:\n"
        + schema_json
        + "\n\nBad output:\n"
        + bad_text
    )

    repair = dspy.Predict(RepairModule)  # one-shot predict
    result = repair(bad_output=prompt)

    # Attempt to parse the repaired JSON.
    cleaned = DspyOutputParser._strip_fences(result.fixed_json)
    try:
        return json.loads(cleaned)
    except Exception as exc:  # pragma: no cover – last-ditch safety
        logging.error(f"DSPy repair failed. Original text: {bad_text[:200]}...")
        logging.error(f"Repaired result: {result.fixed_json[:200]}...")
        raise ValueError("DSPy failed to repair JSON") from exc
