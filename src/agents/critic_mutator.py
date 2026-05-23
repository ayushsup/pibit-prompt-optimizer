"""
Agentic components of the optimization pipeline.

Extractor : Produces structured JSON from raw document text.
Critic    : Performs semantic diff between prediction and gold standard,
            identifying specific failure modes with actionable labels.
Mutator   : Prompt engineer agent that synthesizes critiques into a
            non-regressive prompt improvement, maintaining a rejection
            memory to avoid re-proposing failed variants.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor(BaseAgent):
    """Executes the current prompt against a document to produce JSON."""

    _SYSTEM_TEMPLATE = """{current_prompt}

TARGET SCHEMA:
{schema}

CRITICAL OUTPUT RULES:
- Return ONLY valid JSON. No markdown fences, no commentary, no preamble.
- Every key in the schema must appear in your output (use null or [] for absent values).
- Do not invent data not present in the document.
- Dates must follow ISO 8601 format where the schema specifies timestamps.
"""

    def extract(self, document_text: str, current_prompt: str, schema: str) -> str:
        system_prompt = self._SYSTEM_TEMPLATE.format(
            current_prompt=current_prompt,
            schema=schema,
        )
        return self.call_llm(
            system_prompt=system_prompt,
            user_prompt=f"DOCUMENT:\n{document_text}",
            role_name="Extractor",
            temperature=0.1,  # Near-deterministic for extraction
        )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(BaseAgent):
    """
    Analyses extraction failures and returns structured, actionable critiques.

    The output is designed to be directly consumable by the Mutator:
    each critique identifies a failure category (missing key, wrong type,
    hallucination, format error, etc.) plus the specific field and evidence.
    """

    _SYSTEM_PROMPT = """You are a precision data-extraction auditor.

Your job is to compare a JSON extraction against the gold-standard annotation
and diagnose exactly why they differ.

For EACH discrepancy, output a critique in this exact format:
  [FIELD: <field_path>] [TYPE: <MISSING|WRONG_VALUE|TYPE_MISMATCH|HALLUCINATED|FORMAT_ERROR|ARRAY_MISMATCH>]
  Predicted: <value or 'ABSENT'>
  Gold:      <value>
  Fix:       <one-sentence actionable instruction for the prompt author>

Rules:
- Be surgical: do not mention fields that match correctly.
- Focus on the top 5 most impactful failures if there are many.
- Do NOT suggest model fine-tuning or data changes — only prompt wording fixes.
- If the extraction is perfect, output exactly: NO_FAILURES
"""

    def critique(
        self,
        document_text: str,
        predicted_json: str,
        gold_json: str,
    ) -> str:
        user_prompt = (
            f"DOCUMENT (first 1500 chars):\n{document_text[:1500]}\n\n"
            f"PREDICTED JSON:\n{predicted_json}\n\n"
            f"GOLD STANDARD JSON:\n{gold_json}\n\n"
            f"List all discrepancies."
        )
        return self.call_llm(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=user_prompt,
            role_name="Critic",
            temperature=0.1,
        )


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------

class Mutator(BaseAgent):
    """
    Automated prompt engineer.

    Strategy:
    1. Receives the current best prompt + a batch of structured critiques.
    2. Maintains a memory of rejected prompts to avoid re-proposing failures.
    3. Detects stalls (repeated patterns) and escalates to a bolder mutation.
    4. Returns ONLY the new prompt text — no preamble, no explanation.
    """

    _SYSTEM_PROMPT = """You are a world-class prompt engineer specializing in structured JSON extraction.

Your task: rewrite the given extraction prompt to fix all listed failure modes WITHOUT
degrading performance on fields that currently work correctly.

Chain-of-thought process (internal only — do NOT include in output):
1. Categorise each critique by failure type (missing, wrong value, format, etc.)
2. Identify which prompt instructions are absent or unclear.
3. Draft targeted rule additions or clarifications.
4. Verify your new rules don't conflict with working parts of the current prompt.
5. Write the final prompt.

OUTPUT FORMAT: Return ONLY the final prompt text. No labels, no markdown, no explanation.
The prompt must be self-contained — it will be sent directly to the extraction model.
"""

    def mutate(
        self,
        current_prompt: str,
        critiques: list[str],
        rejected_prompts: list[str] | None = None,
        stall_count: int = 0,
    ) -> str:
        """
        Propose an improved prompt.

        Parameters
        ----------
        current_prompt  : The currently accepted best prompt.
        critiques       : List of Critic outputs from failed documents.
        rejected_prompts: Prompts already tried and rejected (last 5 shown).
        stall_count     : Consecutive iterations with no improvement.
                          If ≥ 3, instructs the mutator to try a bolder change.
        """
        critique_block = "\n\n---\n\n".join(
            f"Critique {i + 1}:\n{c}" for i, c in enumerate(critiques)
        )

        rejection_block = ""
        if rejected_prompts:
            recent = rejected_prompts[-5:]
            rejection_block = (
                "\n\nREJECTED PROMPTS (do NOT re-propose these — they scored worse):\n"
                + "\n\n".join(f"[REJECTED {i+1}]:\n{p}" for i, p in enumerate(recent))
            )

        stall_note = ""
        if stall_count >= 3:
            stall_note = (
                f"\n\n⚠️  STALL DETECTED: The score has not improved for {stall_count} "
                f"consecutive iterations. Try a significantly different structure, "
                f"add concrete field-by-field extraction rules, or include a worked example."
            )

        user_prompt = (
            f"CURRENT BEST PROMPT:\n{current_prompt}\n\n"
            f"FAILURE CRITIQUES FROM VALIDATION SET:\n{critique_block}"
            f"{rejection_block}"
            f"{stall_note}\n\n"
            f"Write the improved prompt now."
        )

        return self.call_llm(
            system_prompt=self._SYSTEM_PROMPT,
            user_prompt=user_prompt,
            role_name="Mutator",
            temperature=0.4,  # Some creativity for mutation
        )