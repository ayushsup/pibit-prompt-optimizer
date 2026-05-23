"""
Central optimization engine.

Implements a Greedy Accept/Reject loop augmented with:
  - Schema-aware F1 scoring via Scorer (independent of this module)
  - Resumability: checks SQLite for an existing trajectory and warm-starts
  - Stall detection: escalates mutation aggressiveness after N no-improvement iterations
  - Rejection memory: Mutator sees previously rejected prompts
  - Budget enforcement: stops on max_iterations OR max_cost_dollars (if > 0)
  - Daily-limit detection: graceful shutdown when OpenRouter quota is exhausted
  - REPORT.md auto-generation: writes the final report after test evaluation

Dataset path handling uses os.path.join throughout for Windows/Unix compatibility.

Val-set safety guard
--------------------
If the val split is empty (tiny dataset edge case), the loop automatically
falls back to using all loaded docs for validation with a clear warning.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.agents.base_agent import DailyLimitError
from src.agents.critic_mutator import Critic, Extractor, Mutator
from src.core.config_parser import load_config
from src.core.state_manager import StateManager
from src.data.loader import ExtractBenchLoader
from src.data.splitter import deterministic_split
from src.evaluation.scorer import Scorer
from src.optimizer.diff_viewer import DiffViewer


class OptimizerLoop:
    """Budget-enforced greedy prompt optimization loop."""

    STALL_THRESHOLD = 3  # Iterations without improvement before escalating mutation

    def __init__(self, config_path: str = "config/base_config.yaml"):
        self.config = load_config(config_path)
        self.state = StateManager()
        self.diff_viewer = DiffViewer()

        cfg = self.config

        self.extractor = Extractor(cfg.models.extractor, self.state)
        self.critic    = Critic(cfg.models.critic,    self.state)
        self.mutator   = Mutator(cfg.models.mutator,  self.state)

        # Build LLM judge callable for stochastic metrics (string_semantic / array_llm)
        judge_client = self.critic.client
        judge_model  = cfg.models.critic

        def llm_judge(pred, gold, metric: str) -> float:
            return self._call_judge(pred, gold, metric, judge_client, judge_model)

        self.scorer = Scorer(state_manager=self.state, judge_callable=llm_judge)

        # ---- Dataset loading ----
        dataset_base = os.path.join(cfg.dataset.base_path, "dataset")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        loader = ExtractBenchLoader(
            base_path=dataset_base,
            schema_name=cfg.dataset.name,
            vision_model=cfg.vision_model,
            openrouter_key=openrouter_key,
        )
        print(f"\n📂 Loading dataset: {cfg.dataset.name}")
        all_docs = loader.load_all_document_pairs()

        if not all_docs:
            raise RuntimeError(
                "No document pairs could be loaded.\n"
                "Check that:\n"
                "  1. data/extract-bench/dataset/<schema_name>/pdf+gold/ exists.\n"
                "  2. The schema name in config matches the folder exactly.\n"
                "  3. For scanned PDFs: set vision_model in config (uses OPENROUTER_API_KEY)."
            )

        self.train_docs, self.val_docs, self.test_docs = deterministic_split(
            all_docs,
            seed=cfg.dataset.split_seed,
            train_ratio=cfg.dataset.train_ratio,
            val_ratio=cfg.dataset.val_ratio,
        )

        # ---- Val-set safety guard ----
        if not self.val_docs:
            print(
                "\n  ⚠️  Val set is empty after split. "
                "Falling back to using ALL loaded docs for validation.\n"
                "  (Normal on tiny datasets — the optimizer will still run correctly.)"
            )
            self.val_docs  = all_docs
            self.test_docs = all_docs

        print(
            f"\n  Split (seed={cfg.dataset.split_seed}): "
            f"{len(self.train_docs)} train | "
            f"{len(self.val_docs)} val | "
            f"{len(self.test_docs)} test"
        )

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def _within_budget(self) -> bool:
        dollar_limit = self.config.budget.max_cost_dollars
        if dollar_limit <= 0.0:
            return True  # 0 means "unlimited" (free tier)
        spent = self.state.get_total_cost()
        if spent >= dollar_limit:
            print(f"💸 Budget exhausted (${spent:.4f} / ${dollar_limit:.2f}). Stopping.")
            return False
        return True

    # ------------------------------------------------------------------
    # Corpus scoring
    # ------------------------------------------------------------------

    def _evaluate_corpus(
        self, docs: List[Dict], prompt: str
    ) -> Tuple[float, Dict]:
        """
        Extract + score every document in `docs` with the current prompt.
        Returns (mean_f1, info_dict).

        Raises DailyLimitError if the API quota is exhausted during extraction.
        """
        total_f1  = 0.0
        breakdown: Dict[str, Dict] = {}
        failed:    List[Dict]       = []

        for doc in docs:
            prediction = self.extractor.extract(doc["text"], prompt, doc["schema"])
            f1, doc_breakdown = self.scorer.score_document(
                pred_json=prediction,
                gold_json=doc["gold_json"],
                schema_str=doc["schema"],
            )
            total_f1 += f1
            breakdown[doc["id"]] = {**doc_breakdown, "f1": f1}

            if f1 < 1.0:
                failed.append({
                    "doc":   doc["text"],
                    "pred":  prediction,
                    "gold":  doc["gold_json"],
                    "score": f1,
                    "id":    doc["id"],
                })

        mean_f1 = total_f1 / max(len(docs), 1)
        return mean_f1, {"docs": breakdown, "failed": failed}

    # ------------------------------------------------------------------
    # LLM judge for stochastic metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _call_judge(pred, gold, metric: str, client, model: str) -> float:
        try:
            if metric == "string_semantic":
                prompt = (
                    "Rate the semantic equivalence of these two strings.\n"
                    "0.0 = completely different meaning, 1.0 = identical meaning.\n"
                    f"String A: {pred}\nString B: {gold}\n"
                    "Return ONLY a single float between 0.0 and 1.0."
                )
            else:  # array_llm
                prompt = (
                    "Rate the semantic equivalence of these two arrays.\n"
                    "0.0 = completely different content, 1.0 = semantically identical.\n"
                    f"Array A: {json.dumps(pred)}\nArray B: {json.dumps(gold)}\n"
                    "Return ONLY a single float between 0.0 and 1.0."
                )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            return max(0.0, min(1.0, float(response.choices[0].message.content.strip())))
        except Exception:
            return 1.0 if str(pred).strip().lower() == str(gold).strip().lower() else 0.0

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config
        print(
            f"\n🚀 Optimizer starting  "
            f"max_iterations={cfg.budget.max_iterations}  "
            f"dataset={cfg.dataset.name}"
        )

        # ---- Resumability: warm-start from a previous interrupted run ----
        prior_best   = self.state.get_best_state()
        last_done    = self.state.get_last_completed_iteration()
        rejected_history: List[str] = self.state.get_rejected_prompts()

        if prior_best and last_done >= 0:
            best_prompt    = prior_best["prompt"]
            best_score     = prior_best["val_score"]
            start_iteration = last_done + 1
            current_prompt  = best_prompt
            print(
                f"♻️  Resuming from iteration {start_iteration} "
                f"(best val F1 so far: {best_score:.4f})"
            )
        else:
            best_prompt    = cfg.seed_prompt
            best_score     = -1.0  # Force first iteration to be accepted as baseline
            start_iteration = 0
            current_prompt  = cfg.seed_prompt
            print("🌱 Starting fresh from seed prompt.")

        stall_count   = 0
        seed_test_score: Optional[float] = None

        try:
            for iteration in range(start_iteration, cfg.budget.max_iterations):
                if not self._within_budget():
                    break

                print(f"\n{'='*60}")
                print(f"  ITERATION {iteration + 1} / {cfg.budget.max_iterations}")
                print(f"{'='*60}")

                # ---- Evaluate current prompt on validation set ----
                val_score, val_info = self._evaluate_corpus(self.val_docs, current_prompt)
                failed_docs = val_info["failed"]

                print(
                    f"  📊 Val F1: {val_score:.4f}  |  "
                    f"Best so far: {max(best_score, 0.0):.4f}  |  "
                    f"Failed docs: {len(failed_docs)}/{len(self.val_docs)}"
                )

                # Print per-doc subtree breakdown for observability
                for doc_id, info in val_info["docs"].items():
                    subtrees = info.get("subtrees", {})
                    if subtrees:
                        field_scores = "  ".join(
                            f"{k}={v['f1']:.2f}" for k, v in subtrees.items()
                        )
                        print(f"    ↳ {doc_id}: [{field_scores}]")

                # ---- Accept / Reject ----
                accepted = val_score > best_score

                if accepted:
                    self.diff_viewer.generate_diff(best_prompt, current_prompt, iteration)
                    best_score  = val_score
                    best_prompt = current_prompt
                    stall_count = 0
                    print(f"  🏆 New best prompt accepted! (F1={val_score:.4f})")
                else:
                    stall_count += 1
                    print(f"  ❌ Rejected (no improvement). Stall count: {stall_count}.")
                    if current_prompt not in rejected_history:
                        rejected_history.append(current_prompt)
                    current_prompt = best_prompt  # rollback

                self.state.log_iteration(
                    iteration=iteration,
                    prompt=current_prompt,
                    val_score=val_score,
                    accepted=accepted,
                    breakdown={
                        "per_doc_f1": {
                            doc_id: info["f1"]
                            for doc_id, info in val_info["docs"].items()
                        }
                    },
                )

                # ---- Early stop on perfect score ----
                if val_score >= 1.0:
                    print("  ✅ Perfect validation score achieved. Halting early.")
                    break

                # ---- Generate critiques from worst-performing documents ----
                if not self._within_budget():
                    break

                if not failed_docs:
                    print("  ℹ️  No failed documents — nothing to critique.")
                    continue

                # Critique the worst failures first
                critique_docs = sorted(failed_docs, key=lambda x: x["score"])[:3]
                critiques: List[str] = []

                for fail in critique_docs:
                    try:
                        critique = self.critic.critique(
                            fail["doc"], fail["pred"], fail["gold"]
                        )
                        critiques.append(critique)
                        print(f"  🔍 Critique generated for: {fail['id']}")
                    except DailyLimitError:
                        raise
                    except Exception as exc:
                        print(f"  ⚠️  Critic failed for '{fail.get('id', '?')}': {exc}")

                if not critiques:
                    print("  ⚠️  All critique attempts failed. Skipping mutation.")
                    continue

                # ---- Mutate ----
                if not self._within_budget():
                    break

                try:
                    current_prompt = self.mutator.mutate(
                        current_prompt=best_prompt,
                        critiques=critiques,
                        rejected_prompts=rejected_history,
                        stall_count=stall_count,
                    )
                    print(f"  ✏️  Mutator drafted a new prompt proposal.")
                except DailyLimitError:
                    raise
                except Exception as exc:
                    print(f"  ⚠️  Mutator failed: {exc}. Retaining current best.")
                    current_prompt = best_prompt

        except DailyLimitError:
            print(
                "\n🚫 Daily API quota exhausted. Run state persisted — "
                "re-run tomorrow to continue from this checkpoint."
            )

        # ------------------------------------------------------------------
        # Final test set evaluation (held-out, evaluated ONCE)
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print("  🧪 FINAL HELD-OUT TEST EVALUATION")
        print(f"{'='*60}")

        try:
            # Score seed prompt on test set for the report
            print("  Evaluating seed prompt on test set...")
            seed_test_score, seed_test_info = self._evaluate_corpus(
                self.test_docs, cfg.seed_prompt
            )
            print(f"  🌱 Seed Test F1: {seed_test_score:.4f}")

            print("  Evaluating final (best) prompt on test set...")
            test_score, test_info = self._evaluate_corpus(self.test_docs, best_prompt)
            print(f"\n  ✅ Final Test F1: {test_score:.4f}")
            print(f"  Best Val F1    : {max(best_score, 0.0):.4f}")

        except DailyLimitError:
            print("  ⚠️  Quota exhausted during final evaluation. Partial results only.")
            test_score   = 0.0
            test_info    = {"docs": {}, "failed": []}
            seed_test_score = 0.0
            seed_test_info  = {"docs": {}, "failed": []}

        print("\n  Per-document breakdown:")
        for doc_id, info in test_info["docs"].items():
            subtrees = info.get("subtrees", {})
            field_scores = "  ".join(
                f"{k}={v['f1']:.2f}" for k, v in subtrees.items()
            ) if subtrees else "n/a"
            print(f"    {doc_id}: F1={info['f1']:.4f}  [{field_scores}]")

        print(f"\n  Diffs logged : logs/diffs/")
        print(f"  Audit trail  : run_state.db")
        print(f"{'='*60}\n")

        # ---- Generate REPORT.md ----
        self._write_report(
            seed_prompt=cfg.seed_prompt,
            best_prompt=best_prompt,
            seed_test_score=seed_test_score or 0.0,
            final_test_score=test_score,
            best_val_score=max(best_score, 0.0),
            test_info=test_info,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _write_report(
        self,
        seed_prompt: str,
        best_prompt: str,
        seed_test_score: float,
        final_test_score: float,
        best_val_score: float,
        test_info: Dict,
    ) -> None:
        """Write REPORT.md with scores, trajectory, diffs, and limitations."""
        trajectory = self.state.get_trajectory()
        accepted_iterations = [t for t in trajectory if t["accepted"]]

        # Build subtree breakdown table
        subtree_rows = []
        for doc_id, info in test_info.get("docs", {}).items():
            subtrees = info.get("subtrees", {})
            for field, fscore in subtrees.items():
                subtree_rows.append(
                    f"| {doc_id} | {field} | {fscore['precision']:.3f} | "
                    f"{fscore['recall']:.3f} | {fscore['f1']:.3f} |"
                )

        subtree_table = "\n".join(subtree_rows) if subtree_rows else "| — | — | — | — | — |"

        # Score curve
        score_curve = "\n".join(
            f"| {t['iteration'] + 1:>3} | {t['val_score']:.4f} | {'✅' if t['accepted'] else '❌'} |"
            for t in trajectory
        )

        # Notable accepted mutations
        notable = []
        for i, t in enumerate(accepted_iterations[1:], 1):  # skip iteration 0 (seed)
            notable.append(
                f"- **Iteration {t['iteration'] + 1}** — Val F1: {t['val_score']:.4f}"
            )
        notable_str = "\n".join(notable) if notable else "- No mutations improved over the seed."

        # Prompt diff summary
        if seed_prompt.strip() == best_prompt.strip():
            diff_summary = "The seed prompt was not improved during this run."
        else:
            diff_summary = (
                f"The final prompt differs from the seed. See `logs/diffs/` for "
                f"unified diffs of each accepted mutation."
            )

        report = textwrap.dedent(f"""\
            # Prompt Optimization Report

            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Dataset:** `{self.config.dataset.name}`
            **Models:** extractor=`{self.config.models.extractor}`  critic=`{self.config.models.critic}`  mutator=`{self.config.models.mutator}`

            ---

            ## 1. Test-Set Scores

            | Prompt  | Test F1 |
            |---------|---------|
            | Seed    | {seed_test_score:.4f} |
            | Final   | {final_test_score:.4f} |
            | **Δ**   | **{final_test_score - seed_test_score:+.4f}** |

            Best validation F1 achieved: **{best_val_score:.4f}**

            ---

            ## 2. Per-Subtree Breakdown (Final Prompt, Test Set)

            | Document | Field | Precision | Recall | F1 |
            |----------|-------|-----------|--------|----|
            {subtree_table}

            ---

            ## 3. Optimization Trajectory

            | Iter | Val F1 | Accepted |
            |------|--------|----------|
            {score_curve}

            ---

            ## 4. Notable Accepted Mutations

            {notable_str}

            ---

            ## 5. Seed Prompt

            ```
            {seed_prompt.strip()}
            ```

            ---

            ## 6. Final Prompt

            ```
            {best_prompt.strip()}
            ```

            ---

            ## 7. Diff Summary

            {diff_summary}

            ---

            ## 8. Limitations

            - **Small dataset:** With only a few documents per schema, validation scores are noisy
              and there is a risk of overfitting the prompt to the validation document(s).
            - **Positional array alignment:** Object arrays are compared positionally; if the
              predicted ordering differs from gold, items at each position are penalised even
              when the content is correct.
            - **Free-tier rate limits:** OpenRouter free models have a daily request cap (~50/day),
              constraining how many iterations can run. A paid plan would allow full 20-iteration runs.
            - **No train split used:** The greedy loop currently uses only the validation set for
              feedback. Train documents are loaded but not yet used for few-shot example selection.
            - **Stochastic judge caching:** `string_semantic` and `array_llm` scores are cached per
              (pred, gold) pair, but the initial LLM judge call for novel pairs is non-deterministic.
            """)

        with open("REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("  📝 REPORT.md written.")
