"""
Unit tests for the scoring subsystem.

Covers:
  - All six evaluation metric types (string_exact, integer_exact,
    number_tolerance, boolean_exact, string_semantic, array_llm)
  - ScoreResult accumulation and F1 arithmetic
  - Scorer.score_document on synthetic schema+gold+pred triples
  - ExtractBench schema_definition envelope unwrapping
  - anyOf field handling (nullable fields, polymorphic types)
  - additionalProperties object handling (grouped skills)
  - Array alignment policy (positional for objects, set-based for primitives)
  - Graceful handling of malformed predictions (empty JSON, parse errors)
  - Scorer.calculate_f1 edge cases (zero division)

Run with: pytest tests/test_scorer.py -v
"""

import json
import pytest

from src.evaluation.metrics import (
    boolean_exact,
    integer_exact,
    number_tolerance,
    string_exact,
    compute_cache_key,
)
from src.evaluation.scorer import ScoreResult, Scorer


# ---------------------------------------------------------------------------
# Deterministic metric tests
# ---------------------------------------------------------------------------

class TestStringExact:
    def test_identical(self):
        assert string_exact("Hello World", "Hello World") == 1.0

    def test_case_insensitive(self):
        assert string_exact("HELLO", "hello") == 1.0

    def test_whitespace_trimmed(self):
        assert string_exact("  hello  ", "hello") == 1.0

    def test_mismatch(self):
        assert string_exact("foo", "bar") == 0.0

    def test_non_string_pred(self):
        assert string_exact(123, "123") == 0.0

    def test_none_values(self):
        assert string_exact(None, "test") == 0.0
        assert string_exact("test", None) == 0.0


class TestIntegerExact:
    def test_exact(self):
        assert integer_exact(42, 42) == 1.0

    def test_coercion_from_string(self):
        assert integer_exact("42", 42) == 1.0

    def test_coercion_from_float(self):
        assert integer_exact(42.0, 42) == 1.0

    def test_mismatch(self):
        assert integer_exact(41, 42) == 0.0

    def test_invalid_pred(self):
        assert integer_exact("not-a-number", 42) == 0.0

    def test_none(self):
        assert integer_exact(None, 42) == 0.0


class TestNumberTolerance:
    def test_exact_match(self):
        assert number_tolerance(100.0, 100.0) == 1.0

    def test_within_tolerance(self):
        assert number_tolerance(104.9, 100.0, tolerance=0.05) == 1.0

    def test_outside_tolerance(self):
        assert number_tolerance(106.0, 100.0, tolerance=0.05) == 0.0

    def test_gold_zero_pred_zero(self):
        assert number_tolerance(0, 0) == 1.0

    def test_gold_zero_pred_nonzero(self):
        assert number_tolerance(1, 0) == 0.0

    def test_string_coercion(self):
        assert number_tolerance("100", 100) == 1.0

    def test_invalid_pred(self):
        assert number_tolerance("invalid", 100) == 0.0


class TestBooleanExact:
    def test_true_true(self):
        assert boolean_exact(True, True) == 1.0

    def test_false_false(self):
        assert boolean_exact(False, False) == 1.0

    def test_true_false(self):
        assert boolean_exact(True, False) == 0.0

    def test_string_true(self):
        assert boolean_exact("true", True) == 1.0

    def test_string_false(self):
        assert boolean_exact("false", False) == 1.0

    def test_integer_1(self):
        assert boolean_exact(1, True) == 1.0

    def test_integer_0(self):
        assert boolean_exact(0, False) == 1.0

    def test_mismatch_string(self):
        assert boolean_exact("false", True) == 0.0

    def test_none_pred(self):
        assert boolean_exact(None, True) == 0.0


# ---------------------------------------------------------------------------
# Cache key test
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        key1 = compute_cache_key("string_semantic", "hello", "world")
        key2 = compute_cache_key("string_semantic", "hello", "world")
        assert key1 == key2

    def test_different_metrics(self):
        k1 = compute_cache_key("string_semantic", "hello", "world")
        k2 = compute_cache_key("array_llm", "hello", "world")
        assert k1 != k2

    def test_order_matters(self):
        k1 = compute_cache_key("string_semantic", "a", "b")
        k2 = compute_cache_key("string_semantic", "b", "a")
        assert k1 != k2


# ---------------------------------------------------------------------------
# ScoreResult tests
# ---------------------------------------------------------------------------

class TestScoreResult:
    def test_perfect_score(self):
        r = ScoreResult(true_positives=5.0, predicted_count=5, gold_count=5)
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_zero_division(self):
        r = ScoreResult(true_positives=0.0, predicted_count=0, gold_count=0)
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_f1_formula(self):
        r = ScoreResult(true_positives=2.0, predicted_count=4, gold_count=4)
        assert abs(r.f1 - 0.5) < 1e-6

    def test_merge(self):
        a = ScoreResult(true_positives=3.0, predicted_count=4, gold_count=5)
        b = ScoreResult(true_positives=2.0, predicted_count=3, gold_count=3)
        a.merge(b)
        assert a.true_positives == 5.0
        assert a.predicted_count == 7
        assert a.gold_count == 8

    def test_to_dict_keys(self):
        r = ScoreResult(true_positives=1.0, predicted_count=1, gold_count=1)
        d = r.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d


# ---------------------------------------------------------------------------
# Scorer.calculate_f1
# ---------------------------------------------------------------------------

class TestScorerF1:
    def setup_method(self):
        self.scorer = Scorer()

    def test_perfect(self):
        assert self.scorer.calculate_f1(1.0, 1.0) == 1.0

    def test_zero_recall(self):
        assert self.scorer.calculate_f1(1.0, 0.0) == 0.0

    def test_zero_precision(self):
        assert self.scorer.calculate_f1(0.0, 1.0) == 0.0

    def test_both_zero(self):
        assert self.scorer.calculate_f1(0.0, 0.0) == 0.0

    def test_harmonic_mean(self):
        f1 = self.scorer.calculate_f1(0.75, 0.5)
        assert abs(f1 - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Scorer.score_document — basic schema
# ---------------------------------------------------------------------------

SCHEMA_SIMPLE = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "evaluation_config": "string_exact"},
        "age":  {"type": "integer", "evaluation_config": "integer_exact"},
    }
})


class TestScorerDocument:
    def setup_method(self):
        self.scorer = Scorer()

    def test_perfect_extraction(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Alice", "age": 30})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 == 1.0

    def test_wrong_value(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Bob", "age": 30})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 < 1.0

    def test_missing_field(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Alice"})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 < 1.0

    def test_malformed_prediction(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        f1, breakdown = self.scorer.score_document("not-json", gold, SCHEMA_SIMPLE)
        assert f1 == 0.0

    def test_json_with_fences(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = "```json\n" + json.dumps({"name": "Alice", "age": 30}) + "\n```"
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert f1 == 1.0

    def test_breakdown_contains_subtrees(self):
        gold = json.dumps({"name": "Alice", "age": 30})
        pred = json.dumps({"name": "Alice", "age": 30})
        _, breakdown = self.scorer.score_document(pred, gold, SCHEMA_SIMPLE)
        assert "subtrees" in breakdown
        assert "name" in breakdown["subtrees"]


# ---------------------------------------------------------------------------
# schema_definition envelope unwrapping (ExtractBench format)
# ---------------------------------------------------------------------------

SCHEMA_WRAPPED = json.dumps({
    "name": "Test Schema",
    "description": "A schema with the ExtractBench envelope",
    "schema_definition": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "evaluation_config": "string_exact"},
            "year":  {"type": "integer", "evaluation_config": "integer_exact"},
        }
    }
})


class TestSchemaEnvelopeUnwrapping:
    def setup_method(self):
        self.scorer = Scorer()

    def test_wrapped_schema_scores_correctly(self):
        gold = json.dumps({"title": "My Paper", "year": 2024})
        pred = json.dumps({"title": "My Paper", "year": 2024})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_WRAPPED)
        assert f1 == 1.0

    def test_wrapped_schema_partial(self):
        gold = json.dumps({"title": "My Paper", "year": 2024})
        pred = json.dumps({"title": "My Paper", "year": 2023})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_WRAPPED)
        assert 0.0 < f1 < 1.0

    def test_wrapped_schema_missing_field(self):
        gold = json.dumps({"title": "My Paper", "year": 2024})
        pred = json.dumps({"title": "My Paper"})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_WRAPPED)
        assert f1 < 1.0

    def test_wrapped_schema_does_not_return_zero_for_perfect_pred(self):
        """Regression: before the fix, wrapped schemas always returned F1=0."""
        gold = json.dumps({"title": "Test", "year": 2020})
        pred = json.dumps({"title": "Test", "year": 2020})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_WRAPPED)
        assert f1 > 0.0


# ---------------------------------------------------------------------------
# anyOf field handling
# ---------------------------------------------------------------------------

SCHEMA_ANY_OF = json.dumps({
    "type": "object",
    "properties": {
        "startDate": {
            "anyOf": [
                {"type": "string",  "evaluation_config": "string_exact"},
                {"type": "integer", "evaluation_config": "integer_exact"},
                {"type": "null"},
            ]
        },
        "isCurrent": {
            "anyOf": [
                {"type": "boolean", "evaluation_config": "boolean_exact"},
                {"type": "null"},
            ]
        },
    }
})


class TestAnyOf:
    def setup_method(self):
        self.scorer = Scorer()

    def test_any_of_integer_match(self):
        gold = json.dumps({"startDate": 2020, "isCurrent": False})
        pred = json.dumps({"startDate": 2020, "isCurrent": False})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ANY_OF)
        assert f1 == 1.0

    def test_any_of_string_match(self):
        gold = json.dumps({"startDate": "Spring 2020", "isCurrent": True})
        pred = json.dumps({"startDate": "Spring 2020", "isCurrent": True})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ANY_OF)
        assert f1 == 1.0

    def test_any_of_null_gold_not_counted(self):
        """Null gold values should contribute 0 to gold_count."""
        gold = json.dumps({"startDate": None, "isCurrent": True})
        pred = json.dumps({"startDate": "something", "isCurrent": True})
        f1, breakdown = self.scorer.score_document(pred, gold, SCHEMA_ANY_OF)
        # isCurrent should still score perfectly
        assert f1 > 0.0

    def test_any_of_wrong_value(self):
        gold = json.dumps({"startDate": 2020, "isCurrent": True})
        pred = json.dumps({"startDate": 2019, "isCurrent": True})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ANY_OF)
        assert f1 < 1.0

    def test_any_of_missing_pred(self):
        gold = json.dumps({"startDate": 2020, "isCurrent": True})
        pred = json.dumps({"startDate": None, "isCurrent": True})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ANY_OF)
        assert f1 < 1.0


# ---------------------------------------------------------------------------
# additionalProperties (grouped skills object)
# ---------------------------------------------------------------------------

SCHEMA_ADDITIONAL_PROPS = json.dumps({
    "type": "object",
    "properties": {
        "skills": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "evaluation_config": "string_exact",
                        "items": {"type": "string", "evaluation_config": "string_exact"},
                    }
                },
                {
                    "type": "array",
                    "evaluation_config": "string_exact",
                    "items": {"type": "string", "evaluation_config": "string_exact"},
                },
                {"type": "null"},
            ]
        }
    }
})


class TestAdditionalProperties:
    def setup_method(self):
        self.scorer = Scorer()

    def test_grouped_skills_perfect(self):
        gold = json.dumps({"skills": {"Technical": ["Python", "SQL"], "Soft": ["Communication"]}})
        pred = json.dumps({"skills": {"Technical": ["Python", "SQL"], "Soft": ["Communication"]}})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ADDITIONAL_PROPS)
        assert f1 == 1.0

    def test_grouped_skills_partial(self):
        gold = json.dumps({"skills": {"Technical": ["Python", "SQL", "Java"]}})
        pred = json.dumps({"skills": {"Technical": ["Python", "SQL"]}})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ADDITIONAL_PROPS)
        assert 0.0 < f1 < 1.0

    def test_flat_skills_perfect(self):
        gold = json.dumps({"skills": ["Python", "SQL"]})
        pred = json.dumps({"skills": ["Python", "SQL"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ADDITIONAL_PROPS)
        assert f1 == 1.0

    def test_null_skills(self):
        gold = json.dumps({"skills": None})
        pred = json.dumps({"skills": None})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ADDITIONAL_PROPS)
        # Null gold → nothing to score → no penalty
        assert f1 == 0.0  # no gold fields → F1 undefined → 0


# ---------------------------------------------------------------------------
# Array alignment policy
# ---------------------------------------------------------------------------

SCHEMA_ARRAY_STR = json.dumps({
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "evaluation_config": "string_exact",
        }
    }
})

SCHEMA_ARRAY_OBJ = json.dumps({
    "type": "object",
    "properties": {
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "evaluation_config": "string_exact"},
                    "title":   {"type": "string", "evaluation_config": "string_exact"},
                }
            }
        }
    }
})


class TestArrayAlignment:
    def setup_method(self):
        self.scorer = Scorer()

    def test_primitive_array_exact_match(self):
        gold = json.dumps({"skills": ["Python", "SQL"]})
        pred = json.dumps({"skills": ["Python", "SQL"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert f1 == 1.0

    def test_primitive_array_partial_match(self):
        gold = json.dumps({"skills": ["Python", "SQL", "Java"]})
        pred = json.dumps({"skills": ["Python", "SQL"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert 0.0 < f1 < 1.0

    def test_primitive_array_no_match(self):
        gold = json.dumps({"skills": ["Python"]})
        pred = json.dumps({"skills": ["Java"]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert f1 == 0.0

    def test_object_array_positional_alignment(self):
        gold = json.dumps({"jobs": [
            {"company": "Acme", "title": "Engineer"},
            {"company": "Beta", "title": "Lead"},
        ]})
        pred = json.dumps({"jobs": [
            {"company": "Acme", "title": "Engineer"},
            {"company": "Beta", "title": "Lead"},
        ]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_OBJ)
        assert f1 == 1.0

    def test_object_array_partial_positional(self):
        gold = json.dumps({"jobs": [{"company": "Acme", "title": "Engineer"}]})
        pred = json.dumps({"jobs": [{"company": "Wrong Corp", "title": "Engineer"}]})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_OBJ)
        assert 0.0 < f1 < 1.0

    def test_empty_pred_array(self):
        gold = json.dumps({"skills": ["Python"]})
        pred = json.dumps({"skills": []})
        f1, _ = self.scorer.score_document(pred, gold, SCHEMA_ARRAY_STR)
        assert f1 == 0.0


# ---------------------------------------------------------------------------
# Stochastic metric fallback (no judge provided)
# ---------------------------------------------------------------------------

class TestStochasticFallback:
    def setup_method(self):
        self.scorer = Scorer()  # No judge_callable → falls back to string_exact

    def test_string_semantic_fallback_match(self):
        score = self.scorer._evaluate_with_config("hello", "hello", "string_semantic")
        assert score == 1.0

    def test_string_semantic_fallback_mismatch(self):
        score = self.scorer._evaluate_with_config("hello", "world", "string_semantic")
        assert score == 0.0


# ---------------------------------------------------------------------------
# JSON fence stripping
# ---------------------------------------------------------------------------

class TestCleanJson:
    def setup_method(self):
        self.scorer = Scorer()

    def test_strips_json_fence(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        cleaned = self.scorer._clean_json(raw)
        assert cleaned == '{"key": "value"}'

    def test_strips_plain_fence(self):
        raw = "```\n{\"key\": 1}\n```"
        cleaned = self.scorer._clean_json(raw)
        assert cleaned == '{"key": 1}'

    def test_no_fence_passthrough(self):
        raw = '{"key": "value"}'
        assert self.scorer._clean_json(raw) == raw
