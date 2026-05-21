import pytest
from src.evaluation.metrics import string_exact, number_tolerance
from src.evaluation.scorer import Scorer

def test_string_exact():
    assert string_exact("Hello World", "hello world") == 1.0
    assert string_exact("Hello", "World") == 0.0
    assert string_exact(None, "test") == 0.0

def test_number_tolerance():
    assert number_tolerance(100, 100) == 1.0
    assert number_tolerance(104, 100, tolerance=0.05) == 1.0
    assert number_tolerance(106, 100, tolerance=0.05) == 0.0
    assert number_tolerance("invalid", 100) == 0.0

def test_scorer_f1():
    scorer = Scorer()
    assert scorer.calculate_f1(1.0, 1.0) == 1.0
    assert scorer.calculate_f1(0.0, 1.0) == 0.0
    assert abs(scorer.calculate_f1(0.75, 0.5) - 0.6) < 0.01