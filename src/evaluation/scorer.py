from typing import Dict, Any
from src.evaluation.metrics import string_exact, number_tolerance

class Scorer:
    def evaluate_leaf(self, pred_val: Any, gold_val: Any, eval_config: str) -> float:
        if eval_config == "string_exact":
            return string_exact(pred_val, gold_val)
        elif eval_config == "number_tolerance":
            return number_tolerance(pred_val, gold_val)
        return 0.0

    def calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)