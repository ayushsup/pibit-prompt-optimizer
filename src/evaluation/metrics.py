def string_exact(pred: str, gold: str) -> float:
    if not isinstance(pred, str) or not isinstance(gold, str):
        return 0.0
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def number_tolerance(pred: float, gold: float, tolerance: float = 0.05) -> float:
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        if gold_val == 0:
            return 1.0 if pred_val == 0 else 0.0
        error = abs(pred_val - gold_val) / abs(gold_val)
        return 1.0 if error <= tolerance else 0.0
    except (ValueError, TypeError):
        return 0.0

# NOTE: array_llm or string_semantic would require LLM calls and MUST be cached in SQLite.