import random
from typing import List, Tuple, Dict, Any

def deterministic_split(data: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> Tuple[List, List, List]:
    """Splits data deterministically based on a seed to ensure reproducibility."""
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train = shuffled_data[:train_end]
    val = shuffled_data[train_end:val_end]
    test = shuffled_data[val_end:]
    
    return train, val, test