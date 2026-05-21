import yaml
from pydantic import BaseModel, Field

class DatasetConfig(BaseModel):
    name: str
    base_path: str
    split_seed: int = 42
    train_ratio: float = 0.6
    val_ratio: float = 0.2

class BudgetConfig(BaseModel):
    max_iterations: int = 20
    max_cost_dollars: float = 5.0

class ModelsConfig(BaseModel):
    extractor: str
    critic: str
    mutator: str

class OptimizerConfig(BaseModel):
    dataset: DatasetConfig
    budget: BudgetConfig
    models: ModelsConfig
    seed_prompt: str

def load_config(yaml_path: str) -> OptimizerConfig:
    with open(yaml_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return OptimizerConfig(**raw_config)