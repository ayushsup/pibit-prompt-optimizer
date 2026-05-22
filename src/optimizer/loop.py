import yaml
from src.core.state_manager import StateManager
from src.agents.critic_mutator import Extractor, Critic, Mutator
from src.data.loader import ExtractBenchLoader

class OptimizerLoop:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.state = StateManager()
        self.extractor = Extractor(self.config['models']['extractor'], self.state)
        self.critic = Critic(self.config['models']['critic'], self.state)
        self.mutator = Mutator(self.config['models']['mutator'], self.state)

        # Setup data loader for real dataset
        dataset_cfg = self.config['dataset']
        self.data = ExtractBenchLoader(
            base_path=dataset_cfg['base_path'] + "/dataset",
            schema_name=dataset_cfg['name']
        )
        # self.scorer = Scorer()

        self.budget_dollars = self.config['budget']['max_cost_dollars']
        self.max_iters = self.config['budget']['max_iterations']

    def check_budget(self) -> bool:
        # Treat 0.0 as an unlimited financial budget (for free tiers/local models)
        if self.budget_dollars <= 0.0:
            return True
            
        current_cost = self.state.get_total_cost()
        if current_cost >= self.budget_dollars:
            print(f"Budget exhausted (${current_cost:.2f} / ${self.budget_dollars:.2f}). Terminating.")
            return False
        return True

    def run(self):
        print(f"Starting Optimization Loop for Ayush Supakar...")
        current_prompt = self.config['seed_prompt']
        best_score = 0.0
        
        # Load ONLY the first 2 validation documents to prevent rate limits during testing
        val_docs = self.data.load_all_document_pairs()[:2]
        
        for iteration in range(self.max_iters):
            if not self.check_budget():
                break
                
            print(f"\n--- Iteration {iteration + 1} ---")
            
            failed_examples = []
            total_score = 0
            
            # 1. Evaluate current prompt
            for doc in val_docs:
                prediction = self.extractor.extract(doc['text'], current_prompt, doc['schema'])
                
                # --- NEW REAL SCORING LOGIC ---
                try:
                    import json
                    # Clean the LLM output in case it wrapped it in markdown
                    clean_pred = prediction.replace("```json", "").replace("```", "").strip()
                    pred_data = json.loads(clean_pred)
                    gold_data = json.loads(doc['gold_json'])
                    
                    # Calculate a simple accuracy score based on matching top-level keys
                    correct_keys = 0
                    total_keys = len(gold_data.keys())
                    
                    for key, gold_val in gold_data.items():
                        if key in pred_data:
                            # If values match exactly, or both are populated lists/strings
                            if pred_data[key] == gold_val:
                                correct_keys += 1
                            elif isinstance(gold_val, list) and isinstance(pred_data[key], list) and len(pred_data[key]) > 0:
                                correct_keys += 0.5 # Partial credit for extracting array items
                            elif isinstance(gold_val, str) and isinstance(pred_data[key], str) and len(pred_data[key]) > 0:
                                correct_keys += 0.5 # Partial credit for extracting a string

                    score = correct_keys / max(total_keys, 1)
                except Exception as e:
                    print(f"JSON Parse Error: Extractor failed to output valid JSON.")
                    score = 0.0  # Total failure if JSON is invalid
                # ------------------------------
                
                total_score += score
                
                if score < 1.0:
                    failed_examples.append({
                        "doc": doc['text'],
                        "pred": prediction,
                        "gold": doc['gold_json']
                    })
            
            avg_score = total_score / max(len(val_docs), 1)
            print(f"Validation Score: {avg_score:.4f}")
            
            # 2. Accept or Reject
            accepted = False
            if avg_score > best_score:
                best_score = avg_score
                accepted = True
                print("New best prompt accepted!")
            else:
                print("Prompt rejected, score did not improve.")
            
            self.state.log_iteration(iteration, current_prompt, avg_score, accepted)

            # 3. Critique & Mutate
            if failed_examples and self.check_budget():
                critiques = []
                # Only critique a sample to save budget
                for fail in failed_examples[:3]:
                    critique = self.critic.critique(fail['doc'], fail['pred'], fail['gold'])
                    critiques.append(critique)
                
                current_prompt = self.mutator.mutate(current_prompt, critiques)
                print("Mutator generated new prompt proposal.")
            elif not failed_examples:
                print("Perfect score achieved on validation set. Stopping early.")
                break

        print("\nOptimization Complete.")
        print(f"Final Best Score: {best_score:.4f}")
        # In a full run, you would now evaluate `current_prompt` against the held-out test split here.