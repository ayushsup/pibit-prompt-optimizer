import yaml
import json
from src.core.state_manager import StateManager
from src.agents.critic_mutator import Extractor, Critic, Mutator
from src.data.loader import ExtractBenchLoader
from src.data.splitter import deterministic_split
from src.optimizer.diff_viewer import DiffViewer

class OptimizerLoop:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.state = StateManager()
        self.extractor = Extractor(self.config['models']['extractor'], self.state)
        self.critic = Critic(self.config['models']['critic'], self.state)
        self.mutator = Mutator(self.config['models']['mutator'], self.state)
        self.diff_viewer = DiffViewer()

        dataset_cfg = self.config['dataset']
        self.data_loader = ExtractBenchLoader(
            base_path=dataset_cfg['base_path'] + "/dataset",
            schema_name=dataset_cfg['name']
        )
        self.split_seed = dataset_cfg.get('split_seed', 42)

        self.budget_dollars = self.config['budget']['max_cost_dollars']
        self.max_iters = self.config['budget']['max_iterations']

    def check_budget(self) -> bool:
        if self.budget_dollars <= 0.0:
            return True
        current_cost = self.state.get_total_cost()
        if current_cost >= self.budget_dollars:
            print(f"Budget exhausted (${current_cost:.2f} / ${self.budget_dollars:.2f}). Terminating.")
            return False
        return True

    def run(self):
        print(f"Starting Production Optimization Loop for Ayush Supakar...")
        current_prompt = self.config['seed_prompt']
        best_prompt = current_prompt
        best_score = 0.0
        rejected_history = []
        
        # 1. Load and deterministically split the data
        all_docs = self.data_loader.load_all_document_pairs()
        if not all_docs:
            print("No documents found. Check dataset path.")
            return

        train_docs, val_docs, test_docs = deterministic_split(
            all_docs, 
            seed=self.split_seed, 
            train_ratio=0.5, 
            val_ratio=0.2
        )
        print(f"Dataset split: {len(train_docs)} Train | {len(val_docs)} Val | {len(test_docs)} Test")
        
        # 2. Main Optimization Loop
        for iteration in range(self.max_iters):
            if not self.check_budget():
                break
                
            print(f"\n--- Iteration {iteration + 1} ---")
            
            failed_examples = []
            total_score = 0
            
            # Evaluate against the FULL validation set
            # Evaluate against the FULL validation set
            for doc in val_docs:
                prediction = self.extractor.extract(doc['text'], current_prompt, doc['schema'])
                
                # --- NEW THROTTLE ---
                import time
                time.sleep(8)  # Give the API a moment to breathe between Extractor reads
                # --------------------

                try:
                    clean_pred = prediction.replace("```json", "").replace("```", "").strip()
                    pred_data = json.loads(clean_pred)
                    gold_data = json.loads(doc['gold_json'])
                    
                    correct_keys = 0
                    for key, gold_val in gold_data.items():
                        if key in pred_data and pred_data[key] == gold_val:
                            correct_keys += 1
                        elif key in pred_data and isinstance(gold_val, (list, dict)) and len(pred_data[key]) > 0:
                            correct_keys += 0.5 
                            
                    score = correct_keys / max(len(gold_data.keys()), 1)
                except Exception:
                    score = 0.0  
                
                total_score += score
                
                if score < 1.0:
                    failed_examples.append({
                        "doc": doc['text'],
                        "pred": prediction,
                        "gold": doc['gold_json']
                    })
            
            avg_score = total_score / max(len(val_docs), 1)
            print(f"Validation Score: {avg_score:.4f}")
            
            accepted = False
            if avg_score > best_score:
                best_score = avg_score
                accepted = True
                print("🏆 New best prompt accepted!")
                self.diff_viewer.generate_diff(best_prompt, current_prompt, iteration)
                best_prompt = current_prompt
            else:
                print("❌ Prompt rejected, score regressed.")
                if current_prompt not in rejected_history:
                    rejected_history.append(current_prompt)
                current_prompt = best_prompt 
            
            self.state.log_iteration(iteration, current_prompt, avg_score, accepted)

            if failed_examples and self.check_budget():
                critiques = []
                for fail in failed_examples[:3]:
                    critique = self.critic.critique(fail['doc'], fail['pred'], fail['gold'])
                    critiques.append(critique)
                    
                    # --- NEW THROTTLE ---
                    time.sleep(8) # Prevent rapid-fire Critic calls from triggering 429s
                    # --------------------
                
                current_prompt = self.mutator.mutate(best_prompt, critiques, rejected_history)
                print("Mutator drafted a new prompt proposal.")
                time.sleep(8) # Pause before the next iteration begins
            elif not failed_examples:
                print("Perfect validation score achieved! Halting optimization.")
                break

        # 3. Final Test Set Evaluation (Assignment Deliverable)
        print("\n=================================")
        print("🚀 RUNNING FINAL TEST EVALUATION")
        print("=================================")
        test_score_total = 0
        
        # Evaluate against the FULL held-out test set
        for doc in test_docs:
            prediction = self.extractor.extract(doc['text'], best_prompt, doc['schema'])
            try:
                clean_pred = prediction.replace("```json", "").replace("```", "").strip()
                pred_data = json.loads(clean_pred)
                gold_data = json.loads(doc['gold_json'])
                
                correct_keys = 0
                for key, gold_val in gold_data.items():
                    if key in pred_data and pred_data[key] == gold_val:
                        correct_keys += 1
                    elif key in pred_data and isinstance(gold_val, (list, dict)) and len(pred_data[key]) > 0:
                        correct_keys += 0.5 
                        
                score = correct_keys / max(len(gold_data.keys()), 1)
                test_score_total += score
            except Exception:
                pass
                
        final_test_score = test_score_total / max(len(test_docs), 1)
        print(f"Final Held-Out Test Score: {final_test_score:.4f}")
        print("Optimization Pipeline Complete. Diffs logged to /logs/diffs/")