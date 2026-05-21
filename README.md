# 🚀 Pibit.ai Prompt Optimizer: Automated Critic-Mutator Framework

[cite_start]A production-grade, highly modular multi-agent pipeline designed to automatically optimize Large Language Model (LLM) prompts for highly structured JSON extraction from complex documents[cite: 5]. [cite_start]This system leverages a decoupled scoring engine [cite: 42][cite_start], deterministic dataset evaluation [cite: 31, 32][cite_start], and a resilient Critic-Mutator architecture to systematically debug, iterate, and improve prompt efficiency within strict budget caps[cite: 12, 47].

[cite_start]Designed natively to target the **ExtractBench** benchmark suites [cite: 22][cite_start], the framework operates entirely via configurations—requiring zero code alterations to retarget new underlying datasets, models, or evaluation schemas[cite: 54, 55, 68].

---

## 🛠️ System Architecture

[cite_start]Unlike primitive "random mutation" greedy loops [cite: 4][cite_start], this platform utilizes a **Multi-Role Agentic Strategy** to emulate standard software debugging lifecycles[cite: 17]:

1. [cite_start]**The Extractor:** Processes raw PDF source texts using the current operational prompt variant to output structured JSON matching the exact targeted schema[cite: 22, 29, 30].
2. **The Critic:** Intercepts runtime failures, performing a deep-dive diff between faulty extractions and ground-truth annotations to isolate exact semantic or structured failure modes.
3. [cite_start]**The Mutator:** Acts as an automated prompt engineer, ingesting historical critiques to refactor prompt rules dynamically, avoiding regressions while aggressively climbing the score curve[cite: 70].

---

## ⚙️ Project Structure

```text
pibit-prompt-optimizer/
├── config/
│   ├── base_config.yaml         # Production optimization configuration
│   └── test_config.yaml         # Fast-cycle framework validation config
├── data/
│   └── extract-bench/           # ExtractBench benchmarking data
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── config_parser.py     # Pydantic configuration validation layer
│   │   └── state_manager.py     # SQLite persistence engine & log tracing
│   ├── data/
│   │   ├── loader.py            # PDF text extraction and schema maps
│   │   └── splitter.py          # Seeded deterministic dataset partitioning
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation primitives (exact match, tolerances)
│   │   └── scorer.py            # Precision, Recall, F1 aggregation tree
│   ├── agents/
│   │   ├── extractor.py         # Handles target extraction interactions
│   │   ├── critic.py            # Analyzes failure edge-cases against gold data
│   │   └── mutator.py           # Proposes strategic, non-regressive mutations
│   └── optimizer/
│       ├── loop.py              # Central budget-enforced execution engine
│       └── diff_viewer.py       # Automated Git-style prompt diff logging
├── tests/
│   └── test_scorer.py           # Unit tests for scoring configurations
├── requirements.txt             # Project environment dependencies
└── run.py                       # Single command pipeline entry point

🚀 Step-by-Step Setup1. Environment PreparationClone this repository and establish your local Python environment:Bashgit clone [https://github.com/your-username/pibit-prompt-optimizer.git](https://github.com/your-username/pibit-prompt-optimizer.git)
cd pibit-prompt-optimizer
pip install -r requirements.txt
2. Dataset InitializationSecure the ExtractBench benchmark data locally:  Bashgit clone [https://github.com/ContextualAI/extract-bench.git](https://github.com/ContextualAI/extract-bench.git) data/extract-bench
3. API Key DeploymentExport your Gemini API Key into your current terminal runtime to leverage the high-performance, zero-cost Gemini Free Tier:Linux/macOS:Bashexport GEMINI_API_KEY="your_free_ai_studio_api_key"
Windows (CMD):DOSset GEMINI_API_KEY=your_free_ai_studio_api_key
📈 Running the SystemExecutionRun the full optimization pipeline using the production configuration:  Bashpython run.py
Fault Tolerance & State RecoveryCrash Resilience: Every LLM call, input/output token count, latency value, and trajectory scoring step is instantly persisted to a local SQLite database (run_state.db).  Graceful Resumption: If an external interruption occurs (e.g., system termination, temporary connection issues), the runtime loop safely picks up right from its last valid persisted checkpoint without data or iteration loss.  🔄 Configuration-Driven Dataset RetargetingThe core optimization framework remains fully decoupled from the evaluation data. To switch testing scopes from hiring/resume to finance/10kq or any custom bundle, you do not modify the source code.  Simply edit the config/base_config.yaml dataset entry:  YAMLdataset:
  name: "finance/10kq"             # Retargets schema path seamlessly
  base_path: "./data/extract-bench"
  split_seed: 42                   # Keeps data splits perfectly reproducible
  train_ratio: 0.6
  val_ratio: 0.2
🔍 Observability & Evaluation TestingPrompt Diff Inspections: The system builds automatic markdown diff views inside logs/diffs/ following every successful mutation step, allowing human operators to audit prompt iteration logic directly.  Scoring Integrity Tests: Run the explicit evaluation suite to guarantee mathematical correctness across the per-field matching heuristics:  Bashpytest tests/test_scorer.py
