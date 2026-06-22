from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "llm_one_shot" / "pendulum"

PROMPTS_DIR = RESULTS_DIR / "prompts"
POLICIES_DIR = RESULTS_DIR / "policies"
EVAL_LOGS_DIR = RESULTS_DIR / "eval_logs"
TRACES_DIR = RESULTS_DIR / "sensory_motor_traces"

ENV_ID = "Pendulum-v1"

# First quick test
N_EVAL_EPISODES = 10

# Use 100 later for final evaluation
FINAL_N_EVAL_EPISODES = 100

SEED = 42

OLLAMA_MODEL = "llama3.1:8b"

TEMPERATURE = 0.0
MAX_RESPONSE_TOKENS = 2048