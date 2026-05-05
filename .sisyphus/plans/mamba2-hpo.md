# Mamba-2 Hyperparameter Optimization Plan

## TL;DR

> **Goal**: Automatically find Mamba-2 hyperparameters that achieve stable reward > GRU baseline (15.32)
> 
> **Approach**: 3-phase HPO using Sample-factory PBT + targeted grid search
> 
> **Deliverables**: 
> - HPO orchestration script with Optuna integration
> - Configurable search space for Mamba-2 parameters
> - Automated result tracking and comparison
> - Best config report with verification
> 
> **Budget**: 1-2 days, ~25 trials, single GPU
> **Success Criteria**: Stable reward > 15.32 with oscillation < ±1.5

---

## Context

### Current State
- **Mamba-2 max reward**: 16.76 (with oscillation ±2.5)
- **GRU baseline**: 15.32 (stable, oscillation ±1.15)
- **Problem**: Mamba-2 has higher peak but less stable; need stable configuration

### Current Parameters (from train_mamba2.py)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `mamba_d_model` | 512 | Research-backed (Drama) |
| `mamba_d_state` | 64 | Conservative |
| `mamba_expand` | 1 | RLBenchNet recommendation |
| `mamba_headdim` | 64 | Standard |
| `mamba_ngroups` | 1 | Critical for stability |
| `rnn_num_layers` | 1 | Simplified |
| `learning_rate` | 1.5e-4 | Lowered for stability |
| `weight_decay` | 0.1 | Prevents norm divergence |
| `batch_size` | 2048 | Reduced for stability |
| `exploration_loss_coeff` | 0.01 | Higher entropy |

### Research Findings
- Mamba-2 is robust to HPO (2.4% variance vs 4.7% for Transformers)
- Sample-factory has built-in PBT for RL-specific params
- Optuna TPE recommended for noisy RL objectives
- 50-100 trials × 50k-500k steps recommended for thorough search

---

## Work Objectives

### Core Objective
Build automated HPO pipeline that finds Mamba-2 configurations achieving stable reward > GRU baseline within 1-2 day budget.

### Concrete Deliverables
- [ ] `hpo/optuna_mamba2.py` - Optuna HPO orchestration script
- [ ] `hpo/search_space.py` - Configurable Mamba-2 parameter search space
- [ ] `hpo/evaluate.py` - Single trial execution and evaluation
- [ ] `hpo/report.py` - Result aggregation and best config report
- [ ] `scripts/manage_gpu.sh` - GPU management script (stops/starts llama-server)
- [ ] Updated `models/train_mamba2.py` with HPO-compatible parameter injection

### Definition of Done
- [ ] HPO completes 25 trials within 48 hours
- [ ] Best config achieves stable reward > 15.32 (GRU baseline)
- [ ] Oscillation amplitude < ±1.5 (matching GRU stability)
- [ ] Results logged to TensorBoard + JSON report

### Must Have
- Optuna TPE sampler for sample-efficient search
- ASHA scheduler for early stopping of bad trials
- Per-trial TensorBoard logging for debugging
- Result persistence across session interruptions

### Must NOT Have (Guardrails)
- No KL adaptive LR scheduling (caused premature convergence to 1.8 reward)
- No more than 25 trials (time budget constraint)
- No architecture changes beyond parameter tuning (scope: HPO only)
- No distributed training (single GPU constraint)
- No running llama-server during HPO (GPU memory conflict)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: No (no existing test suite)
- **Automated tests**: None (HPO-specific, no unit tests needed)
- **Agent-Executed QA**: Every task includes verification scenarios

### QA Policy
Every task MUST include agent-executed QA scenarios with evidence capture.

---

## Execution Strategy

### Parallel Execution Waves

> Single GPU constraint means sequential execution. Each wave = sequential dependency.

```
Wave 1 (Foundation - MUST complete first):
├── Task 1: HPO directory structure + config [quick]
├── Task 2: Search space definition [quick]
└── Task 3: Single trial execution script [quick]

Wave 2 (Core HPO - depends on Wave 1):
├── Task 4: Optuna study setup + sampler [deep]
├── Task 5: Trial execution wrapper [deep]
├── Task 6: Result tracking + early stopping [deep]
└── Task 7: Train script HPO compatibility [quick]

Wave 3 (Orchestration + Reporting):
├── Task 8: Main HPO orchestrator [deep]
├── Task 9: Result aggregation + reporting [quick]
└── Task 10: Best config verification [deep]

Wave FINAL (Verification):
├── Task F1: Plan compliance audit
├── Task F2: Code quality review
├── Task F3: HPO dry-run test
└── Task F4: Scope fidelity check
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| T1-T4 | None | T5-T8 |
| T5 | T1, T2 | T9 |
| T6 | T1, T4 | T9 |
| T7 | T1 | T9 |
| T8 | T4 | T9 |
| T9 | T5-T8 | T10, T11 |
| T10 | T9 | T11 |
| T11 | T9, T10 | None |
| F1-F4 | T11 | None |

### Agent Dispatch Summary

- **Wave 1**: 4 tasks → `quick` (foundation, independent)
- **Wave 2**: 4 tasks → 3× `deep` + 1× `quick`
- **Wave 3**: 3 tasks → 2× `deep` + 1× `quick`
- **Final**: 4 tasks → F1 `oracle`, F2-F4 `unspecified-high`

---

## TODOs

### Wave 1: Foundation

- [x] **1. Create HPO directory structure + config**

  **What to do**:
  - Create `hpo/` directory with `__init__.py`
  - Create `hpo/config.yaml` with default search space parameters
  - Create `hpo/constants.py` with budget constraints (25 trials, 48h)

  **Must NOT do**:
  - No external dependencies beyond Optuna + Sample-factory

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocks**: T4-T7

  **References**:
  - `models/train_mamba2.py` - Current parameter structure
  - Sample-factory docs: https://www.samplefactory.dev/07-advanced-topics/pbt/

  **Acceptance Criteria**:
  - [ ] `hpo/` directory exists with `__init__.py`
  - [ ] `config.yaml` contains default search space
  - [ ] `constants.py` defines TRIAL_LIMIT=25, TIME_BUDGET_HOURS=48

  **QA Scenarios**:
  ```
  Scenario: Directory structure created
    Tool: Bash (ls)
    Steps:
      1. ls -la hpo/
      2. Verify __init__.py, config.yaml, constants.py exist
    Expected: All files present
    Evidence: .sisyphus/evidence/task-1-dir-structure.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): add directory structure and config`

---

- [x] **2. Define Mamba-2 search space**

  **What to do**:
  - Create `hpo/search_space.py` with Optuna-compatible search space
  - Include Mamba-2 specific params: d_model, d_state, headdim, expand
  - Include RL params: learning_rate, exploration_loss_coeff, batch_size
  - Use categorical for discrete params, loguniform for continuous

  **Must NOT do**:
  - No KL adaptive scheduling parameters
  - No distributed training parameters

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocks**: T4

  **References**:
  - Research report: Mamba-2 reference sizes (256, 512, 1024 for d_model)
  - Optuna docs: https://optuna.readthedocs.io/

  **Search Space Definition**:
  ```python
  mamba_params = {
      "d_model": [256, 512, 1024],  # High impact, discrete
      "d_state": [64, 128],  # Medium impact
      "headdim": [64, 128],  # Medium impact
      "expand": [1, 2],  # RLBenchNet recommends 1
      "learning_rate": (1e-5, 1e-3, "loguniform"),  # High impact
      "exploration_loss_coeff": (1e-4, 1e-2, "loguniform"),  # High impact
      "batch_size": [2048, 4096],  # Medium impact
  }
  ```

  **Acceptance Criteria**:
  - [ ] `search_space.py` exports `get_search_space()` function
  - [ ] All Mamba-2 params included with appropriate distributions
  - [ ] No KL adaptive scheduling params

  **QA Scenarios**:
  ```
  Scenario: Search space exports correctly
    Tool: Bash (python -c)
    Steps:
      1. python -c "from hpo.search_space import get_search_space; print(get_search_space())"
    Expected: Dict with all params
    Evidence: .sisyphus/evidence/task-2-search-space.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): define Mamba-2 search space`

---

- [x] **3. Create single trial execution script**

  **What to do**:
  - Create `hpo/evaluate.py` that runs one training trial
  - Accepts params dict, returns reward metrics
  - Logs to TensorBoard with trial-specific prefix
  - Implements timeout for trial duration

  **Must NOT do**:
  - No HPO orchestration (just single trial)
  - No result aggregation

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocks**: T5, T7

  **References**:
  - `models/train_mamba2.py` - Training script structure
  - Sample-factory `run_rl()` API

  **Acceptance Criteria**:
  - [ ] `evaluate.py` has `run_trial(params)` function
  - [ ] Returns dict with `max_reward`, `mean_reward`, `std_reward`
  - [ ] Logs to `train_dir/trial_{trial_id}/`

  **QA Scenarios**:
  ```
  Scenario: Single trial runs without error
    Tool: Bash (timeout + python)
    Steps:
      1. timeout 60 python -c "from hpo.evaluate import run_trial; run_trial({'d_model': 512, ...})"
      2. Verify no exceptions
    Expected: Clean exit or timeout (60s is too short for full training)
    Evidence: .sisyphus/evidence/task-3-trial-run.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): add single trial execution`

---

- [x] **4. Create GPU management script**

  **What to do**:
  - Create `scripts/manage_gpu.sh` bash script
  - Stops llama-server process before HPO starts
  - Restarts llama-server after HPO completes
  - Verifies GPU memory is available before training
  - Handles error cases (process not found, restart failure)

  **Must NOT do**:
  - No killing other processes (only llama-server)
  - No permanent system modifications

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Blocks**: T8 (orchestrator must call this)

  **References**:
  - Local LLM config: `/home/kir/.config/opencode/AGENTS.md`
  - Server URL: `http://localhost:8080`
  - Model: `llama.cpp with CUDA`, RTX 5090 (24GB VRAM)

  **Acceptance Criteria**:
  - [ ] `scripts/manage_gpu.sh` has `stop-llm`, `start-llm`, `check-gpu` commands
  - [ ] Script verifies llama-server is stopped before returning success
  - [ ] Script handles case where llama-server is not running

  **QA Scenarios**:
  ```
  Scenario: GPU management script works
    Tool: Bash (./scripts/manage_gpu.sh)
    Steps:
      1. ./scripts/manage_gpu.sh stop-llm
      2. Verify process stopped: pgrep -f llama-server
      3. ./scripts/manage_gpu.sh start-llm
      4. Verify process started: pgrep -f llama-server
    Expected: Process stops and starts correctly
    Evidence: .sisyphus/evidence/task-4-gpu-mgmt.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): add GPU management script`

---

### Wave 2: Core HPO

- [x] **5. Setup Optuna study with TPE sampler**

  **What to do**:
  - Create `hpo/study.py` with Optuna study configuration
  - Use TPESampler for sample-efficient search
  - Configure direction: maximize `mean_reward` with stability bonus
  - Setup storage for resumption

  **Must NOT do**:
  - No random search (use TPE for efficiency)
  - No more than 25 trials

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T2)
  - **Blocks**: T8

  **References**:
  - Optuna TPE docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
  - Research: TPE recommended for noisy RL objectives

  **Acceptance Criteria**:
  - [ ] `study.py` exports `create_study()` function
  - [ ] Uses TPESampler with n_startup_trials=10
  - [ ] Study direction: maximize

  **QA Scenarios**:
  ```
  Scenario: Study creates with correct sampler
    Tool: Bash (python -c)
    Steps:
      1. python -c "from hpo.study import create_study; s = create_study(); print(s.sampler)"
    Expected: TPESampler instance
    Evidence: .sisyphus/evidence/task-4-study-setup.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): setup Optuna study with TPE`

---

- [x] **6. Create trial execution wrapper**

  **What to do**:
  - Create `hpo/trial_wrapper.py` that integrates Optuna trial with training
  - Handles trial suggestion → param dict → training → metric return
  - Implements trial timeout (2 hours max per trial)
  - Catches exceptions and reports to Optuna

  **Must NOT do**:
  - No HPO orchestration (just trial wrapper)
  - No early stopping logic (separate task)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T3)
  - **Blocks**: T8

  **References**:
  - Optuna trial API: https://optuna.readthedocs.io/en/stable/reference/trial.html

  **Acceptance Criteria**:
  - [ ] `trial_wrapper.py` has `objective(trial)` function
  - [ ] Implements 2-hour timeout per trial
  - [ ] Returns stability-weighted reward metric

  **QA Scenarios**:
  ```
  Scenario: Trial wrapper handles timeout
    Tool: Bash (python + timeout)
    Steps:
      1. Create mock trial with long-running training
      2. Verify timeout triggers and returns partial results
    Expected: Graceful timeout with metric return
    Evidence: .sisyphus/evidence/task-5-timeout.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): add trial execution wrapper`

---

- [x] **7. Implement ASHA early stopping**

  **What to do**:
  - Create `hpo/early_stopping.py` with ASHA scheduler
  - Configure grace_period=10 epochs, max_t=100 epochs
  - Prune trials with reward < baseline after grace period
  - Log pruning decisions to TensorBoard

  **Must NOT do**:
  - No aggressive pruning (grace period essential)
  - No pruning based on single metric

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T1)
  - **Blocks**: T8

  **References**:
  - Research: ASHA recommended for RL HPO
  - Optuna pruning: https://optuna.readthedocs.io/en/stable/reference/pruners.html

  **Acceptance Criteria**:
  - [ ] `early_stopping.py` has `should_prune(trial, metrics)` function
  - [ ] Grace period of 10 epochs enforced
  - [ ] Pruning logged to TensorBoard

  **QA Scenarios**:
  ```
  Scenario: Early stopping respects grace period
    Tool: Bash (python -c)
    Steps:
      1. Test with 5 epochs of bad metrics - should NOT prune
      2. Test with 15 epochs of bad metrics - should prune
    Expected: Correct pruning behavior
    Evidence: .sisyphus/evidence/task-6-early-stopping.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): implement ASHA early stopping`

---

- [x] **8. Update train_mamba2.py for HPO compatibility**

  **What to do**:
  - Add `--hpo_trial_id` argument for trial-specific logging
  - Add `--hpo_params` JSON argument for parameter injection
  - Ensure all Mamba-2 params can be overridden via CLI
  - Add trial completion callback for metric reporting

  **Must NOT do**:
  - No breaking changes to existing training API
  - No KL adaptive scheduling re-introduction

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T3)
  - **Blocks**: T8

  **References**:
  - `models/train_mamba2.py` - Current implementation
  - Sample-factory argument parsing

  **Acceptance Criteria**:
  - [ ] `--hpo_trial_id` argument accepted
  - [ ] `--hpo_params` JSON argument accepted
  - [ ] All Mamba-2 params overridable

  **QA Scenarios**:
  ```
  Scenario: HPO params override defaults
    Tool: Bash (python)
    Steps:
      1. python models/train_mamba2.py --hpo_params '{"d_model": 256}' --help
      2. Verify params are accepted without error
    Expected: Clean argument parsing
    Evidence: .sisyphus/evidence/task-7-hpo-compat.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): add HPO compatibility to train script`

---

### Wave 3: Orchestration + Reporting

- [x] **9. Create main HPO orchestrator**

  **What to do**:
  - Create `hpo/run_hpo.py` - main entry point
  - Integrates study, trial wrapper, early stopping
  - Implements trial limit (25) and time budget (48h)
  - Logs all trials to TensorBoard + JSON
  - **CRITICAL**: Calls `scripts/manage_gpu.sh stop-llm` before HPO starts
  - **CRITICAL**: Calls `scripts/manage_gpu.sh start-llm` after HPO completes
  - Verifies GPU memory available before each trial

  **Must NOT do**:
  - No distributed training
  - No more than 25 trials

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T4-T7)
  - **Blocks**: T9, T10

  **References**:
  - Optuna study optimization loop
  - All Wave 2 components

  **Acceptance Criteria**:
  - [ ] `run_hpo.py` is executable entry point
  - [ ] Respects 25 trial limit
  - [ ] Respects 48-hour time budget
  - [ ] Logs all trials to JSON + TensorBoard

  **QA Scenarios**:
  ```
  Scenario: HPO orchestrator starts correctly
    Tool: Bash (timeout + python)
    Steps:
      1. timeout 30 python -m hpo.run_hpo --dry-run
    Expected: Clean start, shows config, exits on dry-run
    Evidence: .sisyphus/evidence/task-8-orchestrator.txt
  ```

  **Commit**: YES
  - Message: `feat(hpo): add main orchestrator`

---

- [x] **10. Create result aggregation + reporting**

  **What to do**:
  - Create `hpo/report.py` for result analysis
  - Aggregates all trial results from JSON logs
  - Computes stability metrics (oscillation amplitude)
  - Generates best config report with verification commands

  **Must NOT do**:
  - No training execution (reporting only)
  - No parameter modification

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T8)
  - **Blocks**: T10

  **References**:
  - Trial log format from T8

  **Acceptance Criteria**:
  - [ ] `report.py` generates JSON report
  - [ ] Includes stability metrics (oscillation amplitude)
  - [ ] Lists top-3 configs with verification commands

  **QA Scenarios**:
  ```
  Scenario: Report generates from mock data
    Tool: Bash (python)
    Steps:
      1. Create mock trial logs
      2. python -m hpo.report --output test_report.json
    Expected: Valid JSON report with top configs
    Evidence: .sisyphus/evidence/task-9-report.json
  ```

  **Commit**: YES
  - Message: `feat(hpo): add result reporting`

---

- [x] **11. Verify best config with multiple seeds**

  **What to do**:
  - Create `hpo/verify.py` for best config verification
  - Runs top-3 configs with 3 seeds each
  - Computes mean ± std reward across seeds
  - Compares against GRU baseline (15.32)

  **Must NOT do**:
  - No HPO execution (verification only)
  - No more than 3 seeds per config

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T8, T9)
  - **Blocks**: None (final task)

  **References**:
  - Best configs from T9 report
  - GRU baseline: 15.32 reward

  **Acceptance Criteria**:
  - [ ] `verify.py` runs verification with 3 seeds
  - [ ] Computes mean ± std reward
  - [ ] Compares against GRU baseline

  **QA Scenarios**:
  ```
  Scenario: Verification runs with seed variation
    Tool: Bash (python)
    Steps:
      1. python -m hpo.verify --config '{"d_model": 512}' --seeds 1,2,3
    Expected: 3 trials with different seeds, aggregated results
    Evidence: .sisyphus/evidence/task-10-verify.json
  ```

  **Commit**: YES
  - Message: `feat(hpo): add best config verification`

---

## Final Verification Wave

- [x] **F1. Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify all "Must Have" implemented, all "Must NOT Have" absent.

- [x] **F2. Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` + linter + check for AI slop patterns.

- [x] **F3. HPO Dry-Run Test** — `unspecified-high`
  Run `python -m hpo.run_hpo --dry-run --max-trials 1` to verify full pipeline.

- [x] **F4. Scope Fidelity Check** — `deep`
  Verify no scope creep: no distributed training, no KL adaptive scheduling, no >25 trials.

---

## Commit Strategy

- **Wave 1**: `feat(hpo): foundation` - T1, T2, T3
- **Wave 2**: `feat(hpo): core` - T4, T5, T6, T7
- **Wave 3**: `feat(hpo): orchestration` - T8, T9, T10
- **Final**: `fix(hpo): verification` - F1-F4

---

## Success Criteria

### Verification Commands
```bash
# Run HPO with trial limit
python -m hpo.run_hpo --max-trials 25 --time-budget 48

# View results
tensorboard --logdir=./train_dir/hpo_

# Generate report
python -m hpo.report --output best_config.json

# Verify best config
python -m hpo.verify --config $(cat best_config.json) --seeds 1,2,3
```

### Final Checklist
- [ ] All "Must Have" present (Optuna TPE, ASHA, 25 trial limit, TensorBoard logging)
- [ ] All "Must NOT Have" absent (no KL adaptive, no distributed, no >25 trials)
- [ ] Best config achieves stable reward > 15.32
- [ ] Oscillation amplitude < ±1.5
