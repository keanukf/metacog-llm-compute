---
name: Refactor pilot run logging
overview: Refactor `run_pilot.py` to use sequential test-then-save logging, reorder tests (TextWorld before ToH), remove redundant outputs (benchmark JSON, cost validation MD), and convert the feasibility report to JSON.
todos:
  - id: rewrite-main
    content: "Rewrite main() in run_pilot.py: new test order, sequential save-after-each-test, remove deferred batch writes"
    status: completed
  - id: remove-stub-test4
    content: Remove run_test4_textworld() stub; rename test5 -> test4, ToH -> test5
    status: in_progress
  - id: inline-mock-benchmark
    content: Move mock benchmark logic out of tests/test_01 into run_pilot.py to break import dependency
    status: pending
  - id: feasibility-json
    content: Replace _write_pilot_feasibility_report (MD) and _write_pilot_cost_validation (MD) with _build_feasibility_report() returning a dict saved as pilot_feasibility.json
    status: pending
  - id: remove-dead-outputs
    content: Remove code writing pilot_benchmark.json, pilot_calibration.json, pilot_cost_validation.md, pilot_feasibility_report.md
    status: pending
  - id: clean-config
    content: "Clean up pilot.yaml: remove unused paths, test4_textworld section, unused test3 keys, fix comments"
    status: pending
  - id: verify-tests
    content: Run pytest to verify no existing tests broke from the refactoring
    status: pending
isProject: false
---

# Refactor Pilot Run and Logging

## Problem

The current pilot runner in [`scripts/run_pilot.py`](scripts/run_pilot.py) has several issues:
- Tests 1-3 (inference speed, TLE, VC) run first but their results are only saved at the very end in a big `pilot_benchmark.json`, after TextWorld and ToH episodes have already been logged individually.
- Test 4 is a pointless TextWorld stub, then ToH runs, and only then the real TextWorld e2e test (Test 5). Confusing order.
- `pilot_benchmark.json` is a dumping ground for all test results -- incomplete and redundant since episodes already have their own files.
- `pilot_cost_validation.md` is a markdown report that adds little value.
- `pilot_calibration.json` just re-bundles the same episode data already saved as `ep_*.json` files.
- `pilot_feasibility_report.md` is markdown; should be machine-readable JSON.

## New Design

**Sequential test-then-save pattern:** Each test runs, logs to terminal, and immediately saves its result to its own JSON file. No deferred batch writing.

**New test order (6 tests down to 5 + feasibility):**

1. **Sanity check** (real mode only) -- quick single-inference check, result saved to `pilot_sanity.json`
2. **Test 1: Inference speed** -- N prompts, tok/s, latency. Result saved to `pilot_test1_inference.json`
3. **Test 2: Token-level entropy** -- TLE extraction. Result saved to `pilot_test2_tle.json`
4. **Test 3: Verbalized confidence** -- VC parsing. Result saved to `pilot_test3_vc.json`
5. **Test 4: TextWorld e2e** -- replaces old test 4 stub + old test 5. Full episodes with C0/C1/C2. Each episode saved to `ep_textworld_{inst}_{stage}_{run}.json` (unchanged)
6. **Test 5: Tower of Hanoi** -- parseability episodes at C0. Each episode saved to `ep_tower_of_hanoi_{i}_C0_0.json` (unchanged)
7. **Feasibility check** -- aggregates results from tests 1-5, computes ECE from TextWorld episodes (absorbs old test 6), produces Go/No-Go checklist. Saved to `pilot_feasibility.json`

**Removed outputs:**
- `pilot_benchmark.json` -- gone (each test saves its own file)
- `pilot_cost_validation.md` -- gone
- `pilot_feasibility_report.md` -- replaced by `pilot_feasibility.json`
- `pilot_calibration.json` -- gone (episodes already saved individually)
- `_write_pilot_cost_validation()` function -- deleted
- `_write_pilot_feasibility_report()` function -- replaced by `_build_feasibility_report()` that returns a dict

## Key Changes

### [`scripts/run_pilot.py`](scripts/run_pilot.py)

- **Remove** `run_test4_textworld()` (stub test). The e2e test (currently `run_test5_e2e`) becomes the new Test 4.
- **Rename** `run_test5_e2e` -> `run_test4_textworld_e2e` and `run_test_tower_of_hanoi_parseability` -> `run_test5_tower_of_hanoi`.
- **Rename** `run_test6_logging_analysis` into `_build_feasibility_report()` that takes all test results and returns a dict (includes ECE computation and Go/No-Go checks).
- **Add** a helper `_save_test_result(name, data, output_dir)` that writes `{output_dir}/{name}.json` and calls `log(f"Wrote {name}.json")` -- used by every test.
- **Rewrite `main()`** to follow the sequential pattern:

```python
# Sanity check (real only)
if real_model:
    sanity = _sanity_check_real_inference(...)
    _save_test_result("pilot_sanity", sanity, output_dir)

# Test 1
test1 = run_test1_inference_speed(...)
_save_test_result("pilot_test1_inference", test1, output_dir)

# Test 2
test2 = run_test2_token_entropy(...)
_save_test_result("pilot_test2_tle", test2, output_dir)

# Test 3
test3 = run_test3_verbalized_confidence(...)
_save_test_result("pilot_test3_vc", test3, output_dir)

# Test 4: TextWorld e2e (episodes saved inside)
tw_episodes = run_test4_textworld_e2e(...)
log(f"Test 4 done -- {len(tw_episodes)} episodes")

# Test 5: Tower of Hanoi (episodes saved inside)
toh_result = run_test5_tower_of_hanoi(...)
log(f"Test 5 done -- parse_rate={toh_result['parse_rate']:.2f}")

# Feasibility
feasibility = _build_feasibility_report(test1, test2, test3, toh_result, tw_episodes, ...)
_save_test_result("pilot_feasibility", feasibility, output_dir)
```

- **Delete** `_write_pilot_cost_validation()` and `_write_pilot_feasibility_report()` functions entirely.
- **Delete** all end-of-`main()` code that writes `pilot_benchmark.json`, `pilot_calibration.json`, and the two `.md` files.
- **Remove** the import of `_run_mock_benchmark` from `tests.test_01_inference_speed`. Move the mock benchmark logic inline or into a small helper in `run_pilot.py` to avoid the pilot script importing from the test suite.
- **Clean up** unused config keys: remove references to `paths.pilot_benchmark`, `paths.pilot_calibration`, `paths.pilot_cost_validation` from code. Remove `test4_textworld` config section (no longer used). Clean up the `test3_verbalized_confidence` config section (never wired in).

### [`configs/pilot.yaml`](configs/pilot.yaml)

- Remove `paths.pilot_benchmark`, `paths.pilot_calibration`, `paths.pilot_cost_validation` entries. Keep `paths.pilot_feasibility_report` but rename to `paths.pilot_feasibility` (now JSON).
- Remove `test4_textworld` section (no longer a separate test).
- Remove `test3_verbalized_confidence.num_easy`/`num_hard` (never used by code).
- Fix the misleading comment on line 25 (`5x3x3 = 45` is wrong).

### Feasibility JSON structure

The `pilot_feasibility.json` will contain the same Go/No-Go data but in machine-readable form:

```json
{
  "pilot_mode": "lmstudio",
  "summary": {
    "tokens_per_sec": 29.6,
    "vc_parse_rate": 0.67,
    "toh_parse_rate": 0.95,
    "ece": 0.34,
    "n_episodes_textworld": 3,
    "n_episodes_toh": 1
  },
  "checks": [
    {"id": 1, "question": "...", "passed": true, "fallback": "..."},
    ...
  ],
  "passed": 8,
  "total": 11,
  "go": true,
  "wall_clock_total_s": 123.4
}
```

### What stays the same

- Per-episode JSON files (`ep_textworld_*.json`, `ep_tower_of_hanoi_*.json`) -- format and naming unchanged.
- `src/utils/logging_utils.py` -- `log_episode()` and other utilities unchanged.
- `src/utils/run_progress.py` -- `log()`, `log_step_line()` unchanged.
- All pytest test files under `tests/` -- unchanged (they test individual components independently).
- The `--config`, `--output-dir`, `--pilot-mode`, `--real` CLI interface -- unchanged.
