# Deprecated environments (reference branch)

**Branch:** `legacy/deprecated-environments`

These modules are **not** part of the current 2×3×2 thesis design. They are preserved here for historical reference only.

| Module | Status |
|--------|--------|
| `src/environments/delayed_cue.py` | Removed from `main` (replaced by Tower of Hanoi) |
| `src/environments/logical_reasoning.py` | Extension stub; removed from `main` |
| `tests/test_07_delayed_cue_env.py` | Tests for delayed_cue |

Do not import these from production runners on `main`. Use TextWorld Cooking and Tower of Hanoi only.
