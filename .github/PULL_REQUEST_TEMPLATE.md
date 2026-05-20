## Phase Scope

- [ ] This PR contains changes from exactly one plan phase/branch.
- [ ] All commits in this PR are atomic and reviewable.

## Definition of Done

- [ ] Phase-specific DoD criteria are satisfied.
- [ ] No unrelated behavioral changes were mixed in.

## Quality Gates

- [ ] `ruff check src tests` is green.
- [ ] `mypy src/signals/verbalized_confidence.py` is green.
- [ ] `python -m pytest tests/ -v` is green.
- [ ] CI quality workflow is green on this PR.
