# Artifact Schema

## Episode JSON (`episode.v1`)

Episode files written by `log_episode(...)` now include `schema_version: "episode.v1"`.
This marks the stable baseline for pilot/phase episode records.

Required high-level keys:

- `episode_id`
- `compute_stage`
- `task_success`
- `steps_detail` (may be synthesized by loaders for legacy files)
- `tle_per_step`
- `vc_per_step`

## Loader Compatibility

`src/analysis/datasets.py` performs a minimal structural validation before flattening
episode artifacts. Records missing required core fields are ignored.

## Migration Notes

- Legacy episodes without `schema_version` are still readable.
- New writes automatically set `schema_version` to `episode.v1`.
- Golden fixture: `tests/fixtures/episode_schema_v1.json` captures the expected key contract.
