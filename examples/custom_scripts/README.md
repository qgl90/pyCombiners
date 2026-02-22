# examples/custom_scripts Folder

This folder contains user-extensible post-processing callbacks for combiner runs.

Each script is loaded by `track-combiner --custom-script ...` and must provide:

```python
def process(results, context):
    ...
```

## Files

- `top_candidates.py`
  - Sorts candidates by quality metrics and writes top-N summary JSON.

- `filter_and_dump.py`
  - Applies tighter analysis cuts to candidate lists and writes selected entries.

## How It Interacts With Other Folders

- Receives `CombinationResult` objects built by `src/trackcomb/combiner.py`.
- Can write derived artifacts for offline analysis in `examples/` or elsewhere.
