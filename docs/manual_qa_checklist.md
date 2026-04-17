# Manual image–label QA (spot-check)

Automation (`scripts/12_diagnose_hypotheses.py identity`) confirms labels load and metrics are self-consistent when prediction equals ground truth. **Content** quality still needs human eyes.

1. Run `python scripts/12_diagnose_hypotheses.py data --split test --sample 20 --seed 42`.
2. Open `results/tables/diagnose_sample_for_review.jsonl`.
3. For **5–10** rows, open `absolute_path` and confirm the visible line matches `gt` (tones, subdots, punctuation).

If several rows fail, inspect consolidation exports; if they pass, treat remaining errors as model limitations.
