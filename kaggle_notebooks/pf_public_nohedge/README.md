# ROGII PF Public No-Hedge

This private one-layer ablation reuses the complete attributed public
reproduction and changes only its final output selection:

```text
submission_before_branch_hedge.csv -> submission.csv
```

The removed parent layer adds exactly +2 ft to all 4,301 hidden rows of
`00e12e8b` and changes no other well. The parent model package guard selected
zero correction, so the pre-hedge and pre-model-package stage files are
identical.

The appended final cell copies the already generated stage file byte-for-byte,
then rechecks columns, row count, ID order, finite values, and SHA-256. It uses
no truth and does not alter any upstream PF, projection, learned, contact, or
visible-prefix computation.
