# ROGII Frontier A28 v1

This private Kaggle notebook is an attribution-preserving verification copy of
`prvsiyan/rogii-public-frontier-blend-research-visuals`.

It retains the public source through the A28 final audit only. The final
`submission.csv` is reconstructed at runtime as:

`0.62 * SP45 projection + 0.38 * learned PF-1.3 trajectory`

The public run report records a source score of 6.433. That value is provenance,
not a score claim for this private rerun. Later Q0522, pair-blend, and visual
cells are deliberately excluded.

Build and statically validate before pushing:

```powershell
.\.venv\Scripts\python.exe scripts\build_frontier_a28_v1.py
.\.venv\Scripts\python.exe -m ruff check scripts\build_frontier_a28_v1.py
```
