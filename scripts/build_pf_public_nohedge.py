"""Build the exact-public full-stack notebook with only the final hedge removed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "kaggle_notebooks"
    / "pf_public_reproduction"
    / "rogii-pf-public-reproduction.ipynb"
)
OUTPUT_DIR = ROOT / "kaggle_notebooks" / "pf_public_nohedge"
OUTPUT = OUTPUT_DIR / "rogii-pf-public-nohedge.ipynb"

FINAL_CELL = """# PF_PUBLIC_NOHEDGE_FINAL_OVERRIDE
# The complete parent pipeline has already written and audited both files.
# This ablation changes only which frozen stage is exposed as submission.csv.
import hashlib as _nohedge_hashlib
import json as _nohedge_json
import shutil as _nohedge_shutil
from pathlib import Path as _NohedgePath

_nohedge_work = _NohedgePath('/kaggle/working')
_nohedge_source = _nohedge_work / 'submission_before_branch_hedge.csv'
_nohedge_output = _nohedge_work / 'submission.csv'
if not _nohedge_source.exists():
    raise FileNotFoundError(_nohedge_source)

_nohedge_shutil.copyfile(_nohedge_source, _nohedge_output)
_nohedge_frame = pd.read_csv(_nohedge_output)
_nohedge_sample = pd.read_csv(
    _NohedgePath(COMPETITION_DATA_ROOT) / 'sample_submission.csv'
)
if _nohedge_frame.columns.tolist() != ['id', 'tvt']:
    raise ValueError(f'unexpected no-hedge columns: {_nohedge_frame.columns.tolist()}')
if len(_nohedge_frame) != len(_nohedge_sample):
    raise ValueError('no-hedge row count differs from sample submission')
if not _nohedge_frame['id'].astype(str).equals(_nohedge_sample['id'].astype(str)):
    raise ValueError('no-hedge id order differs from sample submission')
_nohedge_values = pd.to_numeric(_nohedge_frame['tvt'], errors='coerce').to_numpy(float)
if not np.isfinite(_nohedge_values).all():
    raise ValueError('no-hedge prediction contains non-finite values')

_nohedge_sha = _nohedge_hashlib.sha256(_nohedge_output.read_bytes()).hexdigest()
_nohedge_report = {
    'profile': 'exact_public_full_stack_no_final_branch_hedge',
    'source_file': _nohedge_source.name,
    'output_file': _nohedge_output.name,
    'rows': int(len(_nohedge_frame)),
    'id_order_matches_sample': True,
    'finite_predictions': True,
    'sha256_submission_csv': _nohedge_sha,
}
(_nohedge_work / 'nohedge_final_audit.json').write_text(
    _nohedge_json.dumps(_nohedge_report, indent=2, sort_keys=True) + '\\n',
    encoding='utf-8',
)
print('PF public no-hedge final override:', _nohedge_report)
"""


def main() -> int:
    notebook = json.loads(PARENT.read_text(encoding="utf-8"))
    cells = copy.deepcopy(notebook["cells"])
    if any(
        "PF_PUBLIC_NOHEDGE_FINAL_OVERRIDE"
        in (
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else str(cell.get("source", ""))
        )
        for cell in cells
    ):
        raise RuntimeError("parent unexpectedly already contains no-hedge override")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "tags": ["nohedge-final-override"],
            },
            "outputs": [],
            "source": FINAL_CELL,
        }
    )
    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["pf_public_nohedge"] = {
        "parent": "heralces/rogii-pf-public-reproduction version 2 source",
        "single_change": (
            "copy submission_before_branch_hedge.csv to final submission.csv"
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
