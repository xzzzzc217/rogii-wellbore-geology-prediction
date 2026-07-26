"""Stream the six OOF columns needed from the 7.4 GB public feature cache.

The source CSV has roughly 200 columns.  Some dataframe CSV readers retain a
large whole-row parser buffer even with ``usecols``.  PyArrow's streaming CSV
reader projects the required columns batch by batch and writes a compact
Parquet artifact for repeatable low-memory evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

COLUMNS = ("well", "id", "last_known_tvt", "pf_ancc", "md_since", "target")
EXPECTED_ROWS = 3_783_989


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-size-mib", type=int, default=64)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        metadata = pq.read_metadata(output)
        if metadata.num_rows != EXPECTED_ROWS:
            raise ValueError(
                f"{output}: existing Parquet has {metadata.num_rows} rows, expected {EXPECTED_ROWS}"
            )
        print(f"reusing {output}: rows={metadata.num_rows}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(
            block_size=args.block_size_mib * (1 << 20),
            use_threads=True,
        ),
        convert_options=pacsv.ConvertOptions(
            include_columns=list(COLUMNS),
            column_types={
                "well": pa.string(),
                "id": pa.string(),
                "last_known_tvt": pa.float64(),
                "pf_ancc": pa.float64(),
                "md_since": pa.float64(),
                "target": pa.float64(),
            },
        ),
    )
    rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch_index, batch in enumerate(reader, start=1):
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    batch.schema,
                    compression="zstd",
                    use_dictionary=("well",),
                )
            writer.write_table(pa.Table.from_batches([batch]))
            rows += batch.num_rows
            if batch_index % 10 == 0:
                print(f"projected batches={batch_index} rows={rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if rows != EXPECTED_ROWS:
        raise ValueError(f"projected {rows} rows, expected {EXPECTED_ROWS}")
    temporary.replace(output)
    print(f"wrote {output}: rows={rows} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
