"""T01: build the working data layer from the raw CSVs.

Outputs (data/): docs_text.parquet, meta.parquet, old_features.parquet,
theft_candidates.parquet; plus output/tables/T01_alignment.csv.

Usage:
    python scripts/T01_build_working_data.py            # full data
    python scripts/T01_build_working_data.py --example  # Data Example/ files, writes to data/example/
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.case_number import parse_case_numbers  # noqa: E402

CHUNK = 50_000

TEXT_COLS = ["case_number", "court_name", "judgment_date", "formatted_datetime",
             "incident_time", "case_type_raw", "details", "judgment"]
META_SRC_COLS = ["province", "prefecture", "prefecture_code", "case1_name",
                 "case1_category", "year"]
ALIGN_COLS = ["case1_name", "value_yuan", "sentence_months", "year", "province"]
READ_COLS = list(dict.fromkeys(TEXT_COLS + META_SRC_COLS + ALIGN_COLS))
STR_COLS = TEXT_COLS + ["province", "prefecture", "case1_name", "case1_category"]

DOCS_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("case_number", pa.string()), ("court_name", pa.string()),
    ("judgment_date", pa.string()), ("formatted_datetime", pa.string()),
    ("incident_time", pa.string()), ("case_type_raw", pa.string()),
    ("details", pa.string()), ("judgment", pa.string()),
    ("len_judgment", pa.int32()), ("len_details", pa.int32()),
])
META_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("case_number", pa.string()), ("court_name", pa.string()),
    ("judgment_year", pa.int16()), ("incident_year", pa.int16()),
    ("province", pa.string()), ("prefecture", pa.string()),
    ("prefecture_code", pa.int64()), ("case1_name", pa.string()),
    ("case1_category", pa.string()), ("court_prov_abbr", pa.string()),
    ("court_province", pa.string()), ("court_code", pa.string()),
    ("doc_kind", pa.string()),
])

VERDICT_SPLIT = "判决如下"
THEFT_IN_VERDICT = re.compile(r"犯盗窃罪")


def year_from_date(s: pd.Series) -> pd.Series:
    y = s.str.extract(r"((?:19|20)\d{2})")[0]
    return pd.to_numeric(y, errors="coerce").astype("Int16")


def build_from_merged(src, out_dir):
    """Single chunked pass over Merged_Full -> docs_text, meta, theft flags, align cols."""
    docs_w = pq.ParquetWriter(out_dir / "docs_text.parquet", DOCS_SCHEMA, compression="zstd")
    meta_w = pq.ParquetWriter(out_dir / "meta.parquet", META_SCHEMA, compression="zstd")
    align_parts, theft_parts = [], []
    offset = 0
    t0 = time.time()
    reader = pd.read_csv(src, usecols=READ_COLS, dtype={c: "string" for c in STR_COLS},
                         chunksize=CHUNK)
    for chunk in reader:
        n = len(chunk)
        chunk.index = pd.RangeIndex(offset, offset + n)
        doc_id = pd.Series(chunk.index.values, index=chunk.index, dtype="int64")

        docs = chunk[TEXT_COLS].copy()
        docs.insert(0, "doc_id", doc_id)
        docs["len_judgment"] = docs["judgment"].str.len().fillna(0).astype("int32")
        docs["len_details"] = docs["details"].str.len().fillna(0).astype("int32")
        docs_w.write_table(pa.Table.from_pandas(docs, schema=DOCS_SCHEMA, preserve_index=False),
                           row_group_size=CHUNK)

        parsed = parse_case_numbers(chunk["case_number"])
        meta = pd.DataFrame({
            "doc_id": doc_id,
            "case_number": chunk["case_number"],
            "court_name": chunk["court_name"],
            "judgment_year": year_from_date(chunk["judgment_date"]),
            "incident_year": pd.to_numeric(chunk["year"], errors="coerce").astype("Int16"),
            "province": chunk["province"],
            "prefecture": chunk["prefecture"],
            "prefecture_code": pd.to_numeric(chunk["prefecture_code"], errors="coerce").astype("Int64"),
            "case1_name": chunk["case1_name"],
            "case1_category": chunk["case1_category"],
        })
        meta = pd.concat([meta, parsed[["court_prov_abbr", "court_province",
                                        "court_code", "doc_kind"]]], axis=1)
        meta_w.write_table(pa.Table.from_pandas(meta, schema=META_SCHEMA, preserve_index=False),
                           row_group_size=CHUNK)

        # theft candidate flags
        name = chunk["case1_name"]
        is_unclass = name.fillna("").str.startswith("未分类")
        verdict = chunk["judgment"].fillna("")
        after = verdict.str.split(VERDICT_SPLIT, n=1).str[1].fillna("")
        hint = is_unclass & after.str.contains(THEFT_IN_VERDICT)
        theft_parts.append(pd.DataFrame({
            "doc_id": doc_id,
            "is_theft_old": (name == "盗窃罪").fillna(False).astype(bool),
            "is_unclassified": is_unclass.astype(bool),
            "unclassified_theft_hint": hint.astype(bool),
            "has_verdict_marker": verdict.str.contains(VERDICT_SPLIT, regex=False).astype(bool),
        }))

        a = chunk[["case_number"] + ALIGN_COLS].copy()
        a.insert(0, "doc_id", doc_id)
        align_parts.append(a)

        offset += n
        print(f"  merged rows {offset:,}  {time.time() - t0:,.0f}s", flush=True)
    docs_w.close()
    meta_w.close()
    theft = pd.concat(theft_parts, ignore_index=True)
    theft.to_parquet(out_dir / "theft_candidates.parquet", index=False)
    return pd.concat(align_parts, ignore_index=True), theft, offset


def same(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        eq = np.isclose(a.fillna(-9e18), b.fillna(-9e18))
        return pd.Series(eq)
    a, b = a.astype("string"), b.astype("string")
    return (a == b).fillna(False) | (a.isna() & b.isna())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", action="store_true")
    args = ap.parse_args()
    if args.example:
        merged_src, ci_src = paths.RAW_EXAMPLE_MERGED, paths.RAW_EXAMPLE_CASE_INFO
        out_dir, tag = paths.DATA / "example", "example"
    else:
        merged_src, ci_src = paths.RAW_MERGED_FULL, paths.RAW_CASE_INFO
        out_dir, tag = paths.DATA, "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    T = time.time()

    print(f"[1] Merged_Full -> docs_text/meta/theft  ({merged_src.name})")
    align, theft, n_merged = build_from_merged(merged_src, out_dir)
    print(f"    rows={n_merged:,}  time={time.time() - T:,.0f}s")

    cn = align["case_number"]
    print(f"    case_number: missing={cn.isna().sum():,}  "
          f"duplicated rows={cn.duplicated(keep=False).sum():,}  unique={cn.nunique():,}")

    print(f"[2] case_information ({ci_src.name})")
    t1 = time.time()
    ci = pd.read_csv(ci_src, low_memory=False)
    if ci.columns[0].startswith("Unnamed"):
        ci = ci.drop(columns=ci.columns[0])
    print(f"    rows={len(ci):,} cols={ci.shape[1]}  time={time.time() - t1:,.0f}s")

    # alignment check: row counts, full-row comparison and a fixed-seed 1000-row sample
    rows = []
    same_n = len(ci) == n_merged
    k = min(len(ci), n_merged)
    rng = np.random.default_rng(paths.SEED)
    sample = np.sort(rng.choice(k, size=min(1000, k), replace=False))
    for col in ALIGN_COLS:
        full_eq = same(align[col].iloc[:k], ci[col].iloc[:k])
        samp_eq = full_eq.iloc[sample]
        rows.append({"column": col, "sample_n": len(sample), "sample_match": samp_eq.mean(),
                     "full_n": k, "full_match": full_eq.mean(),
                     "full_mismatch_rows": int((~full_eq).sum())})
    al = pd.DataFrame(rows)
    al.insert(0, "rows_merged", n_merged)
    al.insert(1, "rows_case_info", len(ci))
    al.to_csv(paths.TABLES / f"T01_alignment_{tag}.csv", index=False)
    print(al.to_string(index=False))
    # Row-by-row alignment is accepted when every overlapping row matches on all
    # comparison columns. If row counts differ, the extra rows must be at the tail
    # (they are left without old features); this is reported, not silently fixed.
    prefix_aligned = bool((al["full_match"] == 1).all())
    print(f"    same row count: {same_n}; overlapping {k:,} rows fully aligned: {prefix_aligned}")
    if not same_n:
        print(f"    WARNING: {abs(n_merged - len(ci)):,} trailing rows of the longer file "
              f"have no counterpart")

    if prefix_aligned:
        ci.insert(0, "doc_id", np.arange(len(ci), dtype="int64"))
        ci.to_parquet(out_dir / "old_features.parquet", index=False, compression="zstd",
                      row_group_size=CHUNK)
        print(f"[3] old_features written: {len(ci):,} rows x {ci.shape[1]} cols")
    else:
        print("[3] NOT aligned -> old_features.parquet NOT written")

    print(f"theft: is_theft_old={theft.is_theft_old.sum():,}  "
          f"is_unclassified={theft.is_unclassified.sum():,}  "
          f"unclassified_theft_hint={theft.unclassified_theft_hint.sum():,}")
    print(f"TOTAL time {time.time() - T:,.0f}s")


if __name__ == "__main__":
    main()
