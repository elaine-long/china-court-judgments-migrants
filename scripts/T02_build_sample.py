"""T02 step 0-1: duplicate flags, analysis universe, treatment assignment.

Outputs: data/sample_flags.parquet, data/treatment.parquet, output/tables/T02_sample_counts.csv,
output/tables/T02_date_formats.csv, output/tables/T02_court_city_source.csv
"""
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.treatment import assign_court_city, assign_treatment, date_format_kind, parse_judgment_date  # noqa: E402

NON_CRIMINAL = ("民事/行政案件", "非刑事案件")


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # --- duplicate flags (flag only, never delete)
    flags = con.execute(f"""
        WITH d AS (
            SELECT d.doc_id, m.case_number, m.court_name, m.doc_kind, m.case1_name, d.len_judgment,
                   md5(coalesce(d.judgment, '')) AS h
            FROM read_parquet('{paths.DOCS_TEXT}') d JOIN read_parquet('{paths.META}') m USING (doc_id))
        SELECT doc_id, doc_kind, case1_name,
               doc_id <> min(doc_id) OVER (PARTITION BY h) AS dup_exact,
               row_number() OVER (PARTITION BY case_number, court_name
                                  ORDER BY len_judgment DESC, doc_id) > 1 AS dup_casecourt
        FROM d ORDER BY doc_id""").df()
    flags["keep"] = ~flags["dup_exact"] & ~flags["dup_casecourt"]
    flags["universe"] = (flags["doc_kind"] == "刑初") & flags["keep"] & ~flags["case1_name"].isin(NON_CRIMINAL)
    out = flags[["doc_id", "dup_exact", "dup_casecourt", "keep", "universe"]]
    out.to_parquet(paths.DATA / "sample_flags.parquet", index=False)
    counts = pd.DataFrame([{
        "n_rows": len(flags), "dup_exact": int(flags.dup_exact.sum()),
        "dup_casecourt": int(flags.dup_casecourt.sum()), "keep": int(flags.keep.sum()),
        "keep_and_刑初": int((flags.keep & (flags.doc_kind == "刑初")).sum()),
        "universe": int(flags.universe.sum())}])
    counts.to_csv(paths.TABLES / "T02_sample_counts.csv", index=False)
    print(counts.to_string(index=False), f"  ({time.time() - t0:.0f}s)")

    # --- treatment
    meta = con.execute(f"""SELECT m.doc_id, m.court_name, m.province, m.prefecture, m.court_province,
                                  d.judgment_date
                           FROM read_parquet('{paths.META}') m
                           JOIN read_parquet('{paths.DOCS_TEXT}') d USING (doc_id) ORDER BY doc_id""").df()
    uniq = pd.Series(meta["judgment_date"].dropna().unique())
    parsed = dict(zip(uniq, uniq.map(parse_judgment_date)))
    kinds = dict(zip(uniq, uniq.map(date_format_kind)))
    meta["judgment_date_parsed"] = pd.to_datetime(meta["judgment_date"].map(parsed))
    meta["date_format"] = meta["judgment_date"].map(kinds).fillna("missing")
    # province treated as the court's province (T01 acceptance); case-number abbreviation as fallback
    meta["court_prov_final"] = meta["province"].fillna(meta["court_province"])
    cc = assign_court_city(meta.assign(province=meta["court_prov_final"]))
    meta = pd.concat([meta, cc], axis=1)
    tr = assign_treatment(meta)
    treat = pd.concat([meta[["doc_id", "judgment_date_parsed", "date_format", "court_prov_final",
                             "court_city", "court_city_source"]], tr], axis=1)
    treat = treat.rename(columns={"court_prov_final": "court_province_final"})
    treat.to_parquet(paths.DATA / "treatment.parquet", index=False)

    fmt = (treat.assign(parsed_ok=treat.judgment_date_parsed.notna())
           .groupby("date_format").agg(n=("doc_id", "size"), parsed_ok=("parsed_ok", "mean")).reset_index())
    fmt.to_csv(paths.TABLES / "T02_date_formats.csv", index=False)
    print(fmt.to_string(index=False))
    print(f"date parse failure rate: {treat.judgment_date_parsed.isna().mean():.4%}")
    u = treat[flags["universe"].values]
    src = pd.concat([treat.court_city_source.value_counts(normalize=True).rename("share_all"),
                     u.court_city_source.value_counts(normalize=True).rename("share_universe")], axis=1)
    src.to_csv(paths.TABLES / "T02_court_city_source.csv")
    print(src.round(4).to_string())
    print(f"pilot share in universe: {u.pilot_city.mean():.3f}; TOTAL {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
