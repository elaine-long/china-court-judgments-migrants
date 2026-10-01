"""Rule extraction on theft and robbery documents in the universe (multiprocess).

T02 built this step; T03 added the gazetteer-based origin fields, the court prefecture and the
rule fixes (T02 outputs archived in data/archive_T02/).
Outputs: data/feat_rules.parquet (doc level, with d1_ fields), data/defendants_rules.parquet
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.extract_rules import extract_doc, set_prefecture_map  # noqa: E402
from src.gazetteer import court_prefecture  # noqa: E402

BATCH = 2_000
N_PROC = max(1, mp.cpu_count() - 1)


def _init(pref_map):
    set_prefecture_map(pref_map)


def _work(batch):
    docs, defs = [], []
    for doc_id, text, jd, prov, cpref in batch:
        d, dd = extract_doc(doc_id, text, jd, prov, cpref)
        docs.append(d)
        defs.extend(dd)
    return docs, defs


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    cand = con.execute(f"""
        SELECT s.doc_id, t.judgment_date_parsed, t.court_province_final, t.court_city,
               m.prefecture_code::VARCHAR AS prefecture_code, m.court_name, m.case1_name, c.is_theft_old
        FROM read_parquet('{paths.DATA / "sample_flags.parquet"}') s
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        JOIN read_parquet('{paths.META}') m USING (doc_id)
        JOIN read_parquet('{paths.THEFT_CANDIDATES}') c USING (doc_id)
        JOIN read_parquet('{paths.DOCS_TEXT}') d USING (doc_id)
        WHERE s.universe AND (c.is_theft_old OR m.case1_name = '抢劫罪'
              OR regexp_matches(d.judgment, '犯(盗窃|抢劫)罪'))
        ORDER BY s.doc_id""").df()
    print(f"candidates: {len(cand):,}  ({time.time() - t0:.0f}s)", flush=True)
    info = cand.set_index("doc_id")
    jd = {k: (v.date() if pd.notna(v) else None) for k, v in info["judgment_date_parsed"].items()}
    prov = info["court_province_final"].to_dict()
    ci = cand[["court_city", "prefecture_code", "court_province_final", "court_name"]].astype(object)
    ci = ci.where(ci.notna(), None)
    cand["court_prefecture_g"] = [court_prefecture(*r) for r in ci.itertuples(index=False)]
    cpref = dict(zip(cand["doc_id"], cand["court_prefecture_g"]))
    pm = con.execute(f"""SELECT prefecture, mode(province) AS p FROM read_parquet('{paths.META}')
                         WHERE prefecture IS NOT NULL AND province IS NOT NULL GROUP BY 1""").df()
    pref_map = dict(zip(pm["prefecture"], pm["p"]))

    wanted = set(cand["doc_id"])

    def batches():
        pf = pq.ParquetFile(paths.DOCS_TEXT)
        buf = []
        for rb in pf.iter_batches(batch_size=20_000, columns=["doc_id", "judgment"]):
            ids = rb.column("doc_id").to_pylist()
            texts = rb.column("judgment").to_pylist()
            for i, t in zip(ids, texts):
                if i in wanted:
                    buf.append((i, t or "", jd.get(i), prov.get(i), cpref.get(i)))
                    if len(buf) >= BATCH:
                        yield buf
                        buf = []
        if buf:
            yield buf

    docs, defs = [], []
    with mp.Pool(N_PROC, initializer=_init, initargs=(pref_map,)) as pool:
        for k, (d, dd) in enumerate(pool.imap_unordered(_work, batches(), chunksize=1)):
            docs.extend(d)
            defs.extend(dd)
            if k % 50 == 0:
                print(f"  docs {len(docs):,}  {time.time() - t0:.0f}s", flush=True)
    print(f"extracted {len(docs):,} docs, {len(defs):,} defendants  ({time.time() - t0:.0f}s)", flush=True)

    D = pd.DataFrame(docs).sort_values("doc_id").reset_index(drop=True)
    F = pd.DataFrame(defs).sort_values(["doc_id", "def_idx"]).reset_index(drop=True)
    D = D.merge(cand[["doc_id", "case1_name", "is_theft_old", "court_prefecture_g"]], on="doc_id", how="left")
    D["in_theft"] = D["is_theft"] | D["is_theft_old"]
    D["in_robbery"] = D["is_robbery"] | (D["case1_name"] == "抢劫罪")
    D = D[D["in_theft"] | D["in_robbery"]].drop(columns=["case1_name"])
    F = F[F["doc_id"].isin(D["doc_id"])]
    for c in ["birth_date"] + [c for c in F.columns if c.endswith("_date")]:
        F[c] = pd.to_datetime(F[c])
    for c in ["migrant_v0", "migrant_city", "migrant_prov", "migrant_city_nores", "local_residence", "probation"]:
        F[c] = F[c].astype("boolean")
    d1 = F[F["def_idx"] == 0].drop(columns=["def_idx", "def_header_raw"]).set_index("doc_id").add_prefix("d1_")
    D = D.merge(d1, left_on="doc_id", right_index=True, how="left")
    D.to_parquet(paths.DATA / "feat_rules.parquet", index=False)
    F.to_parquet(paths.DATA / "defendants_rules.parquet", index=False)
    print(f"feat_rules: {len(D):,} docs (theft {int(D.in_theft.sum()):,}, robbery {int(D.in_robbery.sum()):,}, "
          f"both {int((D.in_theft & D.in_robbery).sum()):,}); defendants: {len(F):,}")
    print(f"TOTAL {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
