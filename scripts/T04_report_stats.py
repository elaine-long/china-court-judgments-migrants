"""T04 step 4: sample lengths and rule-based expected positives (for the report only; never in annotation files)."""
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.io import load_texts  # noqa: E402

ANN = paths.ROOT / "annotation"
T = paths.TABLES


def main():
    main_s = pd.read_csv(ANN / "sample_main.csv")
    pilot = pd.read_csv(ANN / "sample_pilot.csv")
    remap = pd.read_csv(ANN / "_private" / "remap.csv")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    ids = pd.concat([main_s.doc_id, pilot.doc_id]).unique()
    con.register("ids", pd.DataFrame({"doc_id": ids}))
    f = con.execute(f"""SELECT f.*, d.len_judgment FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
                        JOIN read_parquet('{paths.DOCS_TEXT}') d USING (doc_id)
                        WHERE doc_id IN (SELECT doc_id FROM ids)""").df().set_index("doc_id")
    txt = load_texts(ids, columns=["judgment"]).set_index("doc_id").judgment

    lens = []
    for name, s in [("main", main_s.doc_id), ("pilot", pilot.doc_id), ("recheck", remap.doc_id)]:
        x = f.loc[s, "len_judgment"]
        lens.append({"sample": name, "n": len(x), "mean": x.mean(), "p10": x.quantile(.1), "median": x.median(),
                     "p90": x.quantile(.9), "max": x.max()})
    lens = pd.DataFrame(lens)
    lens.to_csv(T / "T04_lengths.csv", index=False)
    print(lens.round(0).to_string(index=False))

    m = f.loc[main_s.doc_id].copy()
    m["weight"] = main_s.set_index("doc_id").loc[m.index, "weight_combined"]
    t = txt.loc[m.index]
    st = m.d1_pretrial_status_at_judgment
    fields = {
        "plea_formal": m.plea_formal, "jiejie": m.jiejie, "procedure=速裁": m.speedy,
        "nonlocal_origin (migrant_city_nores=1)": m.d1_migrant_city_nores == True,  # noqa: E712
        "nonlocal_origin known": m.d1_migrant_city_nores.notna(),
        "residence_local=0 (住址在外地)": m.d1_local_residence == False,  # noqa: E712
        "no_fixed_residence": t.str.contains("无固定住所|居无定所|流窜"),
        "ever_detained": m.d1_detained, "ever_arrested": m.d1_arrested, "ever_bail": m.d1_bail,
        "status=取保": st == "取保", "status=监视居住": st == "rsl",
        "probation": m.d1_probation.fillna(False), "counsel_d1 (any)": m.counsel,
        "counsel=指定": m.counsel_type == "指定", "counsel=值班律师": m.duty_lawyer,
        "sentencing_rec (建议判处)": t.str.contains("建议判处|建议对被告人.{0,10}判处|量刑建议"),
        "denial_reason_text": m.probation_denial_text.notna(),
        "denial: residence": m.probation_denial_text.fillna("").str.contains("社区矫正|无固定住所|居所"),
        "detention_days_known": t.str.contains(r"即自\d{4}年"),
        "spec_burglary": m.spec_burglary, "spec_pickpocket": m.spec_pickpocket,
        "spec_multiple": m.spec_multiple, "spec_weapon": m.spec_weapon,
        "recidivist": m.d1_recidivist, "surrender": m.surrender, "restitution": m.restitution,
        "forgiveness": m.forgiveness, "penalty=管制": m.d1_penalty_type == "管制",
        "penalty=单处罚金": m.d1_penalty_type == "单处罚金",
    }
    rows = []
    for k, v in fields.items():
        v = v.fillna(False).astype(bool)
        rows.append({"field": k, "positives_in_300": int(v.sum()), "share_unweighted": v.mean(),
                     "share_weighted": (v * m.weight).sum() / m.weight.sum(),
                     "lt_20": int(v.sum()) < 20})
    ep = pd.DataFrame(rows)
    ep.to_csv(T / "T04_expected_positives.csv", index=False)
    print(ep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
