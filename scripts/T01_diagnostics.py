"""T01 diagnostics (a)-(e) on the working data layer. Tables -> output/tables/T01_*.csv."""
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.io import load_texts  # noqa: E402

T = paths.TABLES
con = duckdb.connect()
con.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet('{paths.META}')")
con.execute(f"CREATE VIEW docs AS SELECT * FROM read_parquet('{paths.DOCS_TEXT}')")
con.execute(f"CREATE VIEW old AS SELECT * FROM read_parquet('{paths.OLD_FEATURES}')")
con.execute(f"CREATE VIEW theft AS SELECT * FROM read_parquet('{paths.THEFT_CANDIDATES}')")


def q(sql):
    return con.execute(sql).df()


def save(df, name):
    df.to_csv(T / f"T01_{name}.csv", index=False)
    print(f"\n== {name}\n{df.to_string(index=False, max_rows=60)}")


t0 = time.time()

# ---------- basic counts
save(q("""SELECT case1_name, count(*) n FROM meta
          WHERE case1_name LIKE '未%' OR case1_name IN ('非刑事案件','民事/行政案件')
             OR case1_name IS NULL GROUP BY 1 ORDER BY n DESC"""), "unclassified_names")
save(q("""SELECT sum(is_theft_old::INT) is_theft_old, sum(is_unclassified::INT) is_unclassified,
                 sum(unclassified_theft_hint::INT) unclassified_theft_hint,
                 sum((is_unclassified AND NOT has_verdict_marker)::INT) unclassified_no_verdict_marker,
                 count(*) n FROM theft"""), "theft_counts")

# ---------- (a) province consistency
save(q("""SELECT count(*) n, avg((court_province IS NOT NULL)::INT) share_court_prov_parsed,
                 avg((province IS NOT NULL)::INT) share_province_nonnull,
                 avg(CASE WHEN court_province IS NOT NULL AND province IS NOT NULL
                          THEN (court_province <> province)::INT END) mismatch_rate
          FROM meta"""), "a_province_overall")
save(q("""SELECT court_province, count(*) n_both,
                 sum((court_province <> province)::INT) n_mismatch,
                 round(avg((court_province <> province)::INT), 4) mismatch_rate,
                 mode(CASE WHEN court_province <> province THEN province END) top_other_province
          FROM meta WHERE court_province IS NOT NULL AND province IS NOT NULL
          GROUP BY 1 ORDER BY mismatch_rate DESC"""), "a_province_mismatch_by_court_prov")
save(q("""SELECT judgment_year, count(*) n, round(avg((court_province IS NOT NULL)::INT),4) share_parsed
          FROM meta GROUP BY 1 ORDER BY 1"""), "a_court_prov_parse_by_year")

# ---------- (b) Shanghai amount missingness
save(q("""SELECT CASE WHEN m.court_province='上海市' OR m.province='上海市' THEN '上海' ELSE '其他省份' END grp,
                 (m.court_province='上海市') court_is_sh, (m.province='上海市') incident_is_sh,
                 count(*) n_theft, round(avg((o.value_yuan IS NULL)::INT),4) value_missing,
                 round(avg((o.sentence_months IS NULL)::INT),4) sentence_missing
          FROM meta m JOIN theft t USING(doc_id) JOIN old o USING(doc_id)
          WHERE t.is_theft_old GROUP BY ALL ORDER BY n_theft DESC"""), "b_value_missing_sh_vs_all")
save(q("""SELECT (m.province='上海市') is_sh, count(*) n_theft,
                 round(avg((d.len_judgment=0)::INT),4) judgment_empty,
                 round(avg((d.len_judgment<500)::INT),4) judgment_lt500,
                 median(d.len_judgment) med_len_judgment, median(d.len_details) med_len_details,
                 round(avg((d.judgment LIKE '%元%')::INT),4) has_yuan,
                 round(avg(regexp_matches(d.judgment, '价值人民币[零一二三四五六七八九十百千万亿0-9]+元')::INT),4) re1_hit,
                 round(avg(regexp_matches(d.judgment, '(盗窃|窃取|骗取|盗走|取走|转走)(被害人[^的]{0,10})?[零一二三四五六七八九十百千万亿0-9]+元')::INT),4) re2_hit,
                 round(avg(regexp_matches(d.judgment, '数额(为|人民币)?[零一二三四五六七八九十百千万亿0-9]+元')::INT),4) re3_hit,
                 round(avg(regexp_matches(d.judgment, '价值[^。，；]{0,10}?[0-9,.]+元')::INT),4) value_loose_hit,
                 round(avg(regexp_matches(d.judgment, '[0-9]+\\.[0-9]+元|[0-9],[0-9]{3}')::INT),4) decimal_or_comma
          FROM meta m JOIN theft t USING(doc_id) JOIN docs d USING(doc_id)
          WHERE t.is_theft_old GROUP BY 1"""), "b_sh_text_profile")

sh_ids = q("""SELECT doc_id FROM meta m JOIN theft t USING(doc_id)
              WHERE t.is_theft_old AND m.province='上海市'""")["doc_id"].to_numpy()
rng = np.random.default_rng(paths.SEED)
pick = np.sort(rng.choice(sh_ids, size=min(20, len(sh_ids)), replace=False))
sh = load_texts(pick, columns=["case_number", "judgment", "len_judgment"])
old = q(f"SELECT doc_id, value_yuan FROM old WHERE doc_id IN ({','.join(map(str, pick))})")
sh = sh.merge(old, on="doc_id")
rows = []
for r in sh.itertuples():
    sents = [s for s in re.split(r"[。；;]", r.judgment or "") if "元" in s]
    rows.append({"doc_id": r.doc_id, "case_number": r.case_number, "len_judgment": r.len_judgment,
                 "old_value_yuan": r.value_yuan, "n_yuan_sents": len(sents),
                 "yuan_snippets": " || ".join(s.strip()[:160] for s in sents[:6])})
save(pd.DataFrame(rows), "b_shanghai_sample20_snippets")

# ---------- (c) document structure
MARKERS = ["公诉机关指控", "经审理查明", "本院认为", "判决如下"]
sel = ",\n".join(f"round(avg(contains(coalesce(d.judgment,''), '{m}')::INT),4) AS \"{m}\"" for m in MARKERS)
save(q(f"""SELECT 'all' grp, count(*) n, {sel},
                  round(avg((d.len_details=0)::INT),4) details_empty,
                  round(avg((d.len_judgment=0)::INT),4) judgment_empty
           FROM docs d
           UNION ALL
           SELECT 'theft_old', count(*), {sel},
                  round(avg((d.len_details=0)::INT),4), round(avg((d.len_judgment=0)::INT),4)
           FROM docs d JOIN theft t USING(doc_id) WHERE t.is_theft_old"""), "c_structure_markers")
save(q("""SELECT quantile_cont(len_judgment, [0.01,0.25,0.5,0.75,0.99]) q_len_judgment,
                 max(len_judgment) max_len_judgment,
                 quantile_cont(len_details, [0.01,0.25,0.5,0.75,0.99]) q_len_details,
                 max(len_details) max_len_details,
                 (SELECT count(*) FROM docs WHERE len_judgment = (SELECT max(len_judgment) FROM docs)) n_at_max
          FROM docs"""), "c_length_quantiles")

# ---------- (d) years
ct = q("""SELECT incident_year, judgment_year, count(*) n FROM meta GROUP BY ALL""")
save(ct.pivot_table(index="incident_year", columns="judgment_year", values="n",
                    aggfunc="sum", fill_value=0, dropna=False).reset_index(), "d_year_crosstab")
save(q("""SELECT count(*) n,
                 avg((incident_year IS NULL)::INT) incident_year_missing,
                 avg((judgment_year IS NULL)::INT) judgment_year_missing,
                 avg(CASE WHEN incident_year IS NOT NULL AND judgment_year IS NOT NULL
                          THEN (abs(judgment_year-incident_year)>=3)::INT END) diff_ge3,
                 avg(CASE WHEN incident_year IS NOT NULL AND judgment_year IS NOT NULL
                          THEN (incident_year>judgment_year)::INT END) incident_after_judgment
          FROM meta"""), "d_year_summary")

# ---------- (e) doc kind
save(q("""SELECT doc_kind, count(*) n, round(count(*)/sum(count(*)) OVER (),4) pct
          FROM meta GROUP BY 1 ORDER BY n DESC"""), "e_doc_kind")
# second-instance docs: find referenced first-instance case numbers in text, check presence
zhong = q("""SELECT m.doc_id, m.case_number, d.judgment FROM meta m JOIN docs d USING(doc_id)
             WHERE m.doc_kind='刑终'""")
pat = re.compile(r"[(（]\d{4}[)）][^()（）号，。]{1,20}?刑[^()（）号，。]{0,6}?初[^()（）号，。]{0,6}?\d+号")
zhong["first_inst_cn"] = zhong["judgment"].fillna("").map(
    lambda s: (m.group(0).replace("（", "(").replace("）", ")") if (m := pat.search(s)) else None))
all_cn = set(q("SELECT replace(replace(case_number,'（','('),'）',')') cn FROM meta")["cn"])
zhong["first_inst_in_data"] = zhong["first_inst_cn"].map(lambda c: c in all_cn if c else False)
save(pd.DataFrame([{
    "n_second_instance": len(zhong),
    "n_first_inst_cn_found_in_text": int(zhong["first_inst_cn"].notna().sum()),
    "n_first_inst_present_in_data": int(zhong["first_inst_in_data"].sum()),
}]), "e_second_instance_overlap")
save(q("""SELECT count(*) n_rows, count(DISTINCT case_number) n_distinct_cn,
                 count(*) - count(DISTINCT case_number) - (count(*) - count(case_number)) n_dup_extra
          FROM meta"""), "e_case_number_dups")

save(q("""WITH x AS (SELECT m.doc_id, m.case_number, m.court_name, m.doc_kind,
                            md5(coalesce(d.judgment,'')) h, d.len_judgment
                     FROM meta m JOIN docs d USING(doc_id))
          SELECT count(*) n_rows,
                 count(*) - count(DISTINCT case_number) extra_rows_by_cn,
                 count(*) - count(DISTINCT (case_number, court_name)) extra_rows_by_cn_court,
                 count(*) - count(DISTINCT h) extra_rows_by_identical_judgment_text,
                 count(*) - count(DISTINCT (case_number, h)) extra_rows_by_cn_and_text,
                 sum((len_judgment=0)::INT) n_empty_judgment
          FROM x"""), "e_duplicates")
save(q("""WITH x AS (SELECT m.doc_id, m.judgment_year, t.is_theft_old,
                            md5(coalesce(d.judgment,'')) h
                     FROM meta m JOIN docs d USING(doc_id) JOIN theft t USING(doc_id)
                     WHERE d.len_judgment>0),
               c AS (SELECT h, count(*) k FROM x GROUP BY 1)
          SELECT x.judgment_year, count(*) n, sum((c.k>1)::INT) n_in_text_dup_group,
                 round(avg((c.k>1)::INT),4) share_in_text_dup_group
          FROM x JOIN c USING(h) GROUP BY 1 ORDER BY 1"""), "e_text_dup_by_year")

print(f"\nDiagnostics done in {time.time() - t0:,.0f}s")
