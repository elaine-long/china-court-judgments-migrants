import pytest

from src import paths
from src.extract_rules import extract_doc, origin_fields, parse_prior_and_measures
from src.gazetteer import GAZ, resolve_place

needs_gaz = pytest.mark.skipif(not GAZ.exists(), reason="gazetteer not built")
needs_data = pytest.mark.skipif(not paths.DOCS_TEXT.exists(), reason="working data not built")

# ----------------------------------------------------------------------------- resolve_place (>= 20)
PLACES = [
    # raw, court_province, expected (province, prefecture, status)
    ("江苏省淮安市盱眙县", None, ("江苏省", "淮安市", "resolved")),          # full province-city-county
    ("盱眙县", None, ("江苏省", "淮安市", "resolved")),                      # county only, unique
    ("长子县", None, ("山西省", "长治市", "resolved")),
    ("贵州省纳雍县", None, ("贵州省", "毕节市", "resolved")),
    ("广西陆川县", None, ("广西壮族自治区", "玉林市", "resolved")),           # province short name
    ("内蒙古赤峰市", None, ("内蒙古自治区", "赤峰市", "resolved")),
    ("云南省文山壮族苗族自治州富宁县", None, ("云南省", "文山壮族苗族自治州", "resolved")),
    ("四川省西昌市人", None, ("四川省", "凉山彝族自治州", "resolved")),        # native place "…人"
    ("湖南省平江县人", None, ("湖南省", "岳阳市", "resolved")),
    ("耒阳", None, ("湖南省", "衡阳市", "resolved")),                        # short county name
    ("蓟县", None, ("天津市", "天津市", "resolved")),                        # historical: 蓟县 -> 蓟州区 (2016)
    ("襄樊市", None, ("湖北省", "襄阳市", "resolved")),                      # historical: renamed 2010
    ("安徽省巢湖市", None, ("安徽省", "合肥市", "resolved")),                # 巢湖 prefecture abolished 2011
    ("上海市浦东新区", None, ("上海市", "上海市", "resolved")),              # municipality
    ("浙江省衢州市柯城区某某街道某号", None, ("浙江省", "衢州市", "resolved")),  # trailing address
    ("朝阳区", "北京市", ("北京市", "北京市", "ambiguous_resolved_by_court_prov")),  # 北京 / 长春
    ("朝阳区", "吉林省", ("吉林省", "长春市", "ambiguous_resolved_by_court_prov")),
    ("朝阳区", None, (None, None, "ambiguous")),
    ("新华区", "河南省", ("河南省", "平顶山市", "ambiguous_resolved_by_court_prov")),
    ("新华区", "河北省", (None, None, "ambiguous")),                         # two 新华区 in Hebei
    ("湖北省", None, ("湖北省", None, "province_only")),
    ("本市", "浙江省", ("浙江省", None, "local_word")),
    ("XXX", None, (None, None, "unresolved")),
    ("中国湖北省利川市", None, ("湖北省", "恩施土家族苗族自治州", "resolved")),
]


@needs_gaz
@pytest.mark.parametrize("raw,court,exp", PLACES)
def test_resolve_place(raw, court, exp):
    p, f, _, st = resolve_place(raw, court)
    assert (p, f, st) == exp


# ----------------------------------------------------------------------------- origin & migrant definitions
@needs_gaz
def test_origin_priority_and_migrant():
    row = {"hukou_raw": "湖北省利川市", "birthplace_raw": "浙江省衢州市", "native_raw": None,
           "residence_raw": "衢州市柯城区"}
    o = origin_fields(row, "浙江省", "衢州市")
    assert o["origin_source"] == "hukou" and o["origin_prefecture"] == "恩施土家族苗族自治州"
    assert o["migrant_city"] is True and o["migrant_prov"] is True
    assert o["migrant_city_nores"] is True and o["local_residence"] is True


@needs_gaz
def test_residence_only_migrant():
    # doc 77426: only "家住贵州省纳雍县", court in 衢州
    o = origin_fields({"residence_raw": "贵州省纳雍县"}, "浙江省", "衢州市")
    assert o["origin_source"] == "residence" and o["migrant_city"] is True
    assert o["migrant_city_nores"] is None


@needs_gaz
def test_same_province_other_city():
    o = origin_fields({"birthplace_raw": "浙江省温州市"}, "浙江省", "衢州市")
    assert o["migrant_city"] is True and o["migrant_prov"] is False


@needs_gaz
def test_province_only_same_province_is_missing_at_city_level():
    o = origin_fields({"hukou_raw": "浙江省"}, "浙江省", "衢州市")
    assert o["migrant_city"] is None and o["migrant_prov"] is False


# ----------------------------------------------------------------------------- status at judgment
STATUS = [
    ("被告人甲。因涉嫌犯盗窃罪，于2018年2月22日被取保候审，同年8月8日经本院决定逮捕，后看守所拒收，同日继续取保候审。现取保候审于其住所地。", "取保"),
    ("被告人甲。因涉嫌犯盗窃罪于2017年5月4日被刑事拘留，同年6月2日被取保候审。现被羁押于某看守所。", "在押"),
    ("被告人甲。因涉嫌犯盗窃罪于2013年9月2日被刑事拘留，2013年9月28日被监视居住。", "rsl"),
    ("被告人甲。因涉嫌犯盗窃罪于2013年9月2日被刑事拘留。现被监视居住。", "rsl"),
    ("被告人甲。因涉嫌犯盗窃罪于2019年3月1日被取保候审。现住某县某村。", "取保"),
    ("被告人甲。因涉嫌犯盗窃罪于2019年3月1日被刑事拘留，同月5日被逮捕。现在某监狱服刑。", "在押"),
]


@pytest.mark.parametrize("para,st", STATUS)
def test_status(para, st):
    assert parse_prior_and_measures(para)["pretrial_status_at_judgment"] == st


# ----------------------------------------------------------------------------- leniency indicators, pseudonyms, outliers
SYN = ("某法院刑事判决书公诉机关某检察院。被告人王某，男，1990年1月1日出生，住某县。因涉嫌犯盗窃罪被刑事拘留。现羁押于某看守所。"
       "被告人王某，男，1985年2月2日出生，住某县。因涉嫌犯盗窃罪被取保候审。现取保候审于其住所地。"
       "某检察院以某号起诉书指控被告人王某、王某犯盗窃罪，于某日提起公诉。本院适用速裁程序审理。"
       "公诉机关指控：两被告人窃得财物价值人民币50000100000元。被告人对指控无异议并签字具结。"
       "本院认为，被告人构成盗窃罪。判决如下：一、被告人王某犯盗窃罪，判处有期徒刑一年，并处罚金人民币2000元。"
       "二、被告人王某犯盗窃罪，判处拘役三个月，缓刑六个月，并处罚金人民币50000100000元。")


def test_synthetic_pseudonym_leniency_outlier():
    d, defs = extract_doc(1, SYN)
    assert d["dup_pseudonym"] and d["n_defendants"] == 2
    assert d["speedy"] and d["jiejie"] and not d["plea_formal"] and d["leniency_proc"]
    assert d["amt_outlier"] and d["amt_yuan_ctrl"] is None and d["amt_yuan"] > 1e7
    assert [x["pretrial_status_at_judgment"] for x in defs] == ["在押", "取保"]
    assert [x["term_months"] for x in defs] == [12, 3]
    assert defs[1]["probation"] and defs[1]["fine_outlier"] and defs[1]["fine_yuan_ctrl"] is None
    assert not defs[0]["fine_outlier"] and defs[0]["fine_yuan_ctrl"] == 2000


@needs_data
@needs_gaz
def test_doc_705157_jiejie_speedy():
    from src.io import load_texts
    t = load_texts([705157], columns=["judgment"]).judgment[0]
    d, _ = extract_doc(705157, t, None, "湖北省", "武汉市")
    assert d["jiejie"] and d["speedy"] and d["leniency_proc"]


@needs_data
@needs_gaz
def test_doc_353767_status_and_pseudonym():
    from src.io import load_texts
    t = load_texts([353767], columns=["judgment"]).judgment[0]
    d, defs = extract_doc(353767, t, None, "江苏省", "淮安市")
    assert defs[0]["pretrial_status_at_judgment"] == "取保"
    assert d["dup_pseudonym"] and d["n_defendants"] == 4
    assert [x["term_months"] for x in defs] == [14, 6, 5, 3]
