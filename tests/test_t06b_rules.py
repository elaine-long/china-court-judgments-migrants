"""T06b §1: the three status parse failures found in T06a2 (all three made the defendant paragraph
undetectable, so status came out NA)."""
import pytest

from src import paths
from src.extract_rules import extract_doc, find_defendants, normalize

needs_data = pytest.mark.skipif(not paths.DOCS_TEXT.exists(), reason="working data not built")

VERDICT = "本院认为，被告人构成盗窃罪。判决如下：被告人{n}犯盗窃罪，判处拘役三个月。"


def test_literal_newline_escapes_removed():
    assert normalize("＼n山东省莱州市人民法院＼n＼n刑事判决书") == "山东省莱州市人民法院刑事判决书"
    assert normalize("公诉机关某院。\\n被告人李某，男") == "公诉机关某院。被告人李某，男"
    t = ("＼n山东省莱州市人民法院＼n＼n刑事判决书＼n＼n（2013）莱州刑初字第332号＼n＼n公诉机关莱州市人民检察院。＼n"
         "被告人李某，男，1983年8月18日出生。因涉嫌犯盗窃罪于同年4月25日被刑事拘留，5月29日被逮捕，现羁押于莱州市看守所。＼n"
         "莱州市人民检察院以莱检刑诉[2013]228号起诉书指控被告人李某犯盗窃罪。" + VERDICT.format(n="李某"))
    _, defs = extract_doc(1, t)
    assert defs[0]["def_name"] == "李某" and defs[0]["pretrial_status_at_judgment"] == "在押"


def test_defendant_right_after_case_number():
    h = "重庆市巴南区人民法院刑事判决书(2016)渝0113刑初696号被告人陈某，女，出生于重庆市渝北区。现关押在重庆市巴南区看守所。"
    assert [n for n, _ in find_defendants(h)] == ["陈某"]
    t = h + "重庆市巴南区人民检察院以渝巴检刑诉〔2016〕585号起诉书指控被告人陈某犯盗窃罪。" + VERDICT.format(n="陈某")
    _, defs = extract_doc(1, t)
    assert defs[0]["pretrial_status_at_judgment"] == "在押"


@pytest.mark.parametrize("header,name", [
    ("公诉机关某检察院。被告人彭某因本案，于2014年11月28日被羁押，同月29日被刑事拘留。现羁押于某看守所。", "彭某"),
    ("公诉机关某检察院。被告人甲曾因犯盗窃罪被判处有期徒刑六个月。现羁押于某看守所。", "甲"),
    ("公诉机关某检察院。被告人乙于2014年3月1日因涉嫌盗窃被刑事拘留。现羁押于某看守所。", "乙"),
    ("公诉机关某检察院。被告人丙因涉嫌盗窃于2015年1月1日被取保候审。", "丙"),
])
def test_name_followed_by_case_history(header, name):
    assert [n for n, _ in find_defendants(header)] == [name]


def test_continuation_sentence_is_not_a_new_defendant():
    # a later "被告人钱某因涉嫌…" of the same person must not open a second paragraph
    h = ("公诉机关某检察院。被告人钱某，男，1960年2月2日出生，住某县。被告人钱某曾因犯罪被判处有期徒刑三年。"
         "被告人钱某因涉嫌犯盗窃罪，于2018年2月22日被取保候审。现取保候审于其住所地。")
    assert [n for n, _ in find_defendants(h)] == ["钱某"]


@needs_data
@pytest.mark.parametrize("doc_id", [105309, 549320, 510265])
def test_t06a2_status_docs(doc_id):
    from src.io import load_texts
    t = load_texts([doc_id], columns=["judgment"]).judgment[0]
    _, defs = extract_doc(doc_id, t)
    assert defs and defs[0]["pretrial_status_at_judgment"] == "在押"
