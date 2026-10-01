"""T06a2 rule fixes: residence wordings, counsel attribution, explicit-only procedure."""
import pytest

from src import paths
from src.extract_rules import counsel_for, defendant_blocks, explicit_procedure, extract_doc, parse_identity

needs_data = pytest.mark.skipif(not paths.DOCS_TEXT.exists(), reason="working data not built")


# ----------------------------------------------------------------------------- residence wordings
@pytest.mark.parametrize("para,hukou,res", [
    ("被告人唐某，男，户籍所在地湖南省沅陵县，捕前住址珠海市香洲区，司机。", "湖南省沅陵县", "珠海市香洲区"),   # 捕前住址
    ("被告人甲，男，汉族，现住址浙江省杭州市西湖区某路，无业。", None, "浙江省杭州市西湖区某路"),                 # 现住址
    ("被告人杜某，男，中专文化，务工，户籍地及现住址山东省潍坊市。", "山东省潍坊市", "山东省潍坊市"),         # 户籍地及现住址
    ("被告人乙，男，户籍所在地及住址河南省南召县，农民。", "河南省南召县", "河南省南召县"),                     # 户籍所在地及住址
    ("被告人袁某，男，无业，户籍地同现住址长沙市望城区。", "长沙市望城区", "长沙市望城区"),                     # 同现住址
    ("被告人丙，男，住院治疗后于2018年出院，家住某县。", None, "某县"),                                          # 住院 is not an address
])
def test_residence_wordings(para, hukou, res):
    out = parse_identity(para, None)
    assert out["hukou_raw"] == hukou
    assert out["residence_raw"] == res


# ----------------------------------------------------------------------------- counsel attribution
# The three T06a disagreements (docs 138946, 77087, 963677) all have the same structure: the counsel
# line follows the SECOND defendant's paragraph and the procedural sentence reads
# "被告人<second>及其辩护人<name>到庭". The first defendant therefore has no counsel (model A was right,
# the old rule, which looked for 辩护人 anywhere before the charge, was wrong).
TWO_DEF = ("公诉机关某检察院。被告人姜某，男，1971年5月5日生，住某市。因涉嫌盗窃被刑事拘留。现羁押于某看守所。"
           "被告人于某，男，1975年5月6日生，住某市。因涉嫌盗窃被刑事拘留。现羁押于某看守所。"
           "辩护人赵英，吉林瑞邦律师事务所律师。")
PROC = "某检察院以某号起诉书指控被告人姜某、于某犯盗窃罪。被告人姜某，被告人于某及其辩护人赵英到庭参加诉讼。"


def test_counsel_belongs_to_the_block_it_follows():
    blocks = defendant_blocks(TWO_DEF)
    names = [n for n, _ in blocks]
    assert names == ["姜某", "于某"]
    assert counsel_for("姜某", blocks[0][1], TWO_DEF + PROC, names) is False
    assert counsel_for("于某", blocks[1][1], TWO_DEF + PROC, names) is True


def test_counsel_named_in_procedural_sentence():
    header = "公诉机关某检察院。被告人甲，男，住某县。被告人乙，男，住某县。辩护人丙，某律师事务所律师。"
    pre = header + "被告人甲的辩护人丁、被告人乙及其辩护人丙到庭参加诉讼。"
    blocks = defendant_blocks(header)
    names = [n for n, _ in blocks]
    assert counsel_for("甲", blocks[0][1], pre, names) is True     # "被告人甲的辩护人"
    assert counsel_for("乙", blocks[1][1], pre, names) is True


def test_single_defendant_and_explicit_none():
    h = "公诉机关某检察院。被告人甲，男，住某县。"
    assert counsel_for("甲", defendant_blocks(h)[0][1], h + "辩护人乙到庭。", ["甲"]) is True
    h2 = "被告人基本情况被告人杜某，男，户籍地及现住址山东省潍坊市。辩护人无公诉机关指控"
    assert counsel_for("杜某", defendant_blocks(h2)[0][1], h2, ["杜某"]) is False


# ----------------------------------------------------------------------------- explicit procedure only
@pytest.mark.parametrize("text,proc", [
    ("本院依法适用简易程序，实行独任审判，公开开庭审理了本案。", "简易"),
    ("本院适用刑事案件速裁程序，实行独任审判。", "速裁"),
    ("本院在审理过程中发现本案不宜适用简易程序，于2018年10月10日决定适用普通程序，并依法组成合议庭。", "普通"),
    ("本院依法组成合议庭，公开开庭审理了本案。", None),                     # panel alone -> NA (codebook v2)
    ("本院受理后，实行独任审判，公开开庭审理了本案。", None),               # sole judge alone -> NA
    ("检察院建议适用简易程序。本院审查认为本案不宜适用简易程序，依法适用普通程序。", "普通"),
    ("公诉机关某检察院适用程序速裁程序被告人冯某", "速裁"),                 # structured Nanning format
    ("山东省青岛市城阳区人民法院刑事判决书（2015）城刑（速）初字第103号被告人张某。", "速裁"),  # case-number marker
    ("本院依法转为普通程序，组成合议庭审理。", "普通"),
])
def test_explicit_procedure(text, proc):
    assert explicit_procedure(text) == proc


def test_panel_trial_recorded_separately():
    text = ("某法院刑事判决书公诉机关某检察院。被告人甲，男，住某县。因涉嫌盗窃被逮捕。"
            "某检察院以某号起诉书指控被告人甲犯盗窃罪，于某日提起公诉。本院依法组成合议庭，公开开庭审理了本案。"
            "公诉机关指控：甲窃得现金1000元。本院认为，被告人甲构成盗窃罪。判决如下：被告人甲犯盗窃罪，判处拘役三个月。")
    d, _ = extract_doc(1, text)
    assert d["procedure"] is None and d["panel_trial"] is True


# ----------------------------------------------------------------------------- the T06a documents
@needs_data
@pytest.mark.parametrize("doc_id", [138946, 77087, 963677])
def test_t06a_counsel_docs_first_defendant_has_no_counsel(doc_id):
    from src.io import load_texts
    t = load_texts([doc_id], columns=["judgment"]).judgment[0]
    _, defs = extract_doc(doc_id, t)
    assert defs[0]["counsel"] is False and defs[1]["counsel"] is True


@needs_data
@pytest.mark.parametrize("doc_id,hukou,res", [
    (1003218, "湖南省沅陵县", "珠海市香洲区"),
    (942846, "山东省潍坊市", "山东省潍坊市"),
    (906316, "长沙市望城区", "长沙市望城区"),
])
def test_t06a_residence_docs(doc_id, hukou, res):
    from src.io import load_texts
    t = load_texts([doc_id], columns=["judgment"]).judgment[0]
    _, defs = extract_doc(doc_id, t)
    assert defs[0]["hukou_raw"] == hukou and defs[0]["residence_raw"] == res
