from datetime import date

import pandas as pd
import pytest

from src.extract_rules import (extract_amount, parse_penalty, parse_prior_and_measures,
                               province_of, set_prefecture_map)
from src.numparse import parse_duration_months, parse_number
from src.segment import find_spans, segment
from src.treatment import court_city_from_name, parse_judgment_date

# ----------------------------------------------------------------------------- numbers (>= 30)
NUMBERS = [
    ("1996", 1996), ("1,996", 1996), ("3，000", 3000), ("20，000", 20000), ("1,234,567", 1234567),
    ("24332.30", 24332.3), ("16.5", 16.5), ("2，983.50", 2983.5), ("6.2万", 62000), ("1.5万", 15000),
    ("3万", 30000), ("1万2千", 12000), ("三千五百", 3500), ("一千", 1000), ("两千", 2000),
    ("二千八百", 2800), ("一万", 10000), ("一万三千八百八十八", 13888), ("十", 10), ("十五", 15),
    ("二十", 20), ("一百零五", 105), ("三十万", 300000), ("五千元"[:-1], 5000), ("１２３", 123),
    ("１，２００", 1200), ("1300余", 1300), ("700多", 700), ("5,967.00", 5967), ("3千", 3000),
    ("二〇一五", 2015), ("壹万", 10000),
]


@pytest.mark.parametrize("s,v", NUMBERS)
def test_parse_number(s, v):
    assert parse_number(s) == pytest.approx(v)


DURATIONS = [("一年六个月", 18), ("八个月", 8), ("十年", 120), ("1年零3个月", 15), ("两年", 24),
             ("十五日", 0.5), ("一年又三个月", 12)]


@pytest.mark.parametrize("s,v", DURATIONS)
def test_duration(s, v):
    assert parse_duration_months(s) == pytest.approx(v, abs=0.01)


# ----------------------------------------------------------------------------- penalties (>= 20)
PENALTIES = [
    ("被告人张某犯盗窃罪，判处有期徒刑一年六个月，并处罚金人民币二千元。", "有期徒刑", 18, False, None, 2000),
    ("被告人张某犯盗窃罪，判处有期徒刑八个月，并处罚金人民币1,000元。", "有期徒刑", 8, False, None, 1000),
    ("被告人张某犯盗窃罪，判处拘役五个月，并处罚金人民币一千元。", "拘役", 5, False, None, 1000),
    ("被告人张某犯盗窃罪，判处拘役三个月，缓刑六个月，并处罚金人民币一千元。", "拘役", 3, True, 6, 1000),
    ("被告人张某犯盗窃罪，判处有期徒刑一年，缓刑一年六个月，并处罚金人民币5000元。", "有期徒刑", 12, True, 18, 5000),
    ("被告人张某犯盗窃罪，判处有期徒刑三年，缓刑五年，并处罚金人民币三万元。", "有期徒刑", 36, True, 60, 30000),
    ("被告人张某犯盗窃罪，判处有期徒刑十年，并处罚金人民币十万元。", "有期徒刑", 120, False, None, 100000),
    ("被告人张某犯盗窃罪，判处有期徒刑一年八个月，并处罚金人民币五千元；与前罪所判处的有期徒刑四年八个月，"
     "并处罚金人民币一万元的刑罚并罚，决定执行有期徒刑六年四个月，并处罚金人民币一万五千元。", "有期徒刑", 76, False, None, 15000),
    ("被告人张某犯盗窃罪，判处有期徒刑一年，并处罚金人民币二千元；犯抢劫罪，判处有期徒刑三年，并处罚金人民币三千元；"
     "决定执行有期徒刑三年六个月，并处罚金人民币五千元。", "有期徒刑", 42, False, None, 5000),
    ("被告人张某犯盗窃罪，判处罚金人民币2000元。", "单处罚金", None, False, None, 2000),
    ("被告人张某犯盗窃罪，免予刑事处罚。", "免予刑事处罚", None, False, None, None),
    ("被告人张某犯盗窃罪，判处管制六个月，并处罚金人民币一千元。", "管制", 6, False, None, 1000),
    ("被告人张某犯抢劫罪，判处无期徒刑，剥夺政治权利终身，并处没收个人全部财产。", "无期徒刑", None, False, None, None),
    ("被告人张某犯盗窃罪，判处有期徒刑七个月，并处罚金3000元(限判决生效后3内缴纳)。（刑期从判决执行之日起计算。"
     "判决执行以前先行羁押的，羁押一日折抵刑期一日，即自2018年3月29日起至2018年10月28日止。）", "有期徒刑", 7, False, None, 3000),
    ("被告人张某犯盗窃罪，判处有期徒刑1年2个月，并处罚金人民币2000元。", "有期徒刑", 14, False, None, 2000),
    ("被告人张某犯盗窃罪，判处有期徒刑二年零六个月，并处罚金人民币八千元。", "有期徒刑", 30, False, None, 8000),
    ("被告人张某犯盗窃罪，判处拘役四个月，缓刑八个月，并处罚金人民币2,000元。", "拘役", 4, True, 8, 2000),
    ("被告人张某犯盗窃罪，判处有期徒刑六个月（刑期自2014年9月15日起至2015年3月14日止），并处罚金人民币二千元。", "有期徒刑", 6, False, None, 2000),
    ("被告人张某犯盗窃罪，判处有期徒刑九年，并处罚金人民币100000元；二、被告人张某退赔被害人赵某人民币380000元", "有期徒刑", 108, False, None, 100000),
    ("被告人张某犯抢劫罪，判处有期徒刑十二年，剥夺政治权利二年，并处罚金人民币二万元。", "有期徒刑", 144, False, None, 20000),
    ("被告人张某犯盗窃罪，判处拘役一个月十五日，并处罚金人民币一千元。", "拘役", 1.5, False, None, 1000),
    ("被告人张某犯故意杀人罪，判处死刑，缓期二年执行，剥夺政治权利终身。", "死缓", None, False, None, None),
]


@pytest.mark.parametrize("text,ptype,term,prob,prob_m,fine", PENALTIES)
def test_penalty(text, ptype, term, prob, prob_m, fine):
    p = parse_penalty(text)
    assert p["penalty_type"] == ptype
    assert p["term_months"] == (pytest.approx(term, abs=0.01) if term is not None else None)
    assert p["probation"] == prob
    assert p["probation_months"] == prob_m
    assert p["fine_yuan"] == fine


def test_old_double_count_bug_fixed():
    # the old code summed two "个月" regexes; "一年六个月" must be 18, not 36
    assert parse_penalty("判处有期徒刑一年六个月")["term_months"] == 18


# ----------------------------------------------------------------------------- pretrial measures (>= 10)
MEASURES = [
    ("被告人甲，男。因涉嫌犯盗窃罪于2017年7月24日被刑事拘留，同年8月5日被逮捕。现押于某看守所。",
     dict(detained=True, detained_date=date(2017, 7, 24), arrested=True, arrested_date=date(2017, 8, 5),
          bail=False, pretrial_status_at_judgment="在押")),
    ("被告人甲，男。2018年11月10日因涉嫌犯盗窃罪被玉环市公安局刑事拘留，同年12月10日转取保候审。现在家。",
     dict(detained=True, bail=True, bail_date=date(2018, 12, 10), pretrial_status_at_judgment="取保")),
    ("被告人甲，男。因本案于2016年6月13日被羁押，次日被刑事拘留，同年7月20日被逮捕。现羁押于某看守所。",
     dict(detained_date=date(2016, 6, 14), arrested_date=date(2016, 7, 20), pretrial_status_at_judgment="在押")),
    ("被告人甲，男。2013年9月2日因涉嫌盗窃罪被刑事拘留，2013年9月28日被监视居住。",
     dict(rsl=True, rsl_date=date(2013, 9, 28), pretrial_status_at_judgment="rsl")),  # T03: rsl coded separately
    ("被告人甲，男。曾因犯盗窃罪于2010年1月19日被判处有期徒刑十个月。2017年2月18日因犯盗窃罪被刑事拘留，同年3月1日被逮捕。",
     dict(prior_record=True, prior_theft=True, detained=True, detained_date=date(2017, 2, 18),
          arrested_date=date(2017, 3, 1), pretrial_status_at_judgment="在押")),
    ("被告人甲，男。因涉嫌犯盗窃罪于2016年9月5日被刑事拘留。同月14日，检察院不批准逮捕，同日被取保候审。",
     dict(arrested=False, bail=True, bail_date=date(2016, 9, 14), pretrial_status_at_judgment="取保")),
    ("被告人甲，男，农民。因涉嫌犯盗窃罪于2017年5月4日被刑事拘留，同年6月2日被该局取保候审。2017年7月10日被本院决定逮捕，次日执行逮捕。",
     dict(bail=True, arrested=True, arrested_date=date(2017, 7, 10), pretrial_status_at_judgment="在押")),
    ("被告人甲，男。2014年1月9日因盗窃被行政拘留十日，因涉嫌犯盗窃罪于2014年1月17日被刑事拘留，同年2月21日被依法逮捕。现羁押于某看守所。",
     dict(prior_record=False, detained_date=date(2014, 1, 17), arrested_date=date(2014, 2, 21))),
    ("被告人甲，男。因涉嫌犯盗窃罪于2015年6月18日被刑事拘留，同年7月2日被逮捕，2015年8月1日被取保候审。",
     dict(bail=True, bail_date=date(2015, 8, 1), pretrial_status_at_judgment="取保")),
    ("被告人甲，女。因涉嫌犯盗窃罪于2014年8月21日被刑事拘留，2014年9月18日被批准并执行逮捕，于2014年11月21日被检察院取报候审，现取保候审于居住地。",
     dict(bail=True, pretrial_status_at_judgment="取保")),
    ("被告人甲，男。无前科。因涉嫌盗窃罪于2019年3月1日被取保候审。",
     dict(prior_record=False, detained=False, bail=True, bail_date=date(2019, 3, 1), pretrial_status_at_judgment="取保")),
]


@pytest.mark.parametrize("para,expect", MEASURES)
def test_measures(para, expect):
    out = parse_prior_and_measures(para)
    for k, v in expect.items():
        assert out[k] == v, k


# ----------------------------------------------------------------------------- segmentation (>= 10)
STD = ("某某法院刑事判决书（2018）粤0902刑初1号公诉机关某检察院。被告人甲，男，1990年1月1日出生。因涉嫌犯盗窃罪被刑事拘留。"
       "某检察院以某检刑诉（2018）1号起诉书指控被告人甲犯盗窃罪，于2018年1月1日向本院提起公诉。"
       "公诉机关指控：2017年被告人甲窃得手机一部，价值1000元。经审理查明，2017年被告人甲窃得手机一部。"
       "本院认为，被告人甲构成盗窃罪。依照某规定，判决如下：被告人甲犯盗窃罪，判处拘役三个月。")
SEG_CASES = [
    (STD, {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("经审理查明", "经本院审理查明"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("经审理查明", "经法庭审理，查明"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("经审理查明，", ""), {"header", "charge", "reasoning", "verdict"}),
    (STD.replace("公诉机关指控：", "某检察院指控，"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("本院认为", "本庭认为"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("判决如下", "判�如下"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("依照某规定，判决如下：", "依照某规定。"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("本院认为", "判决理由").replace("判决如下：", "判决结果"), {"header", "charge", "facts", "reasoning", "verdict"}),
    (STD.replace("被告人甲，男", "被告�甲，男"), {"header", "charge", "facts", "reasoning", "verdict"}),
    ("某法院刑事判决书公诉机关某检察院。被告人乙。某检察院以某号起诉书指控被告人乙犯盗窃罪，于某日提起公诉。"
     "被告人乙对此无异议并认罪认罚。经本院审理，本案事实清楚，证据充分。依照某规定，判决如下：被告人乙犯盗窃罪，判处拘役一个月。",
     {"header", "reasoning", "verdict"}),
]


@pytest.mark.parametrize("text,found", SEG_CASES)
def test_segments(text, found):
    sp = find_spans(text)
    assert {k for k, v in sp.items() if v} == found


def test_segment_contents():
    seg = segment(STD)
    assert seg["header"].endswith("被刑事拘留。")
    assert seg["facts"].startswith("经审理查明")
    assert seg["reasoning"].startswith("本院认为")
    assert seg["verdict"].startswith("：被告人甲犯盗窃罪")


# ----------------------------------------------------------------------------- city / date / place (>= 10)
PREFS = {"茂名市", "三门峡市", "南京市", "杭州市", "黔南布依族苗族自治州", "大兴安岭地区", "锡林郭勒盟"}
COURTS = [
    ("茂名市茂南区人民法院", "茂名市"), ("南京市玄武区人民法院", "南京市"), ("江苏省南京市中级人民法院", "南京市"),
    ("浙江省杭州市西湖区人民法院", "杭州市"), ("灵宝市人民法院", None), ("黔南布依族苗族自治州中级人民法院", "黔南布依族苗族自治州"),
    ("大兴安岭地区中级人民法院", "大兴安岭地区"), ("锡林郭勒盟中级人民法院", "锡林郭勒盟"), ("某县人民法院", None),
]


@pytest.mark.parametrize("court,city", COURTS)
def test_court_city(court, city):
    assert court_city_from_name(court, PREFS) == city


DATES = [("2018-08-28", "2018-08-28"), ("2018年8月28日", "2018-08-28"), ("2015年十二月二十九日", "2015-12-29"),
         ("二〇一五年一月五日", "2015-01-05"), ("2016年十月三十日", "2016-10-30"), ("", None), (None, None)]


@pytest.mark.parametrize("s,d", DATES)
def test_judgment_date(s, d):
    got = parse_judgment_date(s)
    assert (pd.isna(got) and d is None) or got == pd.Timestamp(d)


def test_province_of():
    set_prefecture_map({"吉林市": "吉林省", "海南藏族自治州": "青海省", "朝阳市": "辽宁省", "南京市": "江苏省"})
    assert province_of("本市") == "LOCAL"
    assert province_of("吉林市船营区") == "吉林省"
    assert province_of("吉林省长春市") == "吉林省"
    assert province_of("海南藏族自治州共和县") == "青海省"
    assert province_of("海南省海口市") == "海南省"
    assert province_of("朝阳区") is None  # district-level stem, ambiguous
    assert province_of("南京浦口区") == "江苏省"
    assert province_of("广西陆川县") == "广西壮族自治区"
    assert province_of("枞阳县") is None
    assert province_of(None) is None


# ----------------------------------------------------------------------------- amounts
def test_amounts():
    assert extract_amount("本院认为，被告人盗窃他人财物，合计价值人民币1,996元，数额较大。", None, None) == (1996, "reasoning_total")
    assert extract_amount(None, "窃得手机一部（经鉴定，价值人民币1,582元，已发还被害人）及手机一部(经鉴定价值人民币557元，已从某处扣押)", None)[0] == 2139
    assert extract_amount("本院认为，被告人盗窃，数额较大，退赔被害人3000元。", "窃得现金人民币3，000元", None) == (3000, "facts_sum")
    assert extract_amount(None, None, "盗窃物品共计价值人民币6.2万元") == (62000, "charge_total")
    assert extract_amount(None, "以1000元的价格予以收购。经价格认定，该车价值人民币5292元。", None)[0] == 5292
