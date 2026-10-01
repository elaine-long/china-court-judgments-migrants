"""Annotation columns from docs/codebook.md v1 (sections 1-6), in codebook order.

Each entry: (column, allowed values or None for free text/number, value description, one-line rule).
"""
YN = ["1", "0"]
YNNA = ["1", "0", "NA"]
PROVINCES = ["北京市", "天津市", "上海市", "重庆市", "河北省", "山西省", "辽宁省", "吉林省", "黑龙江省",
             "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省",
             "广东省", "海南省", "四川省", "贵州省", "云南省", "陕西省", "甘肃省", "青海省", "台湾省",
             "内蒙古自治区", "广西壮族自治区", "西藏自治区", "宁夏回族自治区", "新疆维吾尔自治区",
             "香港特别行政区", "澳门特别行政区", "NA"]
DENIAL_CODES = ["residence", "record", "severity", "attitude", "generic"]

COLUMNS = [
    # 1. origin
    ("hukou_text", None, "原文片段或 NA", "户籍地/户籍所在地/户口所在地后面的地名，原样抄下"),
    ("birthplace_text", None, "原文片段或 NA", "“出生于/生于 ××”中的地名"),
    ("native_text", None, "原文片段或 NA", "“××省××县人”这种籍贯写法"),
    ("residence_text", None, "原文片段或 NA", "住/现住/暂住/住所地/家住后面的地名"),
    ("origin_prov", PROVINCES, "省名或 NA", "户籍 > 出生地 > 籍贯取第一个非空项判断省份；不用住址；据县名推断的在 note 注明"),
    ("origin_city", None, "地级市名或 NA", "同上判断到地级市；直辖市填省名；县级市归入所属地级市"),
    ("nonlocal_origin", YNNA, "1 / 0 / NA", "origin_city 与法院所在地级市不同为 1，相同为 0，无法判断为 NA"),
    ("residence_local", YNNA, "1 / 0 / NA", "住址在法院所在地级市为 1，在外地为 0，没写住址为 NA"),
    ("no_fixed_residence", YN, "1 / 0", "原文写“无固定住所”“居无定所”“流窜”为 1"),
    # 2. pretrial measures
    ("ever_detained", YN, "1 / 0", "被刑事拘留过为 1；只受过行政拘留不算"),
    ("ever_arrested", YN, "1 / 0", "被逮捕过为 1，包括“决定逮捕”"),
    ("ever_bail", YN, "1 / 0", "被取保候审过为 1，不论什么阶段"),
    ("status_at_judgment", ["在押", "取保", "监视居住", "其他", "NA"], "在押/取保/监视居住/其他/NA",
     "以“现……”句为准；没有时取最后一个事件；“现在家”且此前取保算取保"),
    ("detention_days_known", YN, "1 / 0", "判决主文写明羁押起止日期（即自……起至……止）为 1"),
    # 3. procedure & counsel
    ("plea_formal", YN, "1 / 0", "出现“认罪认罚”字样为 1"),
    ("jiejie", YN, "1 / 0", "出现“具结”“具结书”“签字具结”为 1"),
    ("procedure", ["速裁", "简易", "普通", "NA"], "速裁/简易/普通/NA", "以“适用……程序”明文为准；没写为 NA；独任审判≠简易"),
    ("sentencing_rec", YN, "1 / 0", "公诉机关提出了具体量刑建议（建议判处……）为 1"),
    ("rec_adopted", YNNA, "1 / 0 / NA", "写“量刑建议适当，予以采纳”或判决与建议一致为 1；无量刑建议为 NA"),
    ("counsel_d1", ["无", "委托", "指定", "值班律师", "NA"], "无/委托/指定/值班律师/NA",
     "第一被告人辩护人类型；只写“辩护人××”算委托；“法律援助”算指定"),
    # 4. outcomes
    ("penalty_type", ["有期徒刑", "拘役", "管制", "单处罚金", "免予刑事处罚", "其他"],
     "有期徒刑/拘役/管制/单处罚金/免予刑事处罚/其他", "取主刑，以判决主文为准"),
    ("term_months", None, "数字或 NA", "主刑期限折算成月；不含缓刑考验期；数罪并罚取“决定执行”"),
    ("probation", YN, "1 / 0", "宣告缓刑为 1"),
    ("probation_months", None, "数字或 NA", "缓刑考验期折算成月"),
    ("fine_yuan", None, "数字或 NA", "罚金金额（元）"),
    # 5. probation denial
    ("denial_reason_text", None, "原句或 NA", "仅 probation=0 且三年以下有期徒刑或拘役时填；“本院认为”段中不适用缓刑的原句"),
    ("denial_reason_cat", None, "代码（可多选，分号分隔）或 NA",
     "residence; record; severity; attitude; generic（见 codebook 第 5 节）"),
    # 6. case controls
    ("amt_theft", None, "数字或 NA", "法院认定的盗窃数额（元）；有总额取总额，否则逐项相加；不计追回/退赃/赔偿/罚金"),
    ("spec_burglary", YN, "1 / 0", "“本院认为”段认定入户盗窃为 1"),
    ("spec_pickpocket", YN, "1 / 0", "“本院认为”段认定扒窃为 1"),
    ("spec_multiple", YN, "1 / 0", "“本院认为”段认定多次盗窃为 1"),
    ("spec_weapon", YN, "1 / 0", "“本院认为”段认定携带凶器盗窃为 1"),
    ("prior_record", YN, "1 / 0", "当事人段落里有任何刑事前科为 1"),
    ("recidivist", YN, "1 / 0", "法院认定“累犯”为 1"),
    ("surrender", YN, "1 / 0", "法院认定“自首”为 1"),
    ("confession", YN, "1 / 0", "法院认定“坦白/如实供述”为 1"),
    ("restitution", YN, "1 / 0", "法院认定“退赃/退赔”为 1"),
    ("forgiveness", YN, "1 / 0", "法院认定“取得谅解”为 1"),
    ("education", ["文盲", "小学", "初中", "高中或中专", "大专及以上", "NA"], "文盲/小学/初中/高中或中专/大专及以上/NA",
     "按原文归类，“肄业”归入该学段"),
    ("occupation_cat", ["无业", "务工", "务农", "个体或经商", "学生", "其他", "NA"],
     "无业/务工/务农/个体或经商/学生/其他/NA", "“无固定职业”归入无业"),
]
TAIL = [
    ("flag", ["1"], "1 或空", "拿不准时填 1，并在 note 写原因"),
    ("note", None, "文字", "一句话说明"),
    ("minutes", None, "数字", "本份用时（分钟）"),
]
