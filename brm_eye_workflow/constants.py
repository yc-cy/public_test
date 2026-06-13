"""Shared constants for the leakage-aware eye-movement workflow."""

from __future__ import annotations

from typing import Dict, List

EYE_TARGETS: List[str] = [
    "IA_FIRST_FIXATION_DURATION",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_FIRST_RUN_FIXATION_COUNT",
    "IA_REGRESSION_IN_COUNT",
    "IA_REGRESSION_PATH_DURATION",
    "IA_DWELL_TIME",
    "IA_FIXATION_%",
    "IA_SKIP",
    "IA_RUN_COUNT",
]

TARGET_LABEL_ZH: Dict[str, str] = {
    "IA_FIRST_FIXATION_DURATION": "首次注视时长",
    "IA_FIRST_RUN_DWELL_TIME": "首次注视停留时间",
    "IA_FIRST_RUN_FIXATION_COUNT": "首次注视次数",
    "IA_REGRESSION_IN_COUNT": "回视进入次数",
    "IA_REGRESSION_PATH_DURATION": "回视路径时长",
    "IA_DWELL_TIME": "总停留时间",
    "IA_FIXATION_%": "注视百分比",
    "IA_SKIP": "是否跳过",
    "IA_RUN_COUNT": "注视序列次数",
}

COUNTRY_LABEL_ZH: Dict[str, str] = {
    "V": "越南语背景",
    "L": "老挝语背景",
    "T": "泰语背景",
    "K": "柬埔寨语背景",
    "M": "缅甸语背景",
}

AOA_BINS_ZH: Dict[str, str] = {
    "(-0.001, 12.0]": "儿童期",
    "(12.0, 18.0]": "青年期",
    "(18.0, 100.0]": "成年期",
}

LOE_BINS_ZH: Dict[str, str] = {
    "(-0.001, 2.0]": "初期接触阶段",
    "(2.0, 5.0]": "过渡习得阶段",
    "(5.0, 100.0]": "稳定运用阶段",
}

HSK_BINS_ZH: Dict[str, str] = {
    "(0, 2.0]": "初级入门阶段",
    "(2.0, 4.0]": "中级发展阶段",
    "(4.0, 6.1]": "高级熟练阶段",
}
