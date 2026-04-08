import os
import re
import io
import zipfile
import tempfile
import random

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF

random.seed(42)


# ==========================================
# 0. 기본 설정
# ==========================================
st.set_page_config(page_title="LEAP V3 테스트 시스템", layout="wide")

FONT_PATH = "NanumBarunGothic.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rc("font", family="NanumBarunGothic")
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 1. 공통 유틸
# ==========================================
def safe_mean(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return np.nan if len(vals) == 0 else vals.mean()


def safe_fmt(value, fmt=".4f", na="NA"):
    if pd.isna(value):
        return na
    try:
        return format(float(value), fmt)
    except Exception:
        return str(value)


def severity_rank(label: str) -> int:
    if "진행성" in label:
        return 5
    if "혼합형" in label:
        return 4
    if "전신 부종" in label:
        return 3
    if "초기 림프 dysfunction" in label or "경계형 림프부종" in label:
        return 2
    if "주의" in label or "경계형 비대칭" in label:
        return 1
    return 0


def extract_file_info(filename: str):
    stem = os.path.splitext(filename)[0]
    date_match = re.search(r"(\d{8}|\d{6})$", stem)
    raw_date = date_match.group(1) if date_match else None

    patient_name = stem[: stem.rfind(raw_date)].rstrip("_- ") if raw_date else stem
    patient_name = patient_name if patient_name else "환자"

    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    if raw_date:
        if len(raw_date) == 6:
            report_date = f"20{raw_date[:2]}-{raw_date[2:4]}-{raw_date[4:]}"
        elif len(raw_date) == 8:
            report_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    return patient_name, report_date


def extract_patient_name_from_sheet(sheet_name: str) -> str:
    parts = re.split(r"[_-]", str(sheet_name).strip())
    name = parts[0].strip() if parts else ""
    return name if name else "환자"


def init_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf_font = "Arial"

    if os.path.exists(FONT_PATH):
        try:
            pdf.add_font("Nanum", "", FONT_PATH, uni=True)
            pdf_font = "Nanum"
        except Exception:
            pdf_font = "Arial"

    return pdf, pdf_font


def check_password():
    if "password_entered" not in st.session_state:
        st.session_state["password_entered"] = False

    if not st.session_state["password_entered"]:
        st.title("🔒 LEAP V3 접근")
        password = st.text_input("접속 비밀번호를 입력하세요:", type="password")
        if st.button("로그인"):
            admin_pw = st.secrets.get("admin_password", None)
            if admin_pw is not None and password == admin_pw:
                st.session_state["password_entered"] = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
        return False
    return True


# ==========================================
# 2. 해석 표시 유틸
# ==========================================
def calculate_corr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s))
    y = s.values
    return np.corrcoef(x, y)[0, 1]


def fmt_ratio_interp(x):
    if pd.isna(x):
        return "NA"
    if x >= 1.05:
        return f"{x:.3f} (확증)"
    elif x >= 1.02:
        return f"{x:.3f} (경계)"
    return f"{x:.3f} (정상)"


def fmt_diff_interp(x):
    if pd.isna(x):
        return "NA"
    if x >= 0.004:
        return f"{x:+.4f} (↑ 증가)"
    elif x <= -0.002:
        return f"{x:+.4f} (↓ 감소)"
    return f"{x:+.4f} (→ 안정)"


def fmt_compare_interp(x):
    if pd.isna(x):
        return "NA"
    if x >= 0.004:
        return f"{x:+.4f} (이전 구간 대비 뚜렷한 증가)"
    elif x >= 0.002:
        return f"{x:+.4f} (이전 구간 대비 증가)"
    elif x <= -0.004:
        return f"{x:+.4f} (이전 구간 대비 뚜렷한 감소)"
    elif x <= -0.002:
        return f"{x:+.4f} (이전 구간 대비 감소)"
    return f"{x:+.4f} (이전 구간 대비 큰 변화 없음)"


def fmt_days_interp(x):
    if pd.isna(x):
        return "NA"
    x = int(x)
    if x == 0:
        return f"{x}일 (정상)"
    elif x == 1:
        return f"{x}일 (경미)"
    elif x == 2:
        return f"{x}일 (기능 저하)"
    return f"{x}일 (지속 이상)"


def fmt_cv_interp(x):
    if pd.isna(x):
        return "NA"
    if x >= 1.5:
        return f"{x:.2f}% (불안정)"
    elif x >= 1.0:
        return f"{x:.2f}% (변동 증가)"
    return f"{x:.2f}% (안정)"


def fmt_trend_interp(x):
    if pd.isna(x):
        return "NA"
    if x >= 0.5:
        return f"{x:+.2f} (↑ 상승 추세)"
    elif x <= -0.5:
        return f"{x:+.2f} (↓ 회복 추세)"
    return f"{x:+.2f} (→ 안정)"


# ==========================================
# 3. 메시지집
# ==========================================
MESSAGE_BANK = {
    "clinician": {
        "intro": [
            "일간 측정 자료를 종합하여 현재 상태를 해석합니다.",
            "최근 측정 자료를 바탕으로 임상적 해석을 제시합니다."
        ],
        "structure_normal": ["양측 비율은 정상 범위입니다."],
        "structure_borderline": ["양측 비율이 경계 수준입니다."],
        "structure_confirmed": ["양측 비율이 확증 범위입니다."],
        "state_up": ["최근 3일 기준 국소 변화는 증가 상태입니다."],
        "state_down": ["최근 3일 기준 국소 변화는 감소 상태입니다."],
        "state_stable": ["최근 3일 기준 국소 변화는 비교적 안정적입니다."],
        "function_mild": ["최근 야간 회복 실패는 경미한 수준입니다."],
        "function_impaired": ["최근 야간 회복 실패가 반복되어 기능 저하가 의심됩니다."],
        "function_persistent": ["야간 회복 기능 저하가 지속적으로 나타납니다."],
        "trend_up": ["최근 7일 추세는 상승 방향입니다."],
        "trend_down": ["최근 7일 추세는 회복 방향입니다."],
        "trend_stable": ["최근 7일 추세는 뚜렷한 방향성 없이 안정적입니다."],
        "systemic_absent": ["하지 및 체간 변화는 크지 않아 국소 변화 가능성이 높습니다."],
        "systemic_present": ["하지 또는 체간 변화가 동반되어 전신 영향 가능성이 있습니다."],
        "variability_high": ["최근 7일 변동성이 증가해 상태가 불안정할 수 있습니다."],
        "action_maintain": ["현재 관리 유지와 정기 모니터링을 권장합니다."],
        "action_followup": ["단기 추적 관찰 및 반복 측정이 필요합니다."],
        "action_consult": ["의료진 평가 및 적극적 치료 개입을 권장합니다."],
        "action_systemic_check": ["전신 체액 변화 가능성을 함께 평가할 필요가 있습니다."],
    },
    "patient": {
        "intro_default": [
            "매일 꾸준히 측정하고 관리하고 계신 {name}님, 늘 응원합니다.",
            "{name}님, 꾸준한 관리 정말 잘하고 계십니다.",
            "{name}님, 성실하게 측정을 이어가고 계셔서 좋은 흐름입니다."
        ],
        "intro_attention": [
            "{name}님, 최근 변화가 보여 조금 더 주의 깊은 관리가 필요한 시점입니다.",
            "{name}님, 최근 상태를 조금 더 세심하게 관찰할 필요가 있습니다."
        ],
        "intro_improving": [
            "{name}님, 최근 상태가 점차 좋아지고 있어 매우 긍정적인 흐름입니다."
        ],
        "analysis_intro": [
            "일간 측정하신 내용을 종합하여 분석 결과를 전해드립니다.",
            "최근 측정 데이터를 바탕으로 현재 상태를 분석해드리겠습니다."
        ],
        "structure_normal": ["현재 수술 받으신 팔과 반대쪽 팔의 체액 차이는 뚜렷하지 않습니다."],
        "structure_borderline": ["현재 수술 받으신 팔과 반대쪽 팔 사이에 약한 차이가 보입니다."],
        "structure_confirmed": ["현재 수술 받으신 팔과 반대쪽 팔 사이의 차이가 분명하게 확인됩니다."],
        "state_up": ["최근 3일 기준으로 수술 받으신 팔의 체수분 변화가 증가하는 경향입니다."],
        "state_down": ["최근 3일 기준으로 수술 받으신 팔의 체수분 변화는 감소 상태입니다."],
        "state_stable": ["최근 3일 기준으로 수술 받으신 팔의 체수분 변화는 비교적 안정적입니다."],
        "function_mild": ["최근 밤사이 회복이 다소 불안정한 날이 있었습니다."],
        "function_impaired": ["최근 밤사이 림프 배출 회복이 충분하지 않은 날이 반복되고 있습니다."],
        "function_persistent": ["밤사이 회복이 잘 되지 않는 상태가 지속되고 있습니다."],
        "trend_up": ["최근 일주일 동안 조금씩 증가하는 방향이 보입니다."],
        "trend_down": ["최근 일주일 동안 회복되는 방향이 보입니다."],
        "trend_stable": ["최근 일주일 동안은 전체적으로 안정적인 흐름입니다."],
        "systemic_absent": ["팔 외 다른 부위 변화는 크지 않아 현재는 수술 받으신 팔의 국소적 변화로 보입니다."],
        "systemic_present": ["팔 외 다른 부위 변화도 함께 보여 전신적인 체액 변화 가능성을 함께 살펴볼 필요가 있습니다."],
        "variability_high": ["최근 측정값 변동성이 다소 커서 상태가 일정하지 않을 수 있습니다."],
        "action_maintain": [
            "현재 관리 방법을 유지해 주세요.",
            "지금처럼 꾸준한 관리 습관을 이어가 주세요."
        ],
        "action_followup": [
            "2~3일 내 다시 측정하며 변화를 확인해 주세요.",
            "짧은 간격으로 반복 측정하여 경과를 확인해 주세요."
        ],
        "action_night_bandage": [
            "밤에 붕대를 감고 주무시는 것을 습관화하여 야간 림프 배출이 원활해지도록 관리해보시기를 권장드립니다.",
            "취침 중 압박 붕대 사용을 꾸준히 유지해 주세요."
        ],
        "action_consult": [
            "담당 의료진과 상담을 권장드립니다.",
            "현재 상태에 대해 의료진 평가를 받아보시는 것이 좋겠습니다."
        ],
        "action_systemic_check": [
            "전신적인 체액 변화 가능성도 있어 몸 전체 상태를 함께 점검해 보시는 것이 좋겠습니다."
        ],
        "action_repeat_measurement": [
            "가능하면 같은 시간대에 반복 측정해 주시면 경과를 더 정확히 볼 수 있습니다."
        ],
    }
}


def pick_random(lst, fallback=""):
    if not lst:
        return fallback
    return random.choice(lst)


def build_intro_block(row, patient_name):
    label = row.get("최종 판정", "")
    bank = MESSAGE_BANK["patient"]

    parts = []
    parts.append(pick_random(bank["intro_default"]).format(name=patient_name))

    if "주의" in label or "경계형" in label or "초기 림프 dysfunction" in label:
        parts.append(pick_random(bank["intro_attention"]).format(name=patient_name))
    elif "회복" in label:
        parts.append(pick_random(bank["intro_improving"]).format(name=patient_name))

    parts.append(pick_random(bank["analysis_intro"]))
    return "\n\n".join(parts)


# ==========================================
# 4. 메시지 코드 선택
# ==========================================
def select_message_codes(row):
    codes = []

    ratio = row.get("ratio", np.nan)
    am_3day_diff = row.get("am_3day_diff", np.nan)
    recovery_fail_3d = row.get("recovery_fail_3d", np.nan)
    fail_7d = row.get("fail_7d", np.nan)
    am_7day_trend = row.get("AM_7day_trend", np.nan)
    leg_3day_diff = row.get("leg_3day_diff", np.nan)
    trunk_3day_diff = row.get("trunk_3day_diff", np.nan)
    cv_7d = row.get("cv_7d", np.nan)
    label = row.get("최종 판정", "")

    codes.append("intro")

    if pd.notna(ratio):
        if ratio >= 1.05:
            codes.append("structure_confirmed")
        elif ratio >= 1.02:
            codes.append("structure_borderline")
        else:
            codes.append("structure_normal")

    if pd.notna(am_3day_diff):
        if am_3day_diff >= 0.004:
            codes.append("state_up")
        elif am_3day_diff <= -0.002:
            codes.append("state_down")
        else:
            codes.append("state_stable")

    if pd.notna(fail_7d) and fail_7d >= 5:
        codes.append("function_persistent")
    elif pd.notna(recovery_fail_3d) and recovery_fail_3d >= 2:
        codes.append("function_impaired")
    elif pd.notna(recovery_fail_3d) and recovery_fail_3d == 1:
        codes.append("function_mild")

    if pd.notna(am_7day_trend):
        if am_7day_trend >= 0.5:
            codes.append("trend_up")
        elif am_7day_trend <= -0.5:
            codes.append("trend_down")
        else:
            codes.append("trend_stable")

    systemic = False
    if pd.notna(leg_3day_diff) and leg_3day_diff >= 0.004:
        systemic = True
    if pd.notna(trunk_3day_diff) and trunk_3day_diff >= 0.004:
        systemic = True
    codes.append("systemic_present" if systemic else "systemic_absent")

    if pd.notna(cv_7d) and cv_7d >= 1.0:
        codes.append("variability_high")

    if "진행성" in label or "혼합형" in label or "전신 부종" in label:
        codes.append("action_consult")
    elif "주의" in label or "경계형" in label or "초기 림프 dysfunction" in label:
        codes.append("action_followup")
        codes.append("action_night_bandage")
    elif "회복 상태" in label:
        codes.append("action_maintain")
    else:
        codes.append("action_maintain")

    if systemic:
        codes.append("action_systemic_check")

    if pd.notna(fail_7d) and fail_7d >= 3:
        codes.append("action_repeat_measurement")

    return codes


def render_message(row, audience="patient", patient_name="환자"):
    codes = select_message_codes(row)
    bank = MESSAGE_BANK[audience]
    lines = []

    for code in codes:
        if code == "intro":
            if audience == "patient":
                lines.append(build_intro_block(row, patient_name))
            else:
                lines.append(pick_random(bank["intro"]))
            continue

        if code not in bank:
            continue

        text = pick_random(bank[code])
        if "{name}" in text:
            text = text.format(name=patient_name)
        lines.append(text)

    return "\n\n".join(lines)


def build_clinician_comparison_comment(row):
    comments = []

    a3 = row.get("am_3d_vs_prev3d", np.nan)
    a7 = row.get("am_7d_vs_prev7d", np.nan)

    if pd.notna(a3):
        if a3 >= 0.002:
            comments.append("최근 3일 평균은 직전 3일 대비 상승했습니다.")
        elif a3 <= -0.002:
            comments.append("최근 3일 평균은 직전 3일 대비 감소했습니다.")

    if pd.notna(a7):
        if a7 >= 0.002:
            comments.append("최근 7일 평균은 직전 7일 대비 상승했습니다.")
        elif a7 <= -0.002:
            comments.append("최근 7일 평균은 직전 7일 대비 감소했습니다.")

    return " ".join(comments)


# ==========================================
# 5. 환측 추론 / 원본 포맷 정리
# ==========================================
def infer_affected_side(sheet_name: str) -> str:
    name = str(sheet_name).strip()
    if "우측" in name:
        return "우측"
    if "좌측" in name:
        return "좌측"
    raise ValueError(
        f"시트명 '{sheet_name}' 에 환측(우측/좌측) 정보가 없습니다. "
        "시트명에 환측을 포함하세요. 예: 홍길동_우측상지"
    )


def format_raw_data(df, sheet_name):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", "", regex=False)

    col_mapping = {
        "우측상지세포외수분비": "우측 상지",
        "우측 상지 세포외수분비": "우측 상지",
        "우측상지": "우측 상지",
        "RightArm": "우측 상지",

        "좌측상지세포외수분비": "좌측 상지",
        "좌측 상지 세포외수분비": "좌측 상지",
        "좌측상지": "좌측 상지",
        "LeftArm": "좌측 상지",

        "체간세포외수분비": "체간",
        "체간 세포외수분비": "체간",
        "체간": "체간",
        "Trunk": "체간",

        "우측하지세포외수분비": "우측 하지",
        "우측 하지 세포외수분비": "우측 하지",
        "우측하지": "우측 하지",
        "RightLeg": "우측 하지",

        "좌측하지세포외수분비": "좌측 하지",
        "좌측 하지 세포외수분비": "좌측 하지",
        "좌측하지": "좌측 하지",
        "LeftLeg": "좌측 하지",

        "검사일시": "검사일시",
        "Date": "검사일시",
        "DateTime": "검사일시",
    }

    rename_dict = {}
    for col in df.columns:
        compact = col.replace(" ", "")
        for src, dst in col_mapping.items():
            if compact == src.replace(" ", ""):
                rename_dict[col] = dst
                break

    df = df.rename(columns=rename_dict)
    df["환측방향"] = infer_affected_side(sheet_name)
    return df


# ==========================================
# 6. 전처리
# ==========================================
def classify_time_period(ts: pd.Timestamp):
    if pd.isna(ts):
        return np.nan
    h = ts.hour
    return "오전" if 4 <= h < 12 else "오후"


def preprocess_data(df_raw):
    df = df_raw.copy()

    required_cols = ["검사일시", "환측방향", "우측 상지", "좌측 상지", "체간", "우측 하지", "좌측 하지"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df["검사일시"] = pd.to_datetime(df["검사일시"], errors="coerce")
    if df["검사일시"].isna().all():
        raise ValueError("검사일시를 날짜/시간으로 변환할 수 없습니다.")

    numeric_cols = ["우측 상지", "좌측 상지", "체간", "우측 하지", "좌측 하지"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["검사일시"]).sort_values("검사일시").reset_index(drop=True)

    affected_is_right = df["환측방향"].astype(str).str.contains("우")
    df["환측"] = np.where(affected_is_right, df["우측 상지"], df["좌측 상지"])
    df["건측"] = np.where(affected_is_right, df["좌측 상지"], df["우측 상지"])

    df["Time_Period"] = df["검사일시"].apply(classify_time_period)
    df["Date_Key"] = df["검사일시"].dt.date

    am_src = (
        df[df["Time_Period"] == "오전"]
        .sort_values("검사일시")
        .groupby("Date_Key", as_index=False)
        .first()
    )
    pm_src = (
        df[df["Time_Period"] == "오후"]
        .sort_values("검사일시")
        .groupby("Date_Key", as_index=False)
        .last()
    )

    am_df = am_src[["Date_Key", "환측", "건측", "우측 하지", "좌측 하지", "체간"]].copy()
    am_df.columns = ["Date_Key", "환측 오전", "건측 오전", "우측 하지", "좌측 하지", "체간"]

    pm_df = pm_src[["Date_Key", "환측", "건측"]].copy()
    pm_df.columns = ["Date_Key", "환측 오후", "건측 오후"]

    daily_df = pd.merge(am_df, pm_df, on="Date_Key", how="outer").sort_values("Date_Key").reset_index(drop=True)
    daily_df["검사일시"] = pd.to_datetime(daily_df["Date_Key"])
    daily_df["하지 평균"] = (daily_df["우측 하지"] + daily_df["좌측 하지"]) / 2

    daily_df["date_diff_days"] = daily_df["검사일시"].diff().dt.days
    daily_df["is_consecutive_day"] = daily_df["date_diff_days"].fillna(1).eq(1)

    return daily_df


# ==========================================
# 7. 지표 계산
# ==========================================
def calculate_metrics(df, baseline_days=3):
    df = df.copy()

    arm_baseline = safe_mean(df["환측 오전"].iloc[:baseline_days])
    leg_baseline = safe_mean(df["하지 평균"].iloc[:baseline_days])
    trunk_baseline = safe_mean(df["체간"].iloc[:baseline_days])

    df["baseline_ref"] = arm_baseline
    df["leg_baseline_ref"] = leg_baseline
    df["trunk_baseline_ref"] = trunk_baseline

    # 당일
    df["ratio"] = df["환측 오전"] / df["건측 오전"]
    df["AM_drift"] = df["환측 오전"] - arm_baseline
    df["day_gain"] = df["환측 오후"] - df["환측 오전"]

    df["prev_pm"] = df["환측 오후"].shift(1)
    df["night_recovery"] = np.where(
        df["is_consecutive_day"],
        df["환측 오전"] - df["prev_pm"],
        np.nan
    )

    df["recovery_fail"] = np.where(
        df["night_recovery"].isna(),
        np.nan,
        np.where(df["night_recovery"] >= 0, 1, 0)
    )

    # 3일
    df["recovery_fail_3d"] = df["recovery_fail"].rolling(3, min_periods=1).sum()

    df["am_3day_mean"] = df["환측 오전"].rolling(3, min_periods=2).mean()
    df["am_3day_diff"] = df["am_3day_mean"] - arm_baseline

    df["leg_3day_mean"] = df["하지 평균"].rolling(3, min_periods=2).mean()
    df["leg_3day_diff"] = df["leg_3day_mean"] - leg_baseline

    df["trunk_3day_mean"] = df["체간"].rolling(3, min_periods=2).mean()
    df["trunk_3day_diff"] = df["trunk_3day_mean"] - trunk_baseline

    # 7일
    df["AM_7day_mean"] = df["환측 오전"].rolling(7, min_periods=3).mean()
    df["AM_7day_trend"] = df["환측 오전"].rolling(7, min_periods=3).apply(calculate_corr, raw=False)

    df["leg_7day_mean"] = df["하지 평균"].rolling(7, min_periods=3).mean()
    df["leg_7day_diff"] = df["leg_7day_mean"] - leg_baseline
    df["leg_7day_trend"] = df["하지 평균"].rolling(7, min_periods=3).apply(calculate_corr, raw=False)

    df["trunk_7day_mean"] = df["체간"].rolling(7, min_periods=3).mean()
    df["trunk_7day_diff"] = df["trunk_7day_mean"] - trunk_baseline
    df["trunk_7day_trend"] = df["체간"].rolling(7, min_periods=3).apply(calculate_corr, raw=False)

    df["leg_drift"] = df["하지 평균"] - leg_baseline
    df["trunk_drift"] = df["체간"] - trunk_baseline

    fail_7d_list = []
    am_range_7d_list = []
    pm_range_7d_list = []
    cv_7d_list = []

    for i in range(len(df)):
        r7 = df.iloc[max(0, i - 6): i + 1]

        fail_7d = int((r7["night_recovery"] >= 0).sum())

        am_range_7d = (
            r7["환측 오전"].max() - r7["환측 오전"].min()
            if r7["환측 오전"].notna().sum() >= 2 else np.nan
        )

        pm_range_7d = (
            r7["환측 오후"].max() - r7["환측 오후"].min()
            if r7["환측 오후"].notna().sum() >= 2 else np.nan
        )

        ratio_mean = r7["ratio"].mean()
        ratio_std = r7["ratio"].std()

        if pd.notna(ratio_mean) and ratio_mean != 0 and r7["ratio"].notna().sum() >= 2:
            cv_7d = (ratio_std / ratio_mean) * 100
        else:
            cv_7d = np.nan

        fail_7d_list.append(fail_7d)
        am_range_7d_list.append(am_range_7d)
        pm_range_7d_list.append(pm_range_7d)
        cv_7d_list.append(cv_7d)

    df["fail_7d"] = fail_7d_list
    df["am_range_7d"] = am_range_7d_list
    df["pm_range_7d"] = pm_range_7d_list
    df["cv_7d"] = cv_7d_list

    warn_ratio_3d = []
    for i in range(len(df)):
        r3 = df.iloc[max(0, i - 2): i + 1]
        warn_ratio_3d.append(int((r3["ratio"] > 1.02).sum()))
    df["warn_ratio_3d"] = warn_ratio_3d

    # 이전 구간 비교 지표
    am_3d_vs_prev3d = []
    leg_3d_vs_prev3d = []
    am_7d_vs_prev7d = []
    leg_7d_vs_prev7d = []

    for i in range(len(df)):
        recent_3 = df.iloc[max(0, i - 2): i + 1]
        prev_3 = df.iloc[max(0, i - 5): max(0, i - 2)]

        if recent_3["환측 오전"].notna().sum() >= 2 and prev_3["환측 오전"].notna().sum() >= 2:
            am_3d_vs_prev3d.append(recent_3["환측 오전"].mean() - prev_3["환측 오전"].mean())
        else:
            am_3d_vs_prev3d.append(np.nan)

        if recent_3["하지 평균"].notna().sum() >= 2 and prev_3["하지 평균"].notna().sum() >= 2:
            leg_3d_vs_prev3d.append(recent_3["하지 평균"].mean() - prev_3["하지 평균"].mean())
        else:
            leg_3d_vs_prev3d.append(np.nan)

        recent_7 = df.iloc[max(0, i - 6): i + 1]
        prev_7 = df.iloc[max(0, i - 13): max(0, i - 6)]

        if recent_7["환측 오전"].notna().sum() >= 3 and prev_7["환측 오전"].notna().sum() >= 3:
            am_7d_vs_prev7d.append(recent_7["환측 오전"].mean() - prev_7["환측 오전"].mean())
        else:
            am_7d_vs_prev7d.append(np.nan)

        if recent_7["하지 평균"].notna().sum() >= 3 and prev_7["하지 평균"].notna().sum() >= 3:
            leg_7d_vs_prev7d.append(recent_7["하지 평균"].mean() - prev_7["하지 평균"].mean())
        else:
            leg_7d_vs_prev7d.append(np.nan)

    df["am_3d_vs_prev3d"] = am_3d_vs_prev3d
    df["leg_3d_vs_prev3d"] = leg_3d_vs_prev3d
    df["am_7d_vs_prev7d"] = am_7d_vs_prev7d
    df["leg_7d_vs_prev7d"] = leg_7d_vs_prev7d

    return df


# ==========================================
# 8. 판정 엔진
# ==========================================
def calculate_scores(row):
    local_score = 0
    systemic_score = 0

    if pd.notna(row["ratio"]):
        if row["ratio"] >= 1.05:
            local_score += 2
        elif row["ratio"] >= 1.02:
            local_score += 1

    if pd.notna(row["am_3day_diff"]):
        if row["am_3day_diff"] >= 0.004:
            local_score += 2
        elif row["am_3day_diff"] >= 0.002:
            local_score += 1
        elif row["am_3day_diff"] <= -0.002:
            local_score -= 1

    if pd.notna(row["recovery_fail_3d"]):
        if row["recovery_fail_3d"] >= 2:
            local_score += 2
        elif row["recovery_fail_3d"] == 1:
            local_score += 1

    if pd.notna(row["AM_7day_trend"]):
        if row["AM_7day_trend"] >= 0.5:
            local_score += 2
        elif row["AM_7day_trend"] >= 0.3:
            local_score += 1
        elif row["AM_7day_trend"] <= -0.5:
            local_score -= 1

    if pd.notna(row["cv_7d"]):
        if row["cv_7d"] >= 1.5:
            local_score += 2
        elif row["cv_7d"] >= 1.0:
            local_score += 1

    if pd.notna(row["leg_3day_diff"]):
        if row["leg_3day_diff"] >= 0.004:
            systemic_score += 2
        elif row["leg_3day_diff"] >= 0.002:
            systemic_score += 1

    if pd.notna(row["trunk_3day_diff"]):
        if row["trunk_3day_diff"] >= 0.004:
            systemic_score += 2
        elif row["trunk_3day_diff"] >= 0.002:
            systemic_score += 1

    if pd.notna(row["leg_7day_trend"]):
        if row["leg_7day_trend"] >= 0.5:
            systemic_score += 2
        elif row["leg_7day_trend"] >= 0.3:
            systemic_score += 1

    if pd.notna(row["trunk_7day_trend"]):
        if row["trunk_7day_trend"] >= 0.5:
            systemic_score += 2
        elif row["trunk_7day_trend"] >= 0.3:
            systemic_score += 1

    if pd.notna(row["fail_7d"]) and row["fail_7d"] >= 3:
        local_score += 1

    return local_score, systemic_score


def classify(row):
    local_score, systemic_score = calculate_scores(row)

    ratio = row.get("ratio", np.nan)
    am_3day_diff = row.get("am_3day_diff", np.nan)
    am_7day_trend = row.get("AM_7day_trend", np.nan)
    cv_7d = row.get("cv_7d", np.nan)
    recovery_fail_3d = row.get("recovery_fail_3d", np.nan)

    if systemic_score >= 3:
        if local_score >= 3:
            return "🟠 혼합형 악화 (림프 + 전신)"
        return "🟠 전신 부종/체액 변화 의심"

    if pd.notna(ratio) and ratio >= 1.05:
        if local_score >= 4:
            return "🔴 진행성 림프부종 (확증 + 악화)"
        return "🔵 안정형 림프부종 (확증)"

    if pd.notna(ratio) and 1.02 <= ratio < 1.05:
        if local_score >= 3:
            return "🟣 경계형 림프부종 (비대칭 + 동태 이상)"
        return "🟡 경계형 비대칭"

    if (
        pd.notna(am_3day_diff) and am_3day_diff >= 0.004 and
        pd.notna(am_7day_trend) and am_7day_trend >= 0.5
    ):
        if pd.notna(cv_7d) and cv_7d >= 1.0:
            return "🟣 초기 림프 dysfunction (변동성 동반 상승)"
        return "🟣 초기 림프 dysfunction (지속 상승)"

    if (
        pd.notna(ratio) and ratio < 1.05 and
        pd.notna(cv_7d) and cv_7d >= 1.5 and
        pd.notna(recovery_fail_3d) and recovery_fail_3d >= 2
    ):
        return "🟣 초기 림프 dysfunction 의심 (불안정 패턴)"

    if local_score >= 2:
        return "🟡 주의 관찰 필요"

    if (
        pd.notna(am_3day_diff) and am_3day_diff <= -0.002 and
        pd.notna(am_7day_trend) and am_7day_trend <= -0.5
    ):
        return "🔵 회복 상태"

    return "🟢 안정 / 정상"


# ==========================================
# 9. 그래프
# ==========================================
def create_figure(analyzed_df):
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1, ax2 = axes

    ax1.plot(
        analyzed_df["검사일시"], analyzed_df["환측 오전"],
        marker="o", markersize=7, linewidth=2.2, label="환측 오전값"
    )
    ax1.plot(
        analyzed_df["검사일시"], analyzed_df["환측 오후"],
        marker="^", linestyle="-", alpha=0.7, label="환측 오후값"
    )
    ax1.plot(
        analyzed_df["검사일시"], analyzed_df["건측 오전"],
        marker="s", linestyle="--", alpha=0.8, label="건측 오전값"
    )

    if pd.notna(analyzed_df["baseline_ref"].iloc[0]):
        ax1.axhline(
            analyzed_df["baseline_ref"].iloc[0],
            color="red", linestyle=":", linewidth=2, label="초기 기준선"
        )

    if len(analyzed_df) >= 3:
        recent_3 = analyzed_df["검사일시"].iloc[-3:]
        ax1.axvspan(
            recent_3.iloc[0], recent_3.iloc[-1],
            color="#fff2cc", alpha=0.4, label="최근 3일"
        )

    warns = analyzed_df[(analyzed_df["ratio"] >= 1.02) & analyzed_df["환측 오전"].notna()]
    if not warns.empty:
        ax1.scatter(
            warns["검사일시"],
            warns["환측 오전"] + 0.0006,
            marker="*",
            color="red",
            s=220,
            zorder=10,
            label="양측 비율 경계 시점 (Ratio ≥ 1.02)"
        )

    ax1.set_title("환측 상지의 일중 변화와 양측 비율 패턴", fontsize=14, fontweight="bold")
    ax1.set_ylabel("ECW 비율")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ax2.plot(
        analyzed_df["검사일시"],
        analyzed_df["night_recovery"],
        marker="D",
        linewidth=2.0,
        label="야간 회복량"
    )
    ax2.plot(
        analyzed_df["검사일시"],
        analyzed_df["leg_drift"],
        marker="o",
        linewidth=1.5,
        alpha=0.8,
        label="하지 기준선 이탈"
    )
    ax2.plot(
        analyzed_df["검사일시"],
        analyzed_df["trunk_drift"],
        marker="s",
        linewidth=1.5,
        alpha=0.8,
        label="체간 기준선 이탈"
    )
    ax2.axhline(0, color="black", linewidth=1)

    fail_days = analyzed_df[
        analyzed_df["night_recovery"].notna() & (analyzed_df["night_recovery"] >= 0)
    ]
    if not fail_days.empty:
        ax2.scatter(
            fail_days["검사일시"],
            fail_days["night_recovery"],
            marker="x",
            s=100,
            color="red",
            zorder=10,
            label="야간 회복 실패 시점"
        )

    ax2.set_title("야간 회복 및 전신 체액 변화", fontsize=12)
    ax2.set_ylabel("변화량")
    ax2.set_xlabel("검사일")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    return fig


# ==========================================
# 10. PDF
# ==========================================
def build_pdf(patient_name, report_date, latest_row, fig):
    clinician_msg = render_message(latest_row, audience="clinician", patient_name=patient_name)
    comparison_comment = build_clinician_comparison_comment(latest_row)
    if comparison_comment:
        clinician_msg = clinician_msg + "\n\n" + comparison_comment

    patient_msg = render_message(latest_row, audience="patient", patient_name=patient_name)

    pdf, pdf_font = init_pdf()

    pdf.set_font(pdf_font, "", 16)
    pdf.cell(0, 10, f"LEAP V3 정밀 분석 리포트 - {patient_name}", ln=1, align="C")

    pdf.set_font(pdf_font, "", 11)
    pdf.cell(0, 8, f"분석 기준일: {report_date}", ln=1)
    pdf.cell(0, 8, f"종합 판정: {latest_row['최종 판정']}", ln=1)
    pdf.ln(4)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[의료진용 해석]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(0, 6, clinician_msg)
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[환자 안내문]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(0, 6, patient_msg)
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[핵심 지표]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.cell(0, 7, f"- 양측 비율(당일): {fmt_ratio_interp(latest_row['ratio'])}", ln=1)
    pdf.cell(0, 7, f"- 환측 3일 평균 차이: {fmt_diff_interp(latest_row['am_3day_diff'])}", ln=1)
    pdf.cell(0, 7, f"- 하지 3일 평균 차이: {fmt_diff_interp(latest_row['leg_3day_diff'])}", ln=1)
    pdf.cell(0, 7, f"- 이전 3일 대비 환측 변화: {fmt_compare_interp(latest_row['am_3d_vs_prev3d'])}", ln=1)
    pdf.cell(0, 7, f"- 이전 3일 대비 하지 변화: {fmt_compare_interp(latest_row['leg_3d_vs_prev3d'])}", ln=1)
    pdf.cell(0, 7, f"- 3일내 야간 회복 실패 일수: {fmt_days_interp(latest_row['recovery_fail_3d'])}", ln=1)
    pdf.cell(0, 7, f"- 환측 7일 추세(r): {fmt_trend_interp(latest_row['AM_7day_trend'])}", ln=1)
    pdf.cell(0, 7, f"- 하지 7일 추세(r): {fmt_trend_interp(latest_row['leg_7day_trend'])}", ln=1)
    pdf.cell(0, 7, f"- 이전 7일 대비 환측 변화: {fmt_compare_interp(latest_row['am_7d_vs_prev7d'])}", ln=1)
    pdf.cell(0, 7, f"- 이전 7일 대비 하지 변화: {fmt_compare_interp(latest_row['leg_7d_vs_prev7d'])}", ln=1)
    pdf.cell(0, 7, f"- 7일내 야간 회복 실패 일수: {fmt_days_interp(latest_row['fail_7d'])}", ln=1)
    pdf.cell(0, 7, f"- 7일 변동계수: {fmt_cv_interp(latest_row['cv_7d'])}", ln=1)
    pdf.ln(4)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmp_path = tmpfile.name
            fig.savefig(tmp_path, format="png", bbox_inches="tight", dpi=180)
        pdf.image(tmp_path, x=10, y=None, w=190)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    try:
        return pdf.output(dest="S").encode("latin-1")
    except Exception:
        return bytes(pdf.output(dest="S"))


# ==========================================
# 11. 메인 앱
# ==========================================
if check_password():
    st.title("🧪 LEAP V3 테스트 시스템")
    st.markdown("V3 프로토타입 테스트 버전입니다.")
    st.caption("시트명에는 반드시 환측 방향(우측/좌측)을 포함하세요. 예: 홍길동_우측상지")

    debug_mode = st.checkbox("디버그 모드", value=True)

    uploaded_file = st.file_uploader(
        "멀티시트 엑셀 업로드 (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            _, report_date = extract_file_info(uploaded_file.name)

            mode = st.radio(
                "작업 모드 선택:",
                ["👤 개별 환자 진료 (V3)", "📦 전체 환자 일괄 출력 (V3)"],
                horizontal=True
            )
            st.markdown("---")

            if mode == "👤 개별 환자 진료 (V3)":
                selected_sheet = st.selectbox("📋 분석할 환자(시트)를 선택하세요:", sheet_names)
                patient_name = extract_patient_name_from_sheet(selected_sheet)

                raw_df = pd.read_excel(xls, sheet_name=selected_sheet)
                if debug_mode:
                    st.write("1. raw read 완료")

                formatted_df = format_raw_data(raw_df, selected_sheet)
                if debug_mode:
                    st.write("2. format 완료")

                daily_df = preprocess_data(formatted_df)
                if debug_mode:
                    st.write("3. preprocess 완료")

                if daily_df.empty:
                    st.error("분석 가능한 일별 데이터가 없습니다.")
                    st.stop()

                analyzed_df = calculate_metrics(daily_df)
                if debug_mode:
                    st.write("4. metrics 완료")

                if analyzed_df.empty:
                    st.error("지표 계산 결과가 없습니다.")
                    st.stop()

                analyzed_df["최종 판정"] = analyzed_df.apply(classify, axis=1)
                if debug_mode:
                    st.write("5. classify 완료")

                latest_row = analyzed_df.iloc[-1]

                clinician_msg = render_message(latest_row, audience="clinician", patient_name=patient_name)
                comparison_comment = build_clinician_comparison_comment(latest_row)
                if comparison_comment:
                    clinician_msg = clinician_msg + "\n\n" + comparison_comment

                patient_msg = render_message(latest_row, audience="patient", patient_name=patient_name)
                sms_msg = " ".join(patient_msg.split("\n\n")[:5])

                st.markdown("## 🧠 의료진용 해석")
                st.success(clinician_msg.replace("\n\n", " "))

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.info(
                        f"""### 🔵 당일 상태

**양측 비율**  
{fmt_ratio_interp(latest_row['ratio'])}

**오전 기준선 이탈**  
{fmt_diff_interp(latest_row['AM_drift'])}

**낮 축적량**  
{fmt_diff_interp(latest_row['day_gain'])}

**야간 회복량**  
{fmt_diff_interp(latest_row['night_recovery'])}
"""
                    )

                with col2:
                    st.warning(
                        f"""### 🟡 최근 3일 상태

**환측 3일 평균 차이**  
{fmt_diff_interp(latest_row['am_3day_diff'])}

**하지 3일 평균 차이**  
{fmt_diff_interp(latest_row['leg_3day_diff'])}

**이전 3일 대비 환측 변화**  
{fmt_compare_interp(latest_row['am_3d_vs_prev3d'])}

**이전 3일 대비 하지 변화**  
{fmt_compare_interp(latest_row['leg_3d_vs_prev3d'])}

**3일내 야간 회복 실패 일수**  
{fmt_days_interp(latest_row['recovery_fail_3d'])}

**3일내 양측 비율 비정상 일수**  
{fmt_days_interp(latest_row['warn_ratio_3d'])}
"""
                    )

                with col3:
                    st.error(
                        f"""### 🔴 최근 7일 상태

**환측 7일 추세 (r)**  
{fmt_trend_interp(latest_row['AM_7day_trend'])}

**하지 7일 추세 (r)**  
{fmt_trend_interp(latest_row['leg_7day_trend'])}

**이전 7일 대비 환측 변화**  
{fmt_compare_interp(latest_row['am_7d_vs_prev7d'])}

**이전 7일 대비 하지 변화**  
{fmt_compare_interp(latest_row['leg_7d_vs_prev7d'])}

**7일내 야간 회복 실패 일수**  
{fmt_days_interp(latest_row['fail_7d'])}

**7일 변동계수**  
{fmt_cv_interp(latest_row['cv_7d'])}
"""
                    )

                st.markdown("### 💌 환자 안내문")
                st.text_area("환자 안내문", patient_msg, height=320)

                st.markdown("### 📱 SMS 초안")
                st.text_area("SMS", sms_msg, height=180)

                st.markdown("### 🔎 선택된 메시지 코드")
                st.write(select_message_codes(latest_row))

                st.markdown("### 📈 시계열 근거 그래프")
                st.caption("상단은 환측/건측의 양측 비율 패턴을, 하단은 야간 회복과 전신 체액 변화를 보여줍니다.")
                fig = create_figure(analyzed_df)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("### 📋 상세 수치")
                view_cols = [
                    "검사일시",
                    "환측 오전", "환측 오후", "건측 오전",
                    "ratio", "AM_drift", "day_gain", "night_recovery",
                    "am_3day_diff", "leg_3day_diff", "trunk_3day_diff",
                    "am_3d_vs_prev3d", "leg_3d_vs_prev3d",
                    "recovery_fail_3d", "warn_ratio_3d",
                    "AM_7day_trend", "leg_7day_trend", "trunk_7day_trend",
                    "am_7d_vs_prev7d", "leg_7d_vs_prev7d",
                    "fail_7d", "am_range_7d", "pm_range_7d", "cv_7d",
                    "하지 평균", "leg_drift", "체간", "trunk_drift",
                    "최종 판정"
                ]
                st.dataframe(analyzed_df[view_cols], use_container_width=True)

                fig_for_pdf = create_figure(analyzed_df)
                pdf_bytes = build_pdf(patient_name, report_date, latest_row, fig_for_pdf)
                plt.close(fig_for_pdf)

                st.download_button(
                    label=f"📥 [{patient_name}] V3 리포트 다운로드 (PDF)",
                    data=pdf_bytes,
                    file_name=f"정밀분석리포트_V3_{patient_name}_{report_date}.pdf",
                    mime="application/pdf",
                    key=f"pdf_v3_{patient_name}"
                )

            else:
                st.subheader("📦 전체 환자 일괄 분석 및 리포트 자동 생성 (V3)")

                if st.button("▶️ 전체 일괄 분석 실행"):
                    progress_bar = st.progress(0)
                    summary_data = []
                    zip_buffer = io.BytesIO()

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for i, sheet in enumerate(sheet_names):
                            try:
                                patient_name_sheet = extract_patient_name_from_sheet(sheet)
                                raw_df = pd.read_excel(xls, sheet_name=sheet)

                                formatted_df = format_raw_data(raw_df, sheet)
                                daily_df = preprocess_data(formatted_df)

                                if daily_df.empty:
                                    raise ValueError("분석 가능한 일별 데이터가 없습니다.")

                                analyzed_df = calculate_metrics(daily_df)

                                if analyzed_df.empty:
                                    raise ValueError("지표 계산 결과가 없습니다.")

                                analyzed_df["최종 판정"] = analyzed_df.apply(classify, axis=1)
                                latest_row = analyzed_df.iloc[-1]

                                final_label = latest_row["최종 판정"]
                                clinician_msg = render_message(latest_row, audience="clinician", patient_name=patient_name_sheet)
                                comparison_comment = build_clinician_comparison_comment(latest_row)
                                if comparison_comment:
                                    clinician_msg = clinician_msg + " " + comparison_comment

                                summary_data.append({
                                    "환자명": patient_name_sheet,
                                    "환측방향": infer_affected_side(sheet),
                                    "최종 판정": final_label,
                                    "위험도": severity_rank(final_label),
                                    "최근 양측 비율": safe_fmt(latest_row["ratio"], ".3f"),
                                    "환측 3일 평균 차이": safe_fmt(latest_row["am_3day_diff"]),
                                    "이전 3일 대비 환측 변화": safe_fmt(latest_row["am_3d_vs_prev3d"]),
                                    "환측 7일 추세(r)": safe_fmt(latest_row["AM_7day_trend"], ".2f"),
                                    "이전 7일 대비 환측 변화": safe_fmt(latest_row["am_7d_vs_prev7d"]),
                                    "비고": clinician_msg.replace("\n\n", " ")
                                })

                                fig = create_figure(analyzed_df)
                                pdf_bytes = build_pdf(patient_name_sheet, report_date, latest_row, fig)
                                plt.close(fig)

                                pdf_filename = f"리포트_V3_{patient_name_sheet}_{report_date}.pdf"
                                zip_file.writestr(pdf_filename, pdf_bytes)

                            except Exception as e:
                                st.warning(f"⚠️ '{sheet}' 환자 처리 중 오류 발생: {e}")

                            progress_bar.progress((i + 1) / len(sheet_names))

                    st.success("✅ 모든 환자의 V3 분석 및 PDF 생성이 완료되었습니다.")
                    st.markdown("### 📊 전체 환자 현황 요약판")

                    summary_df = pd.DataFrame(summary_data).sort_values(
                        by=["위험도", "환자명"], ascending=[False, True]
                    )
                    st.dataframe(summary_df, use_container_width=True)

                    st.download_button(
                        label="📦 묶음 리포트 다운로드 (V3 전체 환자 PDF ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"전체환자리포트_V3_{report_date}.zip",
                        mime="application/zip"
                    )

        except Exception as e:
            st.error(f"오류가 발생했습니다. (에러: {e})")
