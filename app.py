import os
import re
import tempfile
import zipfile
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF

# ==========================================
# 0. 기본 설정
# ==========================================
st.set_page_config(page_title="LEAP 림프부종 정밀 분석 시스템", layout="wide")

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
    if "혼합형 악화" in label:
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
        st.title("?? 시스템 접근")
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
# 1-1. 해석 표시 유틸
# ==========================================
def calculate_slope(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s))
    y = s.values
    return np.polyfit(x, y, 1)[0]


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
    if x > 0.001:
        return f"{x:+.4f} (↑ 증가)"
    elif x < -0.0005:
        return f"{x:+.4f} (↓ 감소)"
    return f"{x:+.4f} (→ 안정)"


def render_interpretation_guide():
    st.markdown("""
### ?? 해석 기준

**양측 비율**
- 1.05 이상: 확증
- 1.02 이상 ~ 1.05 미만: 경계
- 1.02 미만: 정상

**오전 기준선 이탈 / 3일 평균 차이**
- +0.004 이상: 증가
- -0.002 이하: 감소
- 그 사이: 안정

**7일 추세**
- +0.001 초과: 상승
- -0.0005 미만: 하강
- 그 사이: 안정

**회복 실패 일수**
- 0일: 정상
- 1일: 경미
- 2일: 기능 저하
- 3일 이상: 지속 이상

**7일 변동계수**
- 1.5% 이상: 불안정
- 1.0% 이상: 변동 증가
- 1.0% 미만: 안정
""")


def make_dashboard_summary(latest_row):
    label = latest_row.get("최종 판정", "")

    ratio = latest_row.get("ratio", np.nan)
    am_3day_diff = latest_row.get("am_3day_diff", np.nan)
    leg_3day_diff = latest_row.get("leg_3day_diff", np.nan)
    am_trend = latest_row.get("AM_7day_trend", np.nan)
    leg_trend = latest_row.get("leg_7day_trend", np.nan)
    fail3 = latest_row.get("recovery_fail_3d", np.nan)
    cv7 = latest_row.get("cv_7d", np.nan)

    lines = []

    if "진행성 림프부종" in label:
        lines.append("현재는 진행성 림프부종에 해당합니다.")
    elif "안정형 림프부종" in label:
        lines.append("현재는 구조적 비대칭은 있으나 비교적 안정된 상태입니다.")
    elif "경계형 림프부종" in label:
        lines.append("현재는 경계형 림프부종으로 판단됩니다.")
    elif "경계형 비대칭" in label:
        lines.append("현재는 경계 수준의 비대칭이 관찰됩니다.")
    elif "초기 림프 dysfunction" in label:
        lines.append("현재는 초기 림프 기능 저하가 의심됩니다.")
    elif "혼합형 악화" in label:
        lines.append("현재는 국소 림프 변화와 전신 변화가 함께 관찰됩니다.")
    elif "전신 부종" in label:
        lines.append("현재는 전신 부종 또는 체액 변화 가능성이 높습니다.")
    elif "회복 상태" in label:
        lines.append("현재는 회복 방향의 변화가 관찰됩니다.")
    else:
        lines.append("현재는 뚜렷한 이상 소견이 두드러지지 않습니다.")

    if pd.notna(ratio):
        if ratio >= 1.05:
            lines.append(f"양측 비율은 {ratio:.3f}로 확증 범위입니다.")
        elif ratio >= 1.02:
            lines.append(f"양측 비율은 {ratio:.3f}로 경계 범위입니다.")
        else:
            lines.append(f"양측 비율은 {ratio:.3f}로 정상 범위입니다.")

    if pd.notna(am_3day_diff):
        if am_3day_diff >= 0.004:
            lines.append(f"환측 3일 평균 차이는 {am_3day_diff:+.4f}로 증가 상태입니다.")
        elif am_3day_diff <= -0.002:
            lines.append(f"환측 3일 평균 차이는 {am_3day_diff:+.4f}로 감소 상태입니다.")
        else:
            lines.append(f"환측 3일 평균 차이는 {am_3day_diff:+.4f}로 비교적 안정적입니다.")

    if pd.notna(fail3):
        if fail3 >= 2:
            lines.append(f"최근 3일간 회복 실패 일수는 {int(fail3)}일로 기능 저하가 의심됩니다.")
        elif fail3 == 1:
            lines.append("최근 3일간 회복 실패는 경미한 수준입니다.")

    if pd.notna(am_trend):
        if am_trend > 0.001:
            lines.append(f"환측 7일 추세는 {am_trend:+.4f}로 상승 방향입니다.")
        elif am_trend < -0.0005:
            lines.append(f"환측 7일 추세는 {am_trend:+.4f}로 회복 방향입니다.")
        else:
            lines.append("환측 7일 추세는 뚜렷한 방향성 없이 안정적입니다.")

    if pd.notna(leg_3day_diff):
        if leg_3day_diff >= 0.004:
            lines.append(f"하지 3일 평균 차이는 {leg_3day_diff:+.4f}로 증가 상태입니다.")
        else:
            lines.append("하지 3일 평균 차이는 크지 않아 국소 변화 가능성이 높습니다.")

    if pd.notna(leg_trend) and leg_trend > 0.001:
        lines.append(f"하지 7일 추세는 {leg_trend:+.4f}로 상승 방향입니다.")

    if pd.notna(cv7) and cv7 >= 1.0:
        lines.append(f"7일 변동계수는 {cv7:.2f}%로 변동 증가 상태입니다.")

    return " ".join(lines)


# ==========================================
# 2. 환측 추론 / 원본 포맷 정리
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
# 3. 전처리
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
# 4. 지표 계산
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

    df["AM_3day_range"] = (
        df["환측 오전"].rolling(3, min_periods=2).max()
        - df["환측 오전"].rolling(3, min_periods=2).min()
    ).fillna(0)

    # 7일 수준
    df["AM_7day_mean"] = df["환측 오전"].rolling(7, min_periods=3).mean()
    df["AM_7day_drift"] = df["AM_7day_mean"] - arm_baseline

    df["leg_7day_mean"] = df["하지 평균"].rolling(7, min_periods=3).mean()
    df["leg_7day_diff"] = df["leg_7day_mean"] - leg_baseline

    df["trunk_7day_mean"] = df["체간"].rolling(7, min_periods=3).mean()
    df["trunk_7day_diff"] = df["trunk_7day_mean"] - trunk_baseline

    # 7일 추세
    df["AM_7day_trend"] = df["환측 오전"].rolling(7, min_periods=3).apply(calculate_slope, raw=False)
    df["leg_7day_trend"] = df["하지 평균"].rolling(7, min_periods=3).apply(calculate_slope, raw=False)
    df["trunk_7day_trend"] = df["체간"].rolling(7, min_periods=3).apply(calculate_slope, raw=False)

    # 전신 변화량
    df["leg_change"] = np.where(
        df["is_consecutive_day"],
        df["하지 평균"] - df["하지 평균"].shift(1),
        np.nan
    )
    df["trunk_change"] = np.where(
        df["is_consecutive_day"],
        df["체간"] - df["체간"].shift(1),
        np.nan
    )
    df["leg_drift"] = df["하지 평균"] - leg_baseline
    df["trunk_drift"] = df["체간"] - trunk_baseline

    # 7일 요약 지표
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

    # 3일 ratio 경고 일수
    warn_ratio_3d = []
    for i in range(len(df)):
        r3 = df.iloc[max(0, i - 2): i + 1]
        warn_ratio_3d.append(int((r3["ratio"] > 1.02).sum()))
    df["warn_ratio_3d"] = warn_ratio_3d

    return df


# ==========================================
# 5. 점수 계산 / 분류
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
        if row["AM_7day_trend"] > 0.001:
            local_score += 2
        elif row["AM_7day_trend"] > 0.0005:
            local_score += 1
        elif row["AM_7day_trend"] < -0.0005:
            local_score -= 1

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
        if row["leg_7day_trend"] > 0.001:
            systemic_score += 2
        elif row["leg_7day_trend"] > 0.0005:
            systemic_score += 1

    if pd.notna(row["trunk_7day_trend"]):
        if row["trunk_7day_trend"] > 0.001:
            systemic_score += 2
        elif row["trunk_7day_trend"] > 0.0005:
            systemic_score += 1

    if pd.notna(row["fail_7d"]) and row["fail_7d"] >= 3:
        local_score += 1

    return local_score, systemic_score


def classify(row):
    local_score, systemic_score = calculate_scores(row)

    ratio = row.get("ratio", np.nan)
    am_3day_diff = row.get("am_3day_diff", np.nan)
    am_7day_trend = row.get("AM_7day_trend", np.nan)

    if systemic_score >= 3:
        if local_score >= 3:
            return "?? 혼합형 악화 (림프 + 전신)"
        return "?? 전신 부종/체액 변화 의심"

    if pd.notna(ratio) and ratio >= 1.05:
        if local_score >= 4:
            return "?? 진행성 림프부종 (확증 + 악화)"
        return "?? 안정형 림프부종 (확증)"

    if pd.notna(ratio) and 1.02 <= ratio < 1.05:
        if local_score >= 3:
            return "?? 경계형 림프부종 (비대칭 + 동태 이상)"
        return "?? 경계형 비대칭"

    if (
        pd.notna(am_3day_diff) and am_3day_diff >= 0.004 and
        pd.notna(am_7day_trend) and am_7day_trend > 0.0005
    ):
        return "?? 초기 림프 dysfunction (동태 이상)"

    if local_score >= 2:
        return "?? 주의 관찰 필요"

    if (
        pd.notna(am_3day_diff) and am_3day_diff <= -0.002 and
        pd.notna(am_7day_trend) and am_7day_trend < -0.0005
    ):
        return "?? 회복 상태"

    return "?? 안정 / 정상"


def explain_latest(row):
    reasons = []

    if pd.notna(row["ratio"]):
        if row["ratio"] >= 1.05:
            reasons.append(f"양측 비율 {row['ratio']:.3f}로 확증 범위입니다")
        elif row["ratio"] >= 1.02:
            reasons.append(f"양측 비율 {row['ratio']:.3f}로 경계 범위입니다")

    if pd.notna(row["am_3day_diff"]):
        if row["am_3day_diff"] >= 0.004:
            reasons.append(f"환측 3일 평균 차이가 {row['am_3day_diff']:.4f}로 증가 상태입니다")
        elif row["am_3day_diff"] <= -0.002:
            reasons.append(f"환측 3일 평균 차이가 {row['am_3day_diff']:.4f}로 감소 상태입니다")

    if pd.notna(row["recovery_fail_3d"]) and row["recovery_fail_3d"] >= 2:
        reasons.append(f"최근 3일간 회복 실패 일수가 {int(row['recovery_fail_3d'])}일로 기능 저하가 의심됩니다")

    if pd.notna(row["AM_7day_trend"]):
        if row["AM_7day_trend"] > 0.001:
            reasons.append(f"환측 7일 추세가 {row['AM_7day_trend']:+.4f}로 상승 방향입니다")
        elif row["AM_7day_trend"] < -0.0005:
            reasons.append(f"환측 7일 추세가 {row['AM_7day_trend']:+.4f}로 회복 방향입니다")

    if pd.notna(row["leg_3day_diff"]) and row["leg_3day_diff"] >= 0.004:
        reasons.append(f"하지 3일 평균 차이가 {row['leg_3day_diff']:.4f}로 증가해 전신 영향 가능성이 있습니다")

    if pd.notna(row["leg_7day_trend"]) and row["leg_7day_trend"] > 0.001:
        reasons.append(f"하지 7일 추세가 {row['leg_7day_trend']:+.4f}로 상승 방향입니다")

    if pd.notna(row["cv_7d"]) and row["cv_7d"] >= 1.0:
        reasons.append(f"7일 변동계수는 {row['cv_7d']:.2f}%로 변동 증가 상태입니다")

    return " / ".join(reasons) if reasons else "뚜렷한 이상 소견이 두드러지지 않습니다."


# ==========================================
# 6. 그래프
# ==========================================
def create_figure(analyzed_df):
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1, ax2 = axes

    ax1.plot(analyzed_df["검사일시"], analyzed_df["환측 오전"], marker="o", markersize=7, linewidth=2.2, label="환측 오전값")
    ax1.plot(analyzed_df["검사일시"], analyzed_df["환측 오후"], marker="^", linestyle="-", alpha=0.7, label="환측 오후값")
    ax1.plot(analyzed_df["검사일시"], analyzed_df["건측 오전"], marker="s", linestyle="--", alpha=0.8, label="건측 오전값")

    if pd.notna(analyzed_df["baseline_ref"].iloc[0]):
        ax1.axhline(analyzed_df["baseline_ref"].iloc[0], color="red", linestyle=":", linewidth=2, label="초기 기준선")

    if len(analyzed_df) >= 3:
        recent_3 = analyzed_df["검사일시"].iloc[-3:]
        ax1.axvspan(recent_3.iloc[0], recent_3.iloc[-1], color="#fff2cc", alpha=0.4, label="최근 3일")

    warns = analyzed_df[(analyzed_df["ratio"] >= 1.02) & analyzed_df["환측 오전"].notna()]
    if not warns.empty:
        ax1.scatter(
            warns["검사일시"],
            warns["환측 오전"] + 0.0006,
            marker="*",
            color="red",
            s=220,
            zorder=10,
            label="비율 경계 시점 (Ratio ≥ 1.02)"
        )
        for d in warns["검사일시"]:
            ax1.axvline(x=d, color="red", linestyle=":", alpha=0.25)

    for i in range(1, len(analyzed_df)):
        prev_pm = analyzed_df["환측 오후"].iloc[i - 1]
        curr_am = analyzed_df["환측 오전"].iloc[i]
        prev_x = analyzed_df["검사일시"].iloc[i - 1]
        curr_x = analyzed_df["검사일시"].iloc[i]

        if pd.notna(prev_pm) and pd.notna(curr_am):
            arrow_color = "blue" if curr_am < prev_pm else "red"
            ax1.annotate(
                "",
                xy=(curr_x, curr_am),
                xytext=(prev_x, prev_pm),
                arrowprops=dict(arrowstyle="->", color=arrow_color, linestyle="--", linewidth=1.2, alpha=0.8)
            )

    ax1.set_title("환측 상지의 일중 변화와 야간 회복 패턴", fontsize=14, fontweight="bold")
    ax1.set_ylabel("ECW 비율")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ax2.plot(analyzed_df["검사일시"], analyzed_df["leg_drift"], marker="o", linewidth=1.8, label="하지 기준선 이탈")
    ax2.plot(analyzed_df["검사일시"], analyzed_df["trunk_drift"], marker="s", linewidth=1.8, label="체간 기준선 이탈")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("전신 체액 변화 추이 (하지/체간)", fontsize=12)
    ax2.set_ylabel("기준선 대비 변화량")
    ax2.set_xlabel("검사일")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    return fig


# ==========================================
# 7. PDF
# ==========================================
def build_pdf(patient_name, report_date, latest_row, fig):
    pdf, pdf_font = init_pdf()

    pdf.set_font(pdf_font, "", 16)
    pdf.cell(0, 10, f"LEAP 정밀 분석 리포트 - {patient_name}", ln=1, align="C")

    pdf.set_font(pdf_font, "", 11)
    pdf.cell(0, 8, f"분석 기준일: {report_date}", ln=1)
    pdf.cell(0, 8, f"종합 판정: {latest_row['최종 판정']}", ln=1)
    pdf.ln(4)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[종합 해석]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(0, 6, make_dashboard_summary(latest_row))
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[해석 기준]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(
        0, 6,
        "양측 비율: 1.05 이상 확증, 1.02 이상 경계\n"
        "오전 기준선 이탈/3일 평균 차이: +0.004 이상 증가, -0.002 이하 감소\n"
        "7일 추세: +0.001 초과 상승, -0.0005 미만 하강\n"
        "회복 실패 일수: 0일 정상, 2일 기능 저하, 3일 이상 지속 이상\n"
        "7일 변동계수: 1.5% 이상 불안정"
    )
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[당일 상태]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(
        0, 6,
        f"양측 비율은 {fmt_ratio_interp(latest_row['ratio'])}, "
        f"오전 기준선 이탈은 {fmt_diff_interp(latest_row['AM_drift'])}, "
        f"낮 축적량은 {fmt_diff_interp(latest_row['day_gain'])}, "
        f"야간 회복량은 {fmt_diff_interp(latest_row['night_recovery'])}입니다."
    )
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[최근 3일 상태]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(
        0, 6,
        f"환측 3일 평균 차이는 {fmt_diff_interp(latest_row['am_3day_diff'])}, "
        f"하지 3일 평균 차이는 {fmt_diff_interp(latest_row['leg_3day_diff'])}, "
        f"3일간 회복 실패 일수는 {fmt_days_interp(latest_row['recovery_fail_3d'])}입니다."
    )
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[최근 7일 추세]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(
        0, 6,
        f"환측 7일 추세는 {fmt_trend_interp(latest_row['AM_7day_trend'])}, "
        f"하지 7일 추세는 {fmt_trend_interp(latest_row['leg_7day_trend'])}, "
        f"7일간 회복 실패 일수는 {fmt_days_interp(latest_row['fail_7d'])}, "
        f"7일 변동계수는 {fmt_cv_interp(latest_row['cv_7d'])}입니다."
    )
    pdf.ln(3)

    pdf.set_font(pdf_font, "", 11)
    pdf.multi_cell(
        0, 6,
        "상지 그래프는 환측 오전/오후 변화와 건측 비교를 통해 일중 축적 및 야간 회복 패턴을 보여줍니다. "
        "하지 및 체간 그래프는 전신 체액 변화 여부를 해석하기 위한 보조 지표입니다."
    )
    pdf.ln(3)

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
# 8. 메인 앱
# ==========================================
if check_password():
    st.title("?? LEAP 정밀 감별 진단 시스템")
    st.markdown("양측 비율, 상지 시계열, 하지/체간 전신성 지표를 통합 평가합니다.")
    st.caption("시트명에는 반드시 환측 방향(우측/좌측)을 포함하세요. 예: 홍길동_우측상지")

    uploaded_file = st.file_uploader(
        "멀티시트 엑셀 업로드 (기계 출력 원본 그대로 업로드하세요!)",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            _, report_date = extract_file_info(uploaded_file.name)

            mode = st.radio(
                "작업 모드 선택:",
                ["?? 개별 환자 진료 (1:1 상담용)", "?? 전체 환자 일괄 출력 (PDF 압축)"],
                horizontal=True
            )
            st.markdown("---")

            if mode == "?? 개별 환자 진료 (1:1 상담용)":
                selected_sheet = st.selectbox("?? 분석할 환자(시트)를 선택하세요:", sheet_names)
                patient_name = selected_sheet.split("_")[0]

                raw_df = pd.read_excel(xls, sheet_name=selected_sheet)
                formatted_df = format_raw_data(raw_df, selected_sheet)
                daily_df = preprocess_data(formatted_df)
                analyzed_df = calculate_metrics(daily_df)
                analyzed_df["최종 판정"] = analyzed_df.apply(classify, axis=1)
                latest_row = analyzed_df.iloc[-1]

                st.markdown("## ?? 종합 해석")
                st.success(make_dashboard_summary(latest_row))

                top_left, top_right = st.columns([2.3, 1])

                with top_left:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.info(
                            f"""### ?? 당일 상태

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
                            f"""### ?? 최근 3일 상태

**환측 3일 평균 차이**  
{fmt_diff_interp(latest_row['am_3day_diff'])}

**하지 3일 평균 차이**  
{fmt_diff_interp(latest_row['leg_3day_diff'])}

**3일간 회복 실패 일수**  
{fmt_days_interp(latest_row['recovery_fail_3d'])}

**3일 ratio 경고 일수**  
{fmt_days_interp(latest_row['warn_ratio_3d'])}
"""
                        )

                    with col3:
                        st.error(
                            f"""### ?? 최근 7일 추세

**환측 7일 추세**  
{fmt_trend_interp(latest_row['AM_7day_trend'])}

**하지 7일 추세**  
{fmt_trend_interp(latest_row['leg_7day_trend'])}

**7일간 회복 실패 일수**  
{fmt_days_interp(latest_row['fail_7d'])}

**7일 변동계수**  
{fmt_cv_interp(latest_row['cv_7d'])}
"""
                        )

                with top_right:
                    render_interpretation_guide()

                st.markdown("### ?? 시계열 근거 그래프")
                st.caption("환측 오전값은 baseline 상태를, 오후값은 일중 축적 상태를 반영합니다. 하지?체간 변화는 전신 영향 여부를 해석하는 데 사용됩니다.")
                fig = create_figure(analyzed_df)
                st.pyplot(fig)

                st.markdown("### ?? 상세 수치")
                view_cols = [
                    "검사일시",
                    "환측 오전", "환측 오후", "건측 오전",
                    "ratio", "AM_drift", "day_gain", "night_recovery",
                    "am_3day_diff", "leg_3day_diff", "trunk_3day_diff",
                    "recovery_fail_3d", "warn_ratio_3d",
                    "AM_7day_trend", "leg_7day_trend", "trunk_7day_trend",
                    "fail_7d", "am_range_7d", "pm_range_7d", "cv_7d",
                    "하지 평균", "leg_drift", "체간", "trunk_drift",
                    "최종 판정"
                ]
                st.dataframe(analyzed_df[view_cols], use_container_width=True)

                pdf_bytes = build_pdf(patient_name, report_date, latest_row, fig)
                st.download_button(
                    label=f"?? [{patient_name}] 리포트 다운로드 (PDF)",
                    data=pdf_bytes,
                    file_name=f"정밀분석리포트_{patient_name}_{report_date}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{patient_name}"
                )

            else:
                st.subheader("?? 전체 환자 일괄 분석 및 리포트 자동 생성")

                if st.button("▶? 전체 일괄 분석 실행"):
                    progress_bar = st.progress(0)
                    summary_data = []
                    zip_buffer = io.BytesIO()

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for i, sheet in enumerate(sheet_names):
                            try:
                                patient_name_sheet = sheet.split("_")[0]
                                raw_df = pd.read_excel(xls, sheet_name=sheet)

                                formatted_df = format_raw_data(raw_df, sheet)
                                daily_df = preprocess_data(formatted_df)
                                analyzed_df = calculate_metrics(daily_df)
                                analyzed_df["최종 판정"] = analyzed_df.apply(classify, axis=1)
                                latest_row = analyzed_df.iloc[-1]

                                final_label = latest_row["최종 판정"]
                                summary_data.append({
                                    "환자명": patient_name_sheet,
                                    "환측방향": infer_affected_side(sheet),
                                    "최종 판정": final_label,
                                    "위험도": severity_rank(final_label),
                                    "최근 Ratio": safe_fmt(latest_row["ratio"], ".3f"),
                                    "환측 3일 평균 차이": safe_fmt(latest_row["am_3day_diff"]),
                                    "하지 3일 평균 차이": safe_fmt(latest_row["leg_3day_diff"]),
                                    "환측 7일 추세": safe_fmt(latest_row["AM_7day_trend"]),
                                    "하지 7일 추세": safe_fmt(latest_row["leg_7day_trend"]),
                                    "비고": make_dashboard_summary(latest_row)
                                })

                                fig = create_figure(analyzed_df)
                                pdf_bytes = build_pdf(patient_name_sheet, report_date, latest_row, fig)
                                plt.close(fig)

                                pdf_filename = f"리포트_{patient_name_sheet}_{report_date}.pdf"
                                zip_file.writestr(pdf_filename, pdf_bytes)

                            except Exception as e:
                                st.warning(f"?? '{sheet}' 환자 처리 중 오류 발생: {e}")

                            progress_bar.progress((i + 1) / len(sheet_names))

                    st.success("? 모든 환자의 분석 및 PDF 리포트 생성이 완료되었습니다!")
                    st.markdown("### ?? 전체 환자 현황 요약판")

                    summary_df = pd.DataFrame(summary_data).sort_values(
                        by=["위험도", "환자명"], ascending=[False, True]
                    )
                    st.dataframe(summary_df, use_container_width=True)

                    st.markdown("---")
                    st.download_button(
                        label="?? 묶음 리포트 다운로드 (전체 환자 PDF 압축파일)",
                        data=zip_buffer.getvalue(),
                        file_name=f"전체환자리포트_일괄출력_{report_date}.zip",
                        mime="application/zip"
                    )

        except Exception as e:
            st.error(f"오류가 발생했습니다. (에러: {e})")
