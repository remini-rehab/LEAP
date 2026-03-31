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

st.set_page_config(page_title="LEAP 림프부종 정밀 분석 시스템", layout="wide")

FONT_PATH = "NanumBarunGothic.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rc("font", family="NanumBarunGothic")
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


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
        st.title("🔒 시스템 접근")
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


def infer_affected_side(sheet_name: str) -> str:
    name = str(sheet_name).strip()
    if "우측" in name:
        return "우측"
    if "좌측" in name:
        return "좌측"
    raise ValueError(
        f"시트명 '{sheet_name}' 에 환측(우측/좌측) 정보가 없습니다. "
        "시트명에 환측을 포함하거나 원본 데이터에 환측 컬럼을 추가하세요."
    )


def format_raw_data(df, sheet_name):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    col_mapping = {
        "우측상지 세포외수분비": "우측 상지",
        "우측 상지 세포외수분비": "우측 상지",
        "좌측상지 세포외수분비": "좌측 상지",
        "좌측 상지 세포외수분비": "좌측 상지",
        "체간 세포외수분비": "체간",
        "우측하지 세포외수분비": "우측 하지",
        "우측 하지 세포외수분비": "우측 하지",
        "좌측하지 세포외수분비": "좌측 하지",
        "좌측 하지 세포외수분비": "좌측 하지",
    }
    df = df.rename(columns=col_mapping)

    df["환측"] = infer_affected_side(sheet_name)
    return df


def classify_time_period(ts: pd.Timestamp):
    if pd.isna(ts):
        return np.nan
    h = ts.hour
    # 권장 시간창: 오전 4~11시, 오후 12~23시
    # 실제 프로토콜 있으면 여기 수정
    return "오전" if 4 <= h < 12 else "오후"


def preprocess_data(df_raw):
    df = df_raw.copy()
    required_cols = ["검사일시", "환측", "우측 상지", "좌측 상지", "체간", "우측 하지", "좌측 하지"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df["검사일시"] = pd.to_datetime(df["검사일시"], errors="coerce")
    if df["검사일시"].isna().all():
        raise ValueError("검사일시를 변환할 수 없습니다.")

    numeric_cols = ["우측 상지", "좌측 상지", "체간", "우측 하지", "좌측 하지"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["검사일시"]).sort_values("검사일시").reset_index(drop=True)

    affected_is_right = df["환측"].astype(str).str.contains("우")
    df["환측_수치"] = np.where(affected_is_right, df["우측 상지"], df["좌측 상지"])
    df["건측_수치"] = np.where(affected_is_right, df["좌측 상지"], df["우측 상지"])
    df["환측"] = df["환측_수치"]
    df["건측"] = df["건측_수치"]

    df["Time_Period"] = df["검사일시"].apply(classify_time_period)
    df["Date_Only"] = df["검사일시"].dt.date

    am_src = (
        df[df["Time_Period"] == "오전"]
        .sort_values("검사일시")
        .groupby("Date_Only", as_index=False)
        .first()
    )
    pm_src = (
        df[df["Time_Period"] == "오후"]
        .sort_values("검사일시")
        .groupby("Date_Only", as_index=False)
        .last()
    )

    am_df = am_src[["Date_Only", "환측", "건측", "우측 하지", "좌측 하지", "체간"]].copy()
    am_df.columns = ["Date_Only", "환측 오전", "건측 오전", "우측 하지", "좌측 하지", "체간"]

    pm_df = pm_src[["Date_Only", "환측", "건측"]].copy()
    pm_df.columns = ["Date_Only", "환측 오후", "건측 오후"]

    daily_df = (
        pd.merge(am_df, pm_df, on="Date_Only", how="outer")
        .sort_values("Date_Only")
        .reset_index(drop=True)
    )

    daily_df["검사일시"] = pd.to_datetime(daily_df["Date_Only"])
    daily_df["하지 평균"] = (daily_df["우측 하지"] + daily_df["좌측 하지"]) / 2
    daily_df["date_diff_days"] = daily_df["검사일시"].diff().dt.days
    daily_df["is_consecutive_day"] = daily_df["date_diff_days"].fillna(1).eq(1)
    return daily_df


def calculate_metrics(df, baseline_days=3):
    df = df.copy()
    arm_baseline = safe_mean(df["환측 오전"].iloc[:baseline_days])
    leg_baseline = safe_mean(df["하지 평균"].iloc[:baseline_days])
    trunk_baseline = safe_mean(df["체간"].iloc[:baseline_days])

    df["baseline_ref"] = arm_baseline
    df["leg_baseline_ref"] = leg_baseline
    df["trunk_baseline_ref"] = trunk_baseline

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
    df["recovery_fail_3d"] = df["recovery_fail"].rolling(3, min_periods=1).sum()
    df["AM_drift"] = df["환측 오전"] - arm_baseline
    df["AM_3day_range"] = (
        df["환측 오전"].rolling(3, min_periods=2).max()
        - df["환측 오전"].rolling(3, min_periods=2).min()
    ).fillna(0)
    df["AM_7day_mean"] = df["환측 오전"].rolling(7, min_periods=3).mean()
    df["AM_7day_drift"] = df["AM_7day_mean"] - arm_baseline
    df["ratio"] = df["환측 오전"] / df["건측 오전"]
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
    return df


def calculate_scores(row):
    UL_state, UL_dynamic, systemic_score = 0, 0, 0

    if pd.notna(row["ratio"]):
        if row["ratio"] >= 1.05:
            UL_state = 2
        elif row["ratio"] >= 1.03:
            UL_state = 1

    if pd.notna(row["AM_drift"]):
        if row["AM_drift"] >= 0.004:
            UL_dynamic += 2
        elif row["AM_drift"] >= 0.002:
            UL_dynamic += 1

    if pd.notna(row["day_gain"]):
        if row["day_gain"] >= 0.008:
            UL_dynamic += 2
        elif row["day_gain"] >= 0.005:
            UL_dynamic += 1

    if pd.notna(row["night_recovery"]):
        if row["night_recovery"] >= 0:
            UL_dynamic += 2
        elif row["night_recovery"] > -0.002:
            UL_dynamic += 1

    if pd.notna(row["recovery_fail_3d"]) and row["recovery_fail_3d"] >= 2:
        UL_dynamic += 1

    if pd.notna(row["AM_3day_range"]):
        if row["AM_3day_range"] >= 0.006:
            UL_dynamic += 2
        elif row["AM_3day_range"] >= 0.003:
            UL_dynamic += 1

    if pd.notna(row["AM_7day_drift"]) and row["AM_7day_drift"] >= 0.004:
        UL_dynamic += 1

    if pd.notna(row["leg_change"]) and row["leg_change"] > 0.003:
        systemic_score += 1
    if pd.notna(row["leg_drift"]) and row["leg_drift"] > 0.005:
        systemic_score += 1
    if pd.notna(row["trunk_change"]) and row["trunk_change"] > 0.004:
        systemic_score += 1
    if pd.notna(row["trunk_drift"]) and row["trunk_drift"] > 0.006:
        systemic_score += 1

    return UL_state, UL_dynamic, systemic_score


def classify(row):
    UL_state, UL_dynamic, systemic = calculate_scores(row)

    if systemic >= 1:
        return "🟠 혼합형 악화 (림프 + 전신)" if (UL_state >= 1 or UL_dynamic >= 3) else "🟠 전신 부종/체액 변화 의심"
    if UL_state >= 2:
        return "🔴 진행성 림프부종 (확증 + 악화)" if UL_dynamic >= 2 else "🔵 안정형 림프부종 (확증)"
    if UL_state == 1:
        return "🟣 경계형 림프부종 (비대칭 + 동태 이상)" if UL_dynamic >= 2 else "🟡 경계형 비대칭"
    if UL_dynamic >= 4:
        return "🟣 초기 림프 dysfunction (동태 이상)"
    if UL_dynamic >= 2:
        return "🟡 주의 관찰 필요"
    return "🟢 안정 / 정상"


def explain_latest(row):
    reasons = []
    if pd.notna(row["ratio"]):
        if row["ratio"] >= 1.05:
            reasons.append(f"양측 비율 {row['ratio']:.3f}로 확증 범위")
        elif row["ratio"] >= 1.03:
            reasons.append(f"양측 비율 {row['ratio']:.3f}로 경계 범위")
    if pd.notna(row["AM_drift"]) and row["AM_drift"] >= 0.002:
        reasons.append(f"아침 기준선 상승 {row['AM_drift']:.4f}")
    if pd.notna(row["day_gain"]) and row["day_gain"] >= 0.005:
        reasons.append(f"낮 축적 증가 {row['day_gain']:.4f}")
    if pd.notna(row["night_recovery"]):
        if row["night_recovery"] >= 0:
            reasons.append(f"야간 회복 실패 {row['night_recovery']:.4f}")
        elif row["night_recovery"] > -0.002:
            reasons.append(f"야간 회복 저하 {row['night_recovery']:.4f}")
    if pd.notna(row["recovery_fail_3d"]) and row["recovery_fail_3d"] >= 2:
        reasons.append(f"최근 3일 회복 실패 {int(row['recovery_fail_3d'])}회")
    if pd.notna(row["leg_drift"]) and row["leg_drift"] > 0.005:
        reasons.append(f"하지 평균 상승 {row['leg_drift']:.4f}")
    if pd.notna(row["trunk_drift"]) and row["trunk_drift"] > 0.006:
        reasons.append(f"체간 상승 {row['trunk_drift']:.4f}")
    return " / ".join(reasons) if reasons else "특이 이상 소견이 두드러지지 않습니다."


def create_figure(analyzed_df):
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1, ax2 = axes

    ax1.plot(analyzed_df["검사일시"], analyzed_df["환측 오전"], marker="o", markersize=7, linewidth=2.2, label="환측 오전")
    ax1.plot(analyzed_df["검사일시"], analyzed_df["건측 오전"], marker="s", linestyle="--", alpha=0.8, label="건측 오전")
    ax1.plot(analyzed_df["검사일시"], analyzed_df["환측 오후"], marker="^", linestyle="", alpha=0.7, label="환측 오후")

    if pd.notna(analyzed_df["baseline_ref"].iloc[0]):
        ax1.axhline(analyzed_df["baseline_ref"].iloc[0], color="red", linestyle=":", linewidth=2, label="초기 baseline")
    if len(analyzed_df) >= 3:
        recent_3 = analyzed_df["검사일시"].iloc[-3:]
        ax1.axvspan(recent_3.iloc[0], recent_3.iloc[-1], color="#fff2cc", alpha=0.5, label="최근 3일")

    for i in range(1, len(analyzed_df)):
        prev_pm = analyzed_df["환측 오후"].iloc[i - 1]
        curr_am = analyzed_df["환측 오전"].iloc[i]
        prev_x = analyzed_df["검사일시"].iloc[i - 1]
        curr_x = analyzed_df["검사일시"].iloc[i]
        if pd.notna(prev_pm) and pd.notna(curr_am):
            arrow_color = "blue" if curr_am < prev_pm else "red"
            ax1.annotate(
                "", xy=(curr_x, curr_am), xytext=(prev_x, prev_pm),
                arrowprops=dict(arrowstyle="->", color=arrow_color, linestyle="--", linewidth=1.3, alpha=0.8)
            )

    ax1.set_title("상지 동태 분석", fontsize=15, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ax2.plot(analyzed_df["검사일시"], analyzed_df["하지 평균"], marker="o", linewidth=1.8, label="하지 평균")
    ax2.plot(analyzed_df["검사일시"], analyzed_df["체간"], marker="s", linewidth=1.8, label="체간")

    if pd.notna(analyzed_df["leg_baseline_ref"].iloc[0]):
        ax2.axhline(analyzed_df["leg_baseline_ref"].iloc[0], linestyle=":", linewidth=1.5, label="하지 baseline")
    if pd.notna(analyzed_df["trunk_baseline_ref"].iloc[0]):
        ax2.axhline(analyzed_df["trunk_baseline_ref"].iloc[0], linestyle="--", linewidth=1.2, label="체간 baseline")

    ax2.set_title("전신 보정 지표 (하지/체간)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    return fig


def build_pdf(patient_name, report_date, latest_row, fig):
    pdf, pdf_font = init_pdf()
    pdf.set_font(pdf_font, "", 16)
    pdf.cell(0, 10, f"LEAP 정밀 분석 리포트 - {patient_name}", ln=1, align="C")

    pdf.set_font(pdf_font, "", 11)
    pdf.cell(0, 8, f"분석 기준일: {report_date}", ln=1)
    pdf.cell(0, 8, f"종합 판정: {latest_row['최종 판정']}", ln=1)
    pdf.ln(4)

    pdf.set_font(pdf_font, "", 13)
    pdf.cell(0, 8, "[핵심 지표 요약]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.cell(0, 7, f"- 양측 비율 (Ratio): {safe_fmt(latest_row['ratio'], '.3f')}", ln=1)
    pdf.cell(0, 7, f"- 아침 기준선 이탈 (AM drift): {safe_fmt(latest_row['AM_drift'])}", ln=1)
    pdf.cell(0, 7, f"- 낮 축적량 (day gain): {safe_fmt(latest_row['day_gain'])}", ln=1)
    pdf.cell(0, 7, f"- 야간 회복량: {safe_fmt(latest_row['night_recovery'])}", ln=1)
    pdf.cell(0, 7, f"- 최근 3일 회복 실패 횟수: {safe_fmt(latest_row['recovery_fail_3d'], '.0f')}", ln=1)
    pdf.ln(4)

    pdf.set_font(pdf_font, "", 12)
    pdf.cell(0, 8, "[요약 해석]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.multi_cell(0, 6, explain_latest(latest_row))
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


if check_password():
    st.title("🏥 LEAP 정밀 감별 진단 시스템")
    st.markdown("양측 비율, 상지 시계열, 하지/체간 전신성 지표를 통합 평가합니다.")

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
                ["👤 개별 환자 진료 (1:1 상담용)", "📦 전체 환자 일괄 출력 (PDF 압축)"],
                horizontal=True
            )
            st.markdown("---")

            if mode == "👤 개별 환자 진료 (1:1 상담용)":
                selected_sheet = st.selectbox("📋 분석할 환자(시트)를 선택하세요:", sheet_names) if len(sheet_names) > 1 else sheet_names[0]
                patient_name = selected_sheet.split('_')[0]

                raw_df = pd.read_excel(xls, sheet_name=selected_sheet)
                formatted_df = format_raw_data(raw_df, selected_sheet)
                daily_df = preprocess_data(formatted_df)
                analyzed_df = calculate_metrics(daily_df)
                analyzed_df["최종 판정"] = analyzed_df.apply(classify, axis=1)
                latest_row = analyzed_df.iloc[-1]

                st.success(f"[{patient_name}] 환자 데이터 정밀 분석 완료 (환측: {infer_affected_side(selected_sheet)})")
                st.markdown(f"### 종합 판정: **{latest_row['최종 판정']}**")
                st.caption(explain_latest(latest_row))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(
                        f"**🔵 당일/단면 관점**\n\n"
                        f"- 양측 비율: {safe_fmt(latest_row['ratio'], '.3f')}\n"
                        f"- AM drift: {safe_fmt(latest_row['AM_drift'])}"
                    )
                with col2:
                    st.warning(
                        f"**🟡 3일 동태 관점**\n\n"
                        f"- 낮 축적량: {safe_fmt(latest_row['day_gain'])}\n"
                        f"- 야간 회복량: {safe_fmt(latest_row['night_recovery'])}\n"
                        f"- 3일 회복 실패: {safe_fmt(latest_row['recovery_fail_3d'], '.0f')}회"
                    )
                with col3:
                    st.error(
                        f"**🔴 7일/전신 관점**\n\n"
                        f"- 7일 AM drift: {safe_fmt(latest_row['AM_7day_drift'])}\n"
                        f"- 하지 drift: {safe_fmt(latest_row['leg_drift'])}\n"
                        f"- 체간 drift: {safe_fmt(latest_row['trunk_drift'])}"
                    )

                fig = create_figure(analyzed_df)
                st.pyplot(fig)

                st.markdown("### 📋 분석 테이블")
                view_cols = [
                    "검사일시", "환측 오전", "환측 오후", "건측 오전",
                    "ratio", "AM_drift", "day_gain", "night_recovery",
                    "recovery_fail_3d", "하지 평균", "체간", "최종 판정"
                ]
                st.dataframe(analyzed_df[view_cols], use_container_width=True)

                pdf_bytes = build_pdf(patient_name, report_date, latest_row, fig)
                st.download_button(
                    label=f"📥 [{patient_name}] 리포트 다운로드 (PDF)",
                    data=pdf_bytes,
                    file_name=f"정밀분석리포트_{patient_name}_{report_date}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{patient_name}"
                )

            else:
                st.subheader("📦 전체 환자 일괄 분석 및 리포트 자동 생성")
                if st.button("▶️ 전체 일괄 분석 실행"):
                    progress_bar = st.progress(0)
                    summary_data = []
                    zip_buffer = io.BytesIO()

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for i, sheet in enumerate(sheet_names):
                            try:
                                patient_name_sheet = sheet.split('_')[0]
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
                                    "비고": explain_latest(latest_row)
                                })

                                fig = create_figure(analyzed_df)
                                pdf_bytes = build_pdf(patient_name_sheet, report_date, latest_row, fig)
                                plt.close(fig)

                                pdf_filename = f"리포트_{patient_name_sheet}_{report_date}.pdf"
                                zip_file.writestr(pdf_filename, pdf_bytes)

                            except Exception as e:
                                st.warning(f"⚠️ '{sheet}' 환자 처리 중 오류 발생: {e}")

                            progress_bar.progress((i + 1) / len(sheet_names))

                    st.success("✅ 모든 환자의 분석 및 PDF 리포트 생성이 완료되었습니다!")
                    st.markdown("### 📊 전체 환자 현황 요약판")

                  
	        summary_df = pd.DataFrame(summary_data).sort_values(
                        by=["위험도", "환자명"], ascending=[False, True]
                    )
                    
                    # 👇 윗줄과 완벽하게 줄이 맞아야 합니다! 👇
                    st.dataframe(summary_df.style.map(
                        lambda x: 'color: red; font-weight: bold' if isinstance(x, str) and '진행성' in x else '',
                        subset=['최종 판정']
                    ), use_container_width=True)

                    st.markdown("---")
                    st.download_button(
                        label="📦 묶음 리포트 다운로드 (전체 환자 PDF 압축파일)",
                        data=zip_buffer.getvalue(),
                        file_name=f"전체환자리포트_일괄출력_{report_date}.zip",
                        mime="application/zip"
                    )

        except Exception as e:
            st.error(f"오류가 발생했습니다. (에러: {e})")