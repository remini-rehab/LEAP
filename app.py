import os
import re
import tempfile
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF

# 1. 폰트 설정
FONT_PATH = "NanumBarunGothic.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rc("font", family="NanumBarunGothic")
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False

# --- 분석 로직 ---
def analyze_data(df_raw, sheet_name):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.replace(" ", "").str.replace("\n", "")
    target_cols = {
        "우측상지": ["우측상지세포외수분비", "우측상지", "RightArm"],
        "좌측상지": ["좌측상지세포외수분비", "좌측상지", "LeftArm"],
        "체간": ["체간세포외수분비", "체간", "Trunk"],
        "우측하지": ["우측하지세포외수분비", "우측하지", "RightLeg"],
        "좌측하지": ["좌측하지세포외수분비", "좌측하지", "LeftLeg"],
        "검사일시": ["검사일시", "Date", "DateTime"]
    }
    for std_name, candidates in target_cols.items():
        for col in df.columns:
            if any(cand in col for cand in candidates):
                df = df.rename(columns={col: std_name})
                break
    side = "우측" if "우측" in str(sheet_name) else "좌측"
    df["검사일시"] = pd.to_datetime(df["검사일시"], errors="coerce")
    df = df.dropna(subset=["검사일시", "우측상지", "좌측상지"]).sort_values("검사일시")
    if len(df) == 0:
        st.error("분석 가능한 데이터가 부족합니다.")
        st.stop()
    df["환측"] = df["우측상지"] if side == "우측" else df["좌측상지"]
    df["건측"] = df["좌측상지"] if side == "우측" else df["우측상지"]
    df["Time"] = df["검사일시"].dt.hour.apply(lambda h: "오전" if 4 <= h < 12 else "오후")
    df["Date_Key"] = df["검사일시"].dt.date
    am = df[df["Time"] == "오전"].groupby("Date_Key").first()
    pm = df[df["Time"] == "오후"].groupby("Date_Key").last()
    daily = pd.merge(am[["환측", "건측", "우측하지", "좌측하지", "체간"]], 
                     pm[["환측", "건측"]], on="Date_Key", how="outer", suffixes=(" 오전", " 오후"))
    daily = daily.sort_index().reset_index().ffill().bfill()
    daily["검사일시"] = pd.to_datetime(daily["Date_Key"])
    daily["하지 평균"] = (daily["우측하지"] + daily["좌측하지"]) / 2
    b_arm = daily["환측 오전"].iloc[:3].mean() if len(daily) >= 3 else daily["환측 오전"].iloc[0]
    b_leg = daily["하지 평균"].iloc[:3].mean() if len(daily) >= 3 else daily["하지 평균"].iloc[0]
    b_trunk = daily["체간"].iloc[:3].mean() if len(daily) >= 3 else daily["체간"].iloc[0]
    daily["ratio"] = daily["환측 오전"] / daily["건측 오전"]
    daily["AM_drift"] = daily["환측 오전"] - b_arm
    daily["night_recovery"] = daily["환측 오전"].shift(-1) - daily["환측 오후"]
    daily["leg_drift"] = daily["하지 평균"] - b_leg
    daily["trunk_drift"] = daily["체간"] - b_trunk
    return daily, b_arm, b_leg, b_trunk

def build_pdf(patient_name, report_date, latest, stats, fig):
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(FONT_PATH):
        pdf.add_font("Nanum", "", FONT_PATH, uni=True)
        pdf.set_font("Nanum", "", 14)
    else: pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"LEAP Analysis Report: {patient_name}", ln=True, align='C')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=120)
        pdf.image(tmp.name, x=10, y=80, w=190)
    return bytes(pdf.output())

# --- 메인 실행부 ---
st.title("🏥 LEAP 정밀 분석 시스템")
uploaded_file = st.file_uploader("엑셀 업로드", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    selected = st.selectbox("환자 선택", xls.sheet_names)
    df, b_arm, b_leg, b_trunk = analyze_data(pd.read_excel(xls, sheet_name=selected), selected)
    
    latest = df.iloc[-1]
    r3, r7 = df.tail(3), df.tail(7)
    stats = {
        'f3': int((r3["night_recovery"] >= 0).sum()),
        'w3': int((r3["ratio"] > 1.02).sum()),
        'cv7': (r7["ratio"].std() / r7["ratio"].mean()) * 100 if len(r7) > 1 else 0,
        'am_r7': r7["환측 오전"].max() - r7["환측 오전"].min()
    }

    # --- 대시보드 섹션 (안전한 문자열 처리) ---
    c1, c2, c3 = st.columns(3)
    
    # 🔵 당일 카드
    date_val = str(latest["Date_Key"])
    ratio_txt = f"{latest['ratio']:.3f}"
    drift_txt = f"{latest['AM_drift']:.4f}"
    c1.markdown(f'<div style="background-color:#E8F1FF;padding:18px;border-radius:15px;height:145px;border:1px solid #D0E2FF;"><h4 style="color:#0056B3;margin:0;font-size:1.2rem;">🔵 당일 <span style="font-size:0.85rem;color:#555;">({date_val})</span></h4><div style="margin-top:15px;"><p style="color:#0056B3;font-size:1.1rem;font-weight:bold;margin-bottom:5px;">비율: {ratio_txt}</p><p style="color:#0056B3;font-size:1.1rem;font-weight:bold;margin:0;">이탈: {drift_txt}</p></div></div>', unsafe_allow_html=True)

    # 🟡 3일 카드
    f3_txt = str(stats["f3"])
    w3_txt = str(stats["w3"])
    c2.markdown(f'<div style="background-color:#FFF9E1;padding:18px;border-radius:15px;height:145px;border:1px solid #FBE8A6;"><h4 style="color:#856404;margin:0;font-size:1.2rem;">🟡 3일</h4><div style="margin-top:25px;"><p style="color:#856404;font-size:1.1rem;font-weight:bold;margin:0;">실패: {f3_txt}회  경고: {w3_txt}회</p></div></div>', unsafe_allow_html=True)

    # 🔴 7일 카드
    cv_txt = f"{stats['cv7']:.2f}"
    am_r_txt = f"{stats['am_r7']:.4f}"
    leg_d_txt = f"{latest['leg_drift']:.4f}"
    c3.markdown(f'<div style="background-color:#FFE8E8;padding:18px;border-radius:15px;height:145px;border:1px solid #FFD1D1;"><h4 style="color:#A94442;margin:0;font-size:1.2rem;">🔴 7일</h4><div style="margin-top:12px;line-height:1.4;"><p style="color:#A94442;font-size:0.95rem;font-weight:bold;margin:0;">CV: {cv_txt}%  오전변동: {am_r_txt}</p><p style="color:#A94442;font-size:1.0rem;font-weight:bold;margin-top:4px;">하지이탈: {leg_d_txt}</p></div></div>', unsafe_allow_html=True)

    # --- 그래프 섹션 (모든 화살표, 별표 기능 포함) ---
    st.write("---")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1, ax2 = axes

    ax1.plot(df["검사일시"], df["환측 오전"], 'o-', color='#FF9999', label="환측 오전", linewidth=2.2)
    ax1.plot(df["검사일시"], df["건측 오전"], 's--', color='#ADD8E6', alpha=0.6, label="건측 오전")
    ax1.plot(df["검사일시"], df["환측 오후"], '^', color='#FF0000', label="환측 오후", alpha=0.8)
    
    # 별표(★) 1.02 초과 마킹
    warns = df[df["ratio"] > 1.02]
    if not warns.empty:
        ax1.scatter(warns["검사일시"], warns["환측 오전"] + 0.0006, marker='*', color='red', s=250, zorder=10)

    # 최근 3일 강조
    if len(df) >= 3:
        ax1.axvspan(df["검사일시"].iloc[-3], df["검사일시"].iloc[-1], color="#FFF2CC", alpha=0.4, label="최근 3일")
    ax1.axhline(b_arm, color="red", linestyle=":", linewidth=1.5, alpha=0.5, label="초기 Baseline")

    # 야간 회복 화살표
    for i in range(len(df)-1):
        p_pm, c_am = df["환측 오후"].iloc[i], df["환측 오전"].iloc[i+1]
        p_x, c_x = df["검사일시"].iloc[i], df["검사일시"].iloc[i+1]
        if pd.notna(p_pm) and pd.notna(c_am):
            color = "blue" if c_am < p_pm else "red"
            ax1.annotate("", xy=(c_x, c_am), xytext=(p_x, p_pm), arrowprops=dict(arrowstyle="->", color=color, linestyle="--", alpha=0.6))

    ax1.set_title("상지 동태 분석 (★: 1.02 초과)", fontsize=15, fontweight="bold")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.plot(df["검사일시"], df["leg_drift"], 'o-', color='purple', label="하지 이탈", linewidth=1.8)
    ax2.plot(df["검사일시"], df["trunk_drift"], 's-', color='green', label="체간 이탈", linewidth=1.8)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("전신 기준선 이탈 추이", fontsize=12)
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax2.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)

    # 리포트 다운로드
    st.write("---")
    p_name = str(selected).split('_')[0]
    pdf_bytes = build_pdf(p_name, date_val, latest, stats, fig)
    st.download_button(label=f"📥 [{p_name}] 리포트 다운로드", data=pdf_bytes, file_name=f"Report_{p_name}.pdf")
