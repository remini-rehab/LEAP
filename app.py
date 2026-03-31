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

# --- 분석 로직 (7일 변동성 및 전신 지표 포함) ---
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
    
    # Baseline: 초기 3일 평균
    b_arm = daily["환측 오전"].iloc[:3].mean() if len(daily) >= 3 else daily["환측 오전"].iloc[0]
    b_leg = daily["하지 평균"].iloc[:3].mean() if len(daily) >= 3 else daily["하지 평균"].iloc[0]
    b_trunk = daily["체간"].iloc[:3].mean() if len(daily) >= 3 else daily["체간"].iloc[0]
    
    # 지표 계산
    daily["ratio"] = daily["환측 오전"] / daily["건측 오전"]
    daily["AM_drift"] = daily["환측 오전"] - b_arm
    daily["day_gain"] = daily["환측 오후"] - daily["환측 오전"]
    daily["night_recovery"] = daily["환측 오전"].shift(-1) - daily["환측 오후"]
    daily["leg_drift"] = daily["하지 평균"] - b_leg
    daily["trunk_drift"] = daily["체간"] - b_trunk
    
    return daily, b_arm, b_leg, b_trunk

# --- PDF 리포트 생성 함수 (임상 지표 반영) ---
def build_pdf(patient_name, report_date, latest, stats, fig):
    pdf = FPDF()
    pdf.add_page()
    
    if os.path.exists(FONT_PATH):
        pdf.add_font("Nanum", "", FONT_PATH, uni=True)
        pdf.set_font("Nanum", "", 16)
    else:
        pdf.set_font("Arial", "B", 16)

    pdf.cell(0, 10, f"LEAP 정밀 분석 리포트: {patient_name}", ln=True, align='C')
    pdf.set_font("Arial" if not os.path.exists(FONT_PATH) else "Nanum", "", 10)
    pdf.cell(0, 8, f"최근 분석일(당일 기준): {report_date}", ln=True)
    pdf.ln(5)

    # 지표 요약 섹션
    pdf.set_font("Arial" if not os.path.exists(FONT_PATH) else "Nanum", "", 12)
    pdf.cell(0, 8, "[1/3/7일 통합 분석 지표]", ln=True)
    pdf.set_font("Arial" if not os.path.exists(FONT_PATH) else "Nanum", "", 10)
    
    pdf.cell(0, 7, f"- 당일 비율: {latest['ratio']:.3f} / AM 이탈(Drift): {latest['AM_drift']:.4f}", ln=True)
    pdf.cell(0, 7, f"- 최근 3일 회복 실패: {stats['f3']}회 / 위험 경고(1.02 초과): {stats['w3']}회", ln=True)
    pdf.cell(0, 7, f"- 7일 비율 CV(불안정성): {stats['cv7']:.2f}% / 오전 변동폭(기초체력): {stats['am_r7']:.4f}", ln=True)
    pdf.cell(0, 7, f"- 하지 기준선 이탈: {latest['leg_drift']:.4f} / 체간 기준선 이탈: {latest['trunk_drift']:.4f}", ln=True)
    pdf.ln(5)

    # 그래프 이미지 삽입
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=120)
        pdf.image(tmp.name, x=10, y=85, w=190)
    
    return bytes(pdf.output())

# --- 메인 UI ---
st.title("🏥 LEAP 정밀 분석 시스템")
uploaded_file = st.file_uploader("멀티시트 엑셀 업로드 (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    selected = st.selectbox("환자 선택", xls.sheet_names)
    
    # 데이터 분석 실행
    df, b_arm, b_leg, b_trunk = analyze_data(pd.read_excel(xls, sheet_name=selected), selected)
    
    latest = df.iloc[-1]
    r3 = df.tail(3); r7 = df.tail(7)
    
    # 7일 관점 지표 산출
    stats = {
        'f3': int((r3["night_recovery"] >= 0).sum()),
        'w3': int((r3["ratio"] > 1.02).sum()),
        'f7': int((r7["night_recovery"] >= 0).sum()),
        'w7': int((r7["ratio"] > 1.02).sum()),
        'cv7': (r7["ratio"].std() / r7["ratio"].mean()) * 100 if len(r7) > 1 else 0,
        'am_r7': r7["환측 오전"].max() - r7["환측 오전"].min(),
        'pm_r7': r7["환측 오후"].max() - r7["환측 오후"].min()
    }

    # --- 대시보드 레이아웃 (이미지 스타일 복구) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div style="background-color: #E8F1FF; padding: 18px; border-radius: 15px; height: 145px; border: 1px solid #D0E2FF;"><h4 style="color: #0056B3; margin: 0; font-size: 1.2rem;">🔵 당일 <span style="font-size: 0.85rem; color: #555;">({latest["Date_Key"]})</span></h4><div style="margin-top: 15px;"><p style="color: #0056B3; font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">비율: {latest["ratio"]:.3f}</p><p style="color: #0056B3; font-size: 1.1rem; font-weight: bold; margin: 0;">이탈: {latest["AM_drift"]:.4f}</p></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="background-color: #FFF9E1; padding: 18px; border-radius: 15px; height: 145px; border: 1px solid #FBE8A6;"><h4 style="color: #856404; margin: 0; font-size: 1.2rem;">🟡 3일</h4><div style="margin-top: 25px;"><p style="color: #856404; font-size: 1.1rem; font-weight: bold; margin: 0;">실패: {stats["f3"]}회  경고: {stats["w3"]}회</p></div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div style="background-color: #FFE8E8; padding: 18px; border-radius: 15px; height: 145px; border: 1px solid #FFD1D1;"><h4 style="color: #A94442; margin: 0; font-size: 1.2rem;">🔴 7일</h4><div style="margin-top: 12px; line-height: 1.4;"><p style="color: #A94442; font-size: 0.95rem; font-weight: bold; margin: 0;">CV: {stats["cv7"]:.2f}%  오전변동: {stats["am_r7"]:.4f}</p><p style="color: #A94442; font-size: 1.0rem; font-weight: bold; margin-top: 4px;">하지이탈: {latest["leg_drift"]:.4f}</p></div></div>', unsafe_allow_html=True)

    # --- 그래프 섹션 ---
    st.write("---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1.8, 1]})
    
    # 상지 동태 분석
    ax1.plot(df["검사일시"], df["환측 오전"], 'o-', color='#FF9999', label="환측 오전", linewidth=2)
    ax1.plot(df["검사일시"], df["환측 오후"], '^-', color='#FF0000', label="환측 오후", alpha=0.7)
    ax1.plot(df["검사일시"], df["건측 오전"], 's--', color='#ADD8E6', alpha=0.5, label="건측 오전")
    ax1.axhline(b_arm, color='gray', linestyle=':', alpha=0.5, label="초기 Baseline")
    
    # 1.02 경고 별표(★) 마킹
    warns = df[df["ratio"] > 1.02]
    if not warns.empty:
        ax1.scatter(warns["검사일시"], warns["환측 오전"] + 0.0006, marker='*', color='red', s=250, zorder=10, label="1.02 초과")
    
    ax1.set_title("상지 동태 분석 (★: 위험 경계치 1.02 초과)", weight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.2)
    
    # 전신 기준선 이탈 추이
    ax2.plot(df["검사일시"], df["leg_drift"], 'o-', color='purple', label="하지 이탈")
    ax2.plot(df["검사일시"], df["trunk_drift"], 's-', color='green', label="체간 이탈")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("전신 기준선 이탈 추이 (하지/체간)")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.2)
    
    st.pyplot(fig)

    # --- 리포트 다운로드 버튼 ---
    st.write("---")
    p_name = str(selected).split('_')[0]
    pdf_bytes = build_pdf(p_name, str(latest['Date_Key']), latest, stats, fig)
    st.download_button(
        label=f"📥 [{p_name}] 정밀 분석 리포트 다운로드 (PDF)", 
        data=pdf_bytes, 
        file_name=f"LEAP_Report_{p_name}_{latest['Date_Key']}.pdf",
        mime="application/pdf"
    )
