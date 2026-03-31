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

# 1. 폰트 및 기본 설정 (나눔바른고딕 적용)
FONT_PATH = "NanumBarunGothic.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rc("font", family="NanumBarunGothic")
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# --- 유틸리티 및 분석 로직 ---
def safe_fmt(value, fmt=".4f"):
    if pd.isna(value): return "-"
    return format(float(value), fmt)

def analyze_data(df_raw, sheet_name):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    # 시트명에서 환측(우측/좌측) 추출
    side = "우측" if "우측" in str(sheet_name) else "좌측"
    df["검사일시"] = pd.to_datetime(df["검사일시"], errors="coerce")
    df = df.dropna(subset=["검사일시"]).sort_values("검사일시")
    
    # 환측/건측 배정
    df["환측"] = df["우측 상지"] if side == "우측" else df["좌측 상지"]
    df["건측"] = df["좌측 상지"] if side == "우측" else df["우측 상지"]
    
    # 오전/오후 통합 로직 (오전 4-11시, 오후 12-23시)
    df["Time"] = df["검사일시"].dt.hour.apply(lambda h: "오전" if 4 <= h < 12 else "오후")
    df["Date"] = df["검사일시"].dt.date
    
    am = df[df["Time"] == "오전"].groupby("Date").first()
    pm = df[df["Time"] == "오후"].groupby("Date").last()
    
    daily = pd.merge(am[["환측", "건측", "우측 하지", "좌측 하지", "체간"]], 
                     pm[["환측", "건측"]], on="Date", how="outer", suffixes=(" 오전", " 오후"))
    daily = daily.sort_index().reset_index()
    daily["검사일시"] = pd.to_datetime(daily["Date"])
    daily["하지 평균"] = (daily["우측 하지"] + daily["좌측 하지"]) / 2
    
    # 지표 계산
    b_arm = daily["환측 오전"].iloc[:3].mean()
    b_leg = daily["하지 평균"].iloc[:3].mean()
    b_trunk = daily["체간"].iloc[:3].mean()
    
    daily["ratio"] = daily["환측 오전"] / daily["건측 오전"]
    daily["AM_drift"] = daily["환측 오전"] - b_arm
    daily["day_gain"] = daily["환측 오후"] - daily["환측 오전"]
    daily["night_recovery"] = daily["환측 오전"].shift(-1) - daily["환측 오후"]
    daily["leg_drift"] = daily["하지 평균"] - b_leg
    daily["trunk_drift"] = daily["체간"] - b_trunk
    
    return daily, b_arm, b_leg, b_trunk

# --- PDF 리포트 생성 함수 ---
def build_pdf(patient_name, report_date, latest, stats, fig):
    pdf = FPDF()
    pdf.add_page()
    
    pdf_font = "Arial"
    if os.path.exists(FONT_PATH):
        pdf.add_font("Nanum", "", FONT_PATH, uni=True)
        pdf_font = "Nanum"
    
    pdf.set_font(pdf_font, "", 16)
    pdf.cell(0, 10, f"LEAP 림프 정밀 분석 리포트 - {patient_name}", ln=1, align="C")
    
    pdf.set_font(pdf_font, "", 11)
    pdf.cell(0, 8, f"분석일(당일 기준): {report_date}", ln=1)
    pdf.ln(5)

    pdf.set_font(pdf_font, "", 12)
    pdf.cell(0, 8, "[1/3/7일 통합 분석 지표]", ln=1)
    pdf.set_font(pdf_font, "", 10)
    pdf.cell(0, 7, f"- 당일 비율: {latest['ratio']:.3f} / 기준선 이탈: {latest['AM_drift']:.4f}", ln=1)
    pdf.cell(0, 7, f"- 최근 3일 회복 실패: {stats['f3']}회 / 경고(>1.02): {stats['w3']}회", ln=1)
    pdf.cell(0, 7, f"- 7일 비율 CV(불안정성): {stats['cv7']:.2f}% / 오전 변동폭(기초 체력): {stats['am_r7']:.4f}", ln=1)
    pdf.cell(0, 7, f"- 하지 기준선 이탈: {latest['leg_drift']:.4f} / 체간 기준선 이탈: {latest['trunk_drift']:.4f}", ln=1)
    pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmpfile.name, x=10, y=None, w=190)
    
    try:
        return pdf.output(dest="S").encode("latin-1")
    except:
        return pdf.output(dest="S")

# --- 메인 실행부 ---
st.title("🏥 LEAP 정밀 분석 시스템")
uploaded_file = st.file_uploader("멀티시트 엑셀 업로드 (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    selected = st.selectbox("환자 선택 (시트명)", sheet_names)
    
    raw_df = pd.read_excel(xls, sheet_name=selected)
    df, b_arm, b_leg, b_trunk = analyze_data(raw_df, selected)
    
    latest = df.iloc[-1]
    r3 = df.tail(3); r7 = df.tail(7)
    
    stats = {
        'f3': int((r3["night_recovery"] >= 0).sum()),
        'w3': int((r3["ratio"] > 1.02).sum()),
        'f7': int((r7["night_recovery"] >= 0).sum()),
        'w7': int((r7["ratio"] > 1.02).sum()),
        'cv7': (r7["ratio"].std() / r7["ratio"].mean()) * 100 if r7["ratio"].mean() != 0 else 0,
        'am_r7': r7["환측 오전"].max() - r7["환측 오전"].min()
    }

    # --- 대시보드 박스 디자인 (이미지 스타일 복구) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div style="background-color: #E8F1FF; padding: 20px; border-radius: 15px; height: 135px;"><h4 style="color: #0056B3; margin-top: 0;">🔵 당일 <span style="font-size: 14px; color: #666;">({latest["Date"]})</span></h4><p style="color: #0056B3; font-size: 18px; font-weight: bold;">비율: {latest["ratio"]:.3f} 이탈: {latest["AM_drift"]:.4f}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="background-color: #FFF9E1; padding: 20px; border-radius: 15px; height: 135px;"><h4 style="color: #856404; margin-top: 0;">🟡 3일</h4><p style="color: #856404; font-size: 18px; font-weight: bold;">실패: {stats["f3"]}회 경고: {stats["w3"]}회</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div style="background-color: #FFE8E8; padding: 20px; border-radius: 15px; height: 135px;"><h4 style="color: #A94442; margin-top: 0;">🔴 7일</h4><p style="color: #A94442; font-size: 15px; font-weight: bold;">CV: {stats["cv7"]:.2f}% 오전변동: {stats["am_r7"]:.4f}<br>하지이탈: {latest["leg_drift"]:.4f}</p></div>', unsafe_allow_html=True)

    # --- 그래프 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1.8, 1]})
    ax1.plot(df["검사일시"], df["환측 오전"], 'o-', color='#FF9999', label="환측 오전", linewidth=2, markersize=6)
    ax1.plot(df["검사일시"], df["환측 오후"], '^-', color='#FF0000', label="환측 오후", linewidth=1.5)
    ax1.plot(df["검사일시"], df["건측 오전"], 's--', color='#ADD8E6', alpha=0.5, label="건측 오전")
    ax1.axhline(b_arm, color='red', linestyle=':', alpha=0.4, label="Baseline")
    
    warns = df[df["ratio"] > 1.02]
    if not warns.empty:
        ax1.scatter(warns["검사일시"], warns["환측 오전"] + 0.0006, marker='*', color='red', s=250, zorder=10, label="1.02 경고")
        for d in warns["검사일시"]:
            ax1.axvline(x=d, color='red', linestyle=':', linewidth=1, alpha=0.2)
    
    ax1.set_title("상지 동태 분석 (★: 위험 경계치 초과)", fontsize=13, weight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.2)
    
    ax2.plot(df["검사일시"], df["leg_drift"], 'o-', color='purple', label="하지 이탈", markersize=5)
    ax2.plot(df["검사일시"], df["trunk_drift"], 's-', color='green', label="체간 이탈", markersize=5)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("전신 기준선 이탈 추이", fontsize=11)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.2)
    
    st.pyplot(fig)

    # --- 리포트 출력 버튼 ---
    st.markdown("---")
    patient_name = str(selected).split('_')[0]
    pdf_bytes = build_pdf(patient_name, str(latest['Date']), latest, stats, fig)
    st.download_button(
        label=f"📥 [{patient_name}] 정밀 리포트 다운로드 (PDF)",
        data=pdf_bytes,
        file_name=f"LEAP_리포트_{patient_name}_{latest['Date']}.pdf",
        mime="application/pdf"
    )
