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

# 1. 폰트 및 기본 설정
FONT_PATH = "NanumBarunGothic.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rc("font", family="NanumBarunGothic")
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# --- 유틸리티 함수 ---
def safe_fmt(value, fmt=".4f"):
    if pd.isna(value): return "-"
    return format(float(value), fmt)

# --- 분석 로직 ---
def analyze_data(df_raw, sheet_name):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.replace(" ", "")

    col_map = {
        "우측상지세포외수분비": "우측상지",
        "좌측상지세포외수분비": "좌측상지",
        "체간세포외수분비": "체간",
        "우측하지세포외수분비": "우측하지",
        "좌측하지세포외수분비": "좌측하지",
        "검사일시": "검사일시"
    }
    
    for raw_col in df.columns:
        for key, standard in col_map.items():
            if key in raw_col:
                df = df.rename(columns={raw_col: standard})

    side = "우측" if "우측" in str(sheet_name) else "좌측"
    df["검사일시"] = pd.to_datetime(df["검사일시"], errors="coerce")
    df = df.dropna(subset=["검사일시"]).sort_values("검사일시")
    
    df["환측"] = df["우측상지"] if side == "우측" else df["좌측상지"]
    df["건측"] = df["좌측상지"] if side == "우측" else df["우측상지"]
    
    df["Time"] = df["검사일시"].dt.hour.apply(lambda h: "오전" if 4 <= h < 12 else "오후")
    df["Date"] = df["검사일시"].dt.date
    
    am = df[df["Time"] == "오전"].groupby("Date").first()
    pm = df[df["Time"] == "오후"].groupby("Date").last()
    
    daily = pd.merge(am[["환측", "건측", "우측하지", "좌측하지", "체간"]], 
                     pm[["환측", "건측"]], on="Date", how="outer", suffixes=(" 오전", " 오후"))
    daily = daily.sort_index().reset_index()
    daily["검사일시"] = pd.to_datetime(daily["Date"])
    daily["하지 평균"] = (daily["우측하지"] + daily["좌측하지"]) / 2
    
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
    return pdf.output(dest="S").encode("latin-1")

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
        'cv7': (r7["ratio"].std() / r7["ratio"].mean()) * 100 if r7["ratio"].mean() != 0 else 0,
        'am_r7': r7["환측 오전"].max() - r7["환측 오전"].min()
    }

    # --- 대시보드 박스 디자인 (Syntax 수정 완료) ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div style="background-color: #E8F1FF; padding: 20px; border-radius: 15px; height: 135px;">
            <h4 style="color: #0056B3; margin: 0;">🔵 당일 <span style="font-size: 14px; color: #666;">({latest['Date']})</span></h4>
            <p style="color: #0056B3; font-size: 18px; font-weight: bold; margin-top: 10px;">
                비율: {latest['ratio']:.3f} 이탈: {latest['AM_drift']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
            
    with c2:
        st.markdown(f"""
        <div style="background-color: #FFF9E1; padding: 20px; border-radius: 15px; height: 135px;">
            <h4 style="color: #856404; margin: 0;">🟡 3일</h4>
            <p style="color: #856404; font-size: 18px; font-weight: bold; margin-top: 10px;">
                실패: {stats['f3']}회 경고: {stats['w3']}회
            </p>
        </div>
        """, unsafe_allow_html=True)
            
    with c3:
        st.markdown(f"""
        <div style="background-color: #FFE8E8; padding: 20px; border-radius: 15px; height: 135px;">
            <h4 style="color: #A94442; margin: 0;">🔴 7일</h4>
            <p style="color: #A94442; font-size: 15px; font-weight: bold; margin-top: 10px;">
                CV: {stats['cv7']:.2f}% 오전변동: {stats['am_r7']:.4f}<br>
                하지이탈: {latest['leg_drift']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- 그래프 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1.8, 1]})
    ax1.plot(df["검사일시"], df["환측 오전"], 'o-', color='#FF9999', label="환측 오전", linewidth=2, markersize=6)
    ax1.plot(df["검사일시"], df["환측 오후"], '^-', color='#FF0000', label="환측 오후", linewidth=1.5)
    ax1.plot(df["검사일시"], df["건측 오전"], 's--', color='#ADD8E6', alpha=0.5, label="건측 오전")
