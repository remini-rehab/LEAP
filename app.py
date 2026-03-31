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

# 1. 폰트 설정 (나눔바른고딕 최우선)
FONT_PATH = "NanumBarunGothic.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rc("font", family="NanumBarunGothic")
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False

# --- 유틸리티 함수 ---
def safe_fmt(value, fmt=".4f", na="-"):
    if pd.isna(value): return na
    try: return format(float(value), fmt)
    except: return str(value)

def extract_file_info(filename: str):
    stem = os.path.splitext(filename)[0]
    date_match = re.search(r"(\d{8}|\d{6})$", stem)
    raw_date = date_match.group(1) if date_match else None
    patient_name = stem[: stem.rfind(raw_date)].rstrip("_- ") if raw_date else stem
    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    return patient_name, report_date

# --- 2. 분석 엔진 (용어/로직 고도화 버전) ---
def preprocess_and_analyze(df_raw, sheet_name):
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    
    # 기계 원본 컬럼 매핑
    col_map = {
        "우측상지 세포외수분비": "우측 상지", "우측 상지 세포외수분비": "우측 상지",
        "좌측상지 세포외수분비": "좌측 상지", "좌측 상지 세포외수분비": "좌측 상지",
        "체간 세포외수분비": "체간", "우측하지 세포외수분비": "우측 하지", "좌측하지 세포외수분비": "좌측 하지"
    }
    df = df.rename(columns=col_map)
    
    # 환측 판단
    side = "우측" if "우측" in str(sheet_name) else "좌측"
    df["검사일시"] = pd.to_datetime(df["검사일시"], errors="coerce")
    df = df.dropna(subset=["검사일시"]).sort_values("검사일시")
    
    df["환측"] = df["우측 상지"] if side == "우측" else df["좌측 상지"]
    df["건측"] = df["좌측 상지"] if side == "우측" else df["우측 상지"]
    
    # 오전/오후 통합
    df["Time_Period"] = df["검사일시"].dt.hour.apply(lambda h: "오전" if 4 <= h < 12 else "오후")
    df["Date_Only"] = df["검사일시"].dt.date
    
    am = df[df["Time_Period"] == "오전"].groupby("Date_Only").first()
    pm = df[df["Time_Period"] == "오후"].groupby("Date_Only").last()
    
    daily = pd.merge(am[["환측", "건측", "우측 하지", "좌측 하지", "체간"]], 
                     pm[["환측", "건측"]], on="Date_Only", how="outer", suffixes=(" 오전", " 오후"))
    daily = daily.sort_index().reset_index()
    daily["검사일시"] = pd.to_datetime(daily["Date_Only"])
    daily["하지 평균"] = (daily["우측 하지"] + daily["좌측 하지"]) / 2
    
    # --- 지표 계산 ---
    b_arm = daily["환측 오전"].iloc[:3].mean()
    b_leg = daily["하지 평균"].iloc[:3].mean()
    b_trunk = daily["체간"].iloc[:3].mean()
    
    daily["ratio"] = daily["환측 오전"] / daily["건측 오전"]
    daily["AM_drift"] = daily["환측 오전"] - b_arm
    daily["day_gain"] = daily["환측 오후"] - daily["환측 오전"]
    daily["night_recovery"] = daily["환측 오전"].shift(-1) - daily["환측 오후"]
    
    daily["leg_drift"] = daily["하지 평균"] - b_leg
    daily["trunk_drift"] = daily["체간"] - b_trunk
    daily["AM_3day_range"] = daily["환측 오전"].rolling(3).apply(lambda x: x.max() - x.min())
    
    return daily, b_arm, b_leg, b_trunk

# --- 3. 그래프 (별표 마킹 시스템) ---
def create_enhanced_figure(df, b_arm):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={'height_ratios': [1.8, 1]})
    
    ax1.plot(df["검사일시"], df["환측 오전"], 'o-', color='#FF9999', label="환측 오전", linewidth=2)
    ax1.plot(df["검사일시"], df["환측 오후"], '^-', color='#FF0000', label="환측 오후", alpha=0.7)
    ax1.plot(df["검사일시"], df["건측 오전"], 's--', color='#ADD8E6', alpha=0.5, label="건측 오전")
    ax1.axhline(b_arm, color='red', linestyle=':', alpha=0.4, label="Baseline")
    
    # 1.02 경고 별표
    warns = df[df["ratio"] > 1.02]
    if not warns.empty:
        ax1.scatter(warns["검사일시"], warns["환측 오전"] + 0.0005, marker='*', color='red', s=200, zorder=10, label="1.02 경고")
        for d in warns["검사일시"]: ax1.axvline(x=d, color='red', linestyle=':', alpha=0.2)
            
    ax1.set_title("상지 동태 분석 (★: 위험 경계치 초과)", fontsize=14, weight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    ax2.plot(df["검사일시"], df["leg_drift"], 'o-', color='purple', label="하지 이탈")
    ax2.plot(df["검사일시"], df["trunk_drift"], 's-', color='green', label="체간 이탈")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("전신 기준선 이탈 추이", fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig

# --- 메인 실행 UI ---
st.title("🏥 LEAP 정밀 분석 시스템")
uploaded_file = st.file_uploader("멀티시트 엑셀 업로드", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    mode = st.radio("모드 선택", ["👤 개별 환자", "📦 일괄 출력"], horizontal=True)

    if mode == "👤 개별 환자":
        selected_sheet = st.selectbox("환자 선택", sheet_names)
        df, b_arm, b_leg, b_trunk = preprocess_and_analyze(pd.read_excel(xls, sheet_name=selected_sheet), selected_sheet)
        latest = df.iloc[-1]; r3 = df.tail(3); r7 = df.tail(7)
        
        # 통계 계산
        f3 = (r3["night_recovery"] >= 0).sum(); w3 = (r3["ratio"] > 1.02).sum()
        f7 = (r7["night_recovery"] >= 0).sum(); w7 = (r7["ratio"] > 1.02).sum()
        am_r7 = r7["환측 오전"].max() - r7["환측 오전"].min()
        cv7 = (r7["ratio"].std() / r7["ratio"].mean()) * 100 if r7["ratio"].mean() != 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1: st.info(f"**🔵 당일**\n\n비율: {latest['ratio']:.3f}\n이탈: {latest['AM_drift']:.4f}")
        with c2: st.warning(f"**🟡 3일**\n\n실패: {f3}회\n경고: {w3}회")
        with c3: st.error(f"**🔴 7일**\n\nCV: {cv7:.2f}%\n오전변동: {am_r7:.4f}\n하지이탈: {latest['leg_drift']:.4f}")

        st.pyplot(create_enhanced_figure(df, b_arm))
