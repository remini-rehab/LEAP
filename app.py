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

# --- UI 메인 ---
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

    # --- 에러 방지용 대시보드 카드 (단일 따옴표 사용) ---
    c1, c2, c3 = st.columns(3)
    
    # 당일 카드
    c1.markdown(f'<div style="background-color: #E8F1FF; padding: 18px; border-radius: 15px; height: 145px; border: 1px solid #D0E2FF;">'
                f'<h4 style="color: #0056B3; margin: 0; font-size: 1.2rem;">🔵 당일 <span style="font-size: 0.85rem; color: #555;">({latest["Date_Key"]})</span></h4>
