"""
Global CSS for the Rainbrain 2 app.
"""

import streamlit as st


def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] { background:#0f172a; border-right:1px solid #1e293b; }
    section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap:6px; background:#f1f5f9; padding:6px 8px; border-radius:12px; border:none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px; padding:8px 18px; font-size:.82rem; font-weight:500;
        color:#64748b; background:transparent; border:none;
    }
    .stTabs [aria-selected="true"] { background:#1e40af !important; color:white !important; }

    [data-testid="metric-container"] {
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
        padding:14px 18px; box-shadow:0 1px 4px rgba(0,0,0,.04);
    }
    [data-testid="metric-container"] label { color:#64748b !important; font-size:.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size:1.5rem !important; font-weight:700; color:#0f172a;
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#1e40af,#3b82f6); color:white; border:none;
        border-radius:8px; font-weight:600; padding:.55rem 1.4rem;
    }
    .section-header {
        display:flex; align-items:center; gap:10px;
        padding:0 0 6px 0; border-bottom:2px solid #e2e8f0; margin-bottom:20px;
    }
    .section-header h2 { margin:0; font-size:1.25rem; font-weight:700; color:#0f172a; }
    .section-badge {
        background:#1e40af; color:white; border-radius:6px;
        padding:3px 9px; font-size:.72rem; font-weight:600; letter-spacing:.04em;
    }
    .info-box {
        background:#eff6ff; border:1px solid #bfdbfe; border-left:4px solid #3b82f6;
        border-radius:8px; padding:12px 16px; font-size:.85rem; color:#1e40af; margin-bottom:16px;
    }
    .positive-beta-box {
        background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #22c55e;
        border-radius:8px; padding:12px 16px; font-size:.85rem; color:#15803d; margin-bottom:16px;
    }
    .ng-box {
        background:#faf5ff; border:1px solid #e9d5ff; border-left:4px solid #9333ea;
        border-radius:8px; padding:12px 16px; font-size:.85rem; color:#6b21a8; margin-bottom:16px;
    }
    .per-channel-box {
        background:#fff7ed; border:1px solid #fed7aa; border-left:4px solid #f97316;
        border-radius:8px; padding:12px 16px; font-size:.85rem; color:#9a3412; margin-bottom:16px;
    }
    .prophet-box {
        background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a;
        border-radius:8px; padding:12px 16px; font-size:.85rem; color:#15803d; margin-bottom:16px;
    }
    .weibull-zone {
        background:#fdf4ff; border:2px dashed #a855f7; border-radius:10px;
        padding:18px 20px; font-family:'JetBrains Mono',monospace;
        font-size:.78rem; color:#6b21a8; margin:12px 0;
    }
    .stDataFrame { border-radius:10px; overflow:hidden; }

    /* ── Bigger, non-clipping multiselect ("variable selection") widgets ──
       Default BaseWeb select caps the tag container's height and scrolls
       once it fills up, hiding already-selected items. Let it grow and
       wrap freely instead, so every selected variable stays visible. */
    .stMultiSelect [data-baseweb="select"] > div {
        max-height: none !important;
        height: auto !important;
        min-height: 46px;
        overflow: visible !important;
        flex-wrap: wrap !important;
        align-items: flex-start !important;
        padding-top: 4px; padding-bottom: 4px;
    }
    .stMultiSelect [data-baseweb="tag"] {
        margin: 3px 4px 3px 0 !important;
        max-width: 100% !important;
    }
    .stMultiSelect [data-baseweb="tag"] span[title] {
        max-width: 220px !important;
        overflow: visible !important;
        white-space: normal !important;
    }
    </style>
    """, unsafe_allow_html=True)
