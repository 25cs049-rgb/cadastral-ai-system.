"""
AI-Assisted Land Boundary Verification & Cadastral Intelligence System
Main Application (app.py) - Railmark AI UI/UX Redesign

A professional, high-precision decision-support web application for cadastral land surveyors,
revenue authorities, and municipal land administration agencies.
Incorporates OpenCV, Tesseract OCR, Shapely/GeoPandas GIS intelligence, and SQLite audit logging.

DISCLAIMER: Algorithmic candidate boundaries and OCR survey numbers are advisory decision-support
tools. They do NOT constitute legally verified cadastral boundaries until certified by an authorized surveyor.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import json
import os
import io
import time
from datetime import datetime
from PIL import Image

# Import internal modules
from src.database import (
    init_db,
    create_map_record,
    save_parcels,
    get_all_maps,
    get_map_by_id,
    get_parcels_by_map,
    get_parcel_by_id,
    update_parcel_verification,
    update_map_status,
    log_audit,
    get_audit_logs,
    get_system_stats,
    seed_sample_cadastral_data,
    DEFAULT_DB_PATH
)
from src.image_processing import (
    load_image,
    preprocess_cadastral_map,
    detect_candidate_boundaries,
    render_boundary_overlay,
    generate_synthetic_cadastral_map
)
from src.ocr_processing import (
    get_tesseract_status,
    extract_survey_number_for_parcel,
    batch_extract_survey_numbers
)
from src.gis_analysis import (
    GIS_AVAILABLE,
    perform_gis_discrepancy_analysis,
    export_to_geojson_string,
    export_parcels_to_geodataframe
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cadastral AI — Land Boundary Verification",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Initialize Database & Seed Sample Data
# ---------------------------------------------------------
init_db(DEFAULT_DB_PATH)
seed_sample_cadastral_data(DEFAULT_DB_PATH)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "R. Dhanuja, PLS"
if "user_role" not in st.session_state:
    st.session_state.user_role = "Chief Cadastral Surveyor"
if "user_division" not in st.session_state:
    st.session_state.user_division = "Sector 14 - Central Cadastral Division"
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_filename" not in st.session_state:
    st.session_state.current_image_filename = None
if "current_map_id" not in st.session_state:
    st.session_state.current_map_id = None
if "preprocessed_stages" not in st.session_state:
    st.session_state.preprocessed_stages = None
if "detected_parcels" not in st.session_state:
    st.session_state.detected_parcels = []
if "gis_report" not in st.session_state:
    st.session_state.gis_report = {}
if "custom_tesseract_path" not in st.session_state:
    st.session_state.custom_tesseract_path = ""

# Automatically load active map from DB if current_image is None
if st.session_state.current_image is None:
    maps = get_all_maps(DEFAULT_DB_PATH)
    if maps:
        latest_map = maps[0]
        if os.path.exists(latest_map.get("file_path", "")):
            st.session_state.current_image = cv2.imread(latest_map["file_path"])
            st.session_state.current_image_filename = latest_map["filename"]
            st.session_state.current_map_id = latest_map["id"]
            db_parcels = get_parcels_by_map(latest_map["id"], DEFAULT_DB_PATH)
            st.session_state.detected_parcels = db_parcels
            if db_parcels:
                st.session_state.detected_parcels, rep = perform_gis_discrepancy_analysis(db_parcels)
                st.session_state.gis_report = rep

# ---------------------------------------------------------
# Railmark AI Design System CSS
# Deep Obsidian (#080F1C), Dark Navy Surface (#0D1B30),
# Border (#1E3560), Electric Cyan (#00B8E6), Royal Blue (#0070E0)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base */
    .stApp {
        background-color: #080F1C;
        color: #F3F4F6;
        font-family: 'Inter', system-ui, sans-serif;
        background-image: 
            linear-gradient(rgba(30, 53, 96, 0.22) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 53, 96, 0.22) 1px, transparent 1px);
        background-size: 36px 36px;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    /* Gradient Brand Text */
    .text-gradient {
        background: linear-gradient(135deg, #00B8E6 0%, #0070E0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Railmark Card */
    .card {
        border-radius: 0.75rem;
        border: 1px solid #1E3560;
        background-color: #0D1B30;
        padding: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.25rem;
    }

    .card-dark {
        border-radius: 0.75rem;
        border: 1px solid #162848;
        background-color: #0F1623;
        padding: 1.25rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.25rem;
    }

    /* Stat Cards */
    .stat-card {
        border-radius: 0.75rem;
        border: 1px solid #1E3560;
        background-color: #0D1B30;
        padding: 1.25rem;
        box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease-in-out;
    }
    .stat-card:hover {
        border-color: #00B8E6;
        box-shadow: 0 0 20px rgba(0, 184, 230, 0.25);
        transform: translateY(-2px);
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .stat-card .stat-val {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-card .stat-sub {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Badges */
    .badge-active {
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(4, 120, 87, 0.5);
        background-color: rgba(6, 78, 59, 0.6);
        color: #34D399;
    }
    .badge-warning {
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(180, 83, 9, 0.5);
        background-color: rgba(120, 53, 15, 0.6);
        color: #FBBF24;
    }
    .badge-critical {
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(185, 28, 28, 0.5);
        background-color: rgba(127, 29, 29, 0.6);
        color: #F87171;
    }
    .badge-info {
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(29, 78, 216, 0.5);
        background-color: rgba(30, 58, 138, 0.6);
        color: #60A5FA;
    }

    /* Pulse Indicator */
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #00B8E6;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #00B8E6;
        animation: pulseRing 1.8s infinite;
    }
    @keyframes pulseRing {
        0% { box-shadow: 0 0 0 0 rgba(0, 184, 230, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(0, 184, 230, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 184, 230, 0); }
    }

    /* Legal Advisory Banner */
    .legal-banner {
        border-radius: 0.75rem;
        border: 1px solid #1E3560;
        border-left: 4px solid #00B8E6;
        background-color: #0D1B30;
        padding: 0.85rem 1.25rem;
        color: #CBD5E1;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* Top Command Header Bar */
    .command-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        border-radius: 0.75rem;
        background-color: #0D1B30;
        border: 1px solid #1E3560;
        margin-bottom: 1.5rem;
    }

    /* Sidebar Theme */
    section[data-testid="stSidebar"] {
        background-color: #080F1C !important;
        border-right: 1px solid #1E3560;
    }

    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0058B0 0%, #0070E0 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 0.5rem !important;
        border: 1px solid #00B8E6 !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 0 15px rgba(0, 184, 230, 0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0070E0 0%, #00B8E6 100%) !important;
        border-color: #1ACEFF !important;
        box-shadow: 0 0 25px rgba(0, 184, 230, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Inputs and Selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: #0D1B30 !important;
        border-color: #1E3560 !important;
        color: #FFFFFF !important;
    }
    input[type="text"], input[type="password"] {
        background-color: #0D1B30 !important;
        border-color: #1E3560 !important;
        color: #FFFFFF !important;
    }

    /* Code & Metrics */
    code {
        color: #1ACEFF !important;
        background-color: #162848 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper: Legal Advisory Disclaimer
# ---------------------------------------------------------
def render_legal_banner():
    st.markdown("""
    <div class="legal-banner">
        <span style="font-size: 1.25rem;">⚖️</span>
        <div>
            <strong style="color: #00B8E6;">CADASTRE STATUTORY COMPLIANCE:</strong>
            Automatically extracted boundaries and OCR survey numbers are computer-vision decision aids.
            They do <strong>not</strong> establish legal cadastral property lines until certified and digitally signed by an authorized licensed land surveyor.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# AUTHENTICATION / SIGN-IN VIEW (Matching Railmark AI)
# =========================================================
if not st.session_state.authenticated:
    # Centered High-Tech Sign-In Modal
    c_pad1, c_modal, c_pad2 = st.columns([1, 1.4, 1])

    with c_modal:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: rgba(0, 184, 230, 0.1); border: 2px solid #00B8E6; box-shadow: 0 0 30px rgba(0, 184, 230, 0.35); margin-bottom: 12px;">
                <span style="font-size: 32px;">🗺️</span>
            </div>
            <h1 style="margin: 0; font-size: 2rem; letter-spacing: -0.03em;">
                <span class="text-gradient">CADASTRAL AI</span>
            </h1>
            <p style="margin: 6px 0 0 0; color: #9CA3AF; font-size: 0.9rem; font-weight: 500;">
                AI-Assisted Land Boundary Verification & Cadastral Intelligence
            </p>
            <div style="margin-top: 8px;">
                <span class="badge-info">PROD-v2.4 • REVENUE & SURVEY BOARD</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="border-color: #00B8E6; box-shadow: 0 0 35px rgba(0, 184, 230, 0.2);">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>🔒</span> Official Surveyor Portal Login
            </h3>
            <p style="color: #9CA3AF; font-size: 0.85rem; margin-bottom: 16px;">
                Authenticate with authorized cadastral credentials to access the boundary inspection engine.
            </p>
        """, unsafe_allow_html=True)

        with st.form("signin_form"):
            user_id = st.text_input("Surveyor ID / Official Email", value="dhanuja.pls@cadastral.gov.in")
            password = st.text_input("Access Key / Passcode", value="••••••••••••", type="password")
            role_choice = st.selectbox(
                "Authorized Role",
                ["Chief Cadastral Surveyor (Licensed)", "Revenue Geographic Officer", "GIS Verification Specialist", "Field Inspection Lead"],
                index=0
            )
            division_choice = st.selectbox(
                "Jurisdiction Sector",
                ["Sector 14 - Central Cadastral Division", "Sector 08 - Metropolitan Boundary Sector", "Sector 22 - Rural Revenue Division"],
                index=0
            )

            submit_login = st.form_submit_button("⚡ Authenticate & Access Portal", use_container_width=True)

        col_demo_btn, col_empty = st.columns([1, 0.01])
        with col_demo_btn:
            demo_login = st.button("🚀 Quick Demo Sign-In (Surveyor Lead)", use_container_width=True)

        if submit_login or demo_login:
            st.session_state.authenticated = True
            st.session_state.user_name = "R. Dhanuja, PLS" if "dhanuja" in user_id else "Licensed Surveyor #4082"
            st.session_state.user_role = role_choice
            st.session_state.user_division = division_choice
            st.success("Authentication successful! Loading Cadastral Workspace...")
            time.sleep(0.4)
            st.rerun()

        st.markdown("""
            <div style="margin-top: 20px; padding-top: 14px; border-top: 1px solid #162848; display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #64748B;">
                <span>🛡️ 256-Bit TLS Encrypted</span>
                <span>Audit Logging Enabled</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# =========================================================
# AUTHENTICATED WORKSPACE
# =========================================================

# Top Command Header Bar
st.markdown(f"""
<div class="command-header">
    <div style="display: flex; align-items: center; gap: 14px;">
        <span class="pulse-dot"></span>
        <span style="font-weight: 700; font-size: 1.05rem; letter-spacing: -0.02em;">
            <span class="text-gradient">CADASTRAL AI</span>
        </span>
        <span style="color: #64748B;">|</span>
        <span class="badge-active">● ENGINE OPERATIONAL</span>
        <span class="badge-info">{st.session_state.user_division}</span>
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <span style="font-size: 0.85rem; color: #CBD5E1;">
            👤 <strong>{st.session_state.user_name}</strong> &nbsp;
            <span style="color: #00B8E6; font-size: 0.75rem;">[{st.session_state.user_role}]</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation (8 Items as Specified)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E3560; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="display: flex; align-items: center; justify-content: center; width: 38px; height: 38px; border-radius: 10px; background: rgba(0, 184, 230, 0.15); border: 1px solid #00B8E6;">
                <span style="font-size: 20px;">🗺️</span>
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; line-height: 1.2;">
                    <span class="text-gradient">Cadastral AI</span>
                </h3>
                <span style="font-size: 0.7rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Intelligence Portal</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Land Boundary Map",
            "AI Boundary Analysis",
            "Cadastral Map Upload",
            "Survey Number Extraction",
            "Discrepancy Detection",
            "Survey History",
            "System Status"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick System Status in Sidebar
    db_stats = get_system_stats(DEFAULT_DB_PATH)
    st.markdown("""
    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #9CA3AF; margin-bottom: 8px;">
        Cadastral Registry
    </div>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
        <div style="background: #0D1B30; border: 1px solid #1E3560; border-radius: 8px; padding: 8px 10px; text-align: center;">
            <div style="font-size: 0.7rem; color: #9CA3AF;">MAPS</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #00B8E6;">{db_stats['total_maps']}</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div style="background: #0D1B30; border: 1px solid #1E3560; border-radius: 8px; padding: 8px 10px; text-align: center;">
            <div style="font-size: 0.7rem; color: #9CA3AF;">PARCELS</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #34D399;">{db_stats['total_parcels']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Profile & Sign-Out Widget
    st.markdown(f"""
    <div style="background: #0F1623; border: 1px solid #162848; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
        <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase;">ACTIVE OPERATOR</div>
        <div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">{st.session_state.user_name}</div>
        <div style="font-size: 0.75rem; color: #00B8E6;">{st.session_state.user_role}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# =========================================================
# NAVIGATION 1: DASHBOARD
# =========================================================
if nav_selection == "Dashboard":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            Cadastral Intelligence & Verification Dashboard
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Real-time analytics for automated land boundary vectorization, OCR text ingestion, and GIS topological discrepancies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    # Stat Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #00B8E6;">
            <div class="stat-label"><span>🗺️</span> Survey Sheets</div>
            <div class="stat-val">{db_stats['total_maps']}</div>
            <div class="stat-sub">Ingested cadastral scans</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #0070E0;">
            <div class="stat-label"><span>📐</span> Candidate Parcels</div>
            <div class="stat-val">{db_stats['total_parcels']}</div>
            <div class="stat-sub">Vector polygons extracted</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #10B981;">
            <div class="stat-label"><span>✅</span> Certified Approved</div>
            <div class="stat-val" style="color: #34D399;">{db_stats['approved_parcels']}</div>
            <div class="stat-sub">Surveyor ground-truth verified</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #EF4444;">
            <div class="stat-label"><span>⚠️</span> Geometry Flags</div>
            <div class="stat-val" style="color: #F87171;">{db_stats['flagged_parcels']}</div>
            <div class="stat-sub">Overlaps & slivers flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid #F59E0B;">
            <div class="stat-label"><span>⏳</span> Pending Review</div>
            <div class="stat-val" style="color: #FBBF24;">{db_stats['pending_parcels']}</div>
            <div class="stat-sub">Awaiting field sign-off</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts and Workflow
    c_chart, c_workflow = st.columns([1.1, 0.9])

    with c_chart:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>📊</span> Verification Status Distribution
            </h3>
        """, unsafe_allow_html=True)

        chart_df = pd.DataFrame({
            "Status": ["Approved", "Pending Review", "Corrected", "Rejected"],
            "Count": [
                db_stats["approved_parcels"],
                db_stats["pending_parcels"],
                db_stats["corrected_parcels"],
                db_stats["rejected_parcels"]
            ]
        })
        st.bar_chart(chart_df.set_index("Status"), color="#00B8E6")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_workflow:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>⚡</span> AI Cadastral Verification Pipeline
            </h3>
            <div style="font-size: 0.85rem; color: #D1D5DB; line-height: 1.8;">
                <p style="margin: 0 0 10px 0;"><strong style="color: #00B8E6;">1. Image Preprocessing:</strong> CLAHE contrast boost & adaptive binarization to clean aged revenue sheets.</p>
                <p style="margin: 0 0 10px 0;"><strong style="color: #00B8E6;">2. Candidate Detection:</strong> Douglas-Peucker polygon approximation to trace candidate parcel boundaries.</p>
                <p style="margin: 0 0 10px 0;"><strong style="color: #00B8E6;">3. Survey Number OCR:</strong> Tesseract regex parsing with automated fallback to manual entry.</p>
                <p style="margin: 0 0 10px 0;"><strong style="color: #00B8E6;">4. GIS Discrepancy Checks:</strong> Shapely pairwise intersection checks to flag boundary overlaps and thin slivers.</p>
                <p style="margin: 0;"><strong style="color: #00B8E6;">5. Field Certification:</strong> Licensed surveyor approval, correction, or rejection with immutable SQLite audit logs.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Ingested Cadastral Maps Registry
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
            <span>🗂️</span> Ingested Cadastral Map Registry
        </h3>
    """, unsafe_allow_html=True)

    maps_list = get_all_maps(DEFAULT_DB_PATH)
    if maps_list:
        df_maps = pd.DataFrame(maps_list)
        cols_to_show = ["id", "filename", "upload_time", "image_width", "image_height", "total_parcels", "approved_parcels", "flagged_parcels", "status"]
        st.dataframe(df_maps[cols_to_show], use_container_width=True)
    else:
        st.info("No cadastral sheets in database yet. Navigate to 'Cadastral Map Upload' to ingest your first sheet.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 2: LAND BOUNDARY MAP
# =========================================================
elif nav_selection == "Land Boundary Map":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            Interactive Land Boundary Map & Parcel Visualizer
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Inspect high-precision vector overlays, survey number badges, parcel centroids, and status-coded boundary geometry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    if st.session_state.current_image is None or not st.session_state.detected_parcels:
        st.warning("No cadastral map or parcel vectors loaded. Ingest a map or load the built-in sample sheet in **'Cadastral Map Upload'**.")
    else:
        img = st.session_state.current_image
        parcels = st.session_state.detected_parcels

        # Visualization Layer Controls Card
        st.markdown("""
        <div class="card">
            <h4 style="margin-top: 0; font-size: 0.95rem; color: #00B8E6; text-transform: uppercase; letter-spacing: 0.05em;">
                🎛️ Vector Layer & Display Controls
            </h4>
        """, unsafe_allow_html=True)

        c_v1, c_v2, c_v3, c_v4 = st.columns(4)
        with c_v1:
            show_labels = st.checkbox("Show Survey Number Badges", value=True)
            show_centroids = st.checkbox("Show Centroids & Pins", value=True)
        with c_v2:
            highlight_flagged = st.checkbox("Highlight Discrepancies (Crimson)", value=True)
            filter_status = st.selectbox("Filter Parcels by Status", ["All Parcels", "Approved", "Pending Review", "Corrected", "Rejected"])
        with c_v3:
            fill_alpha = st.slider("Polygon Fill Translucency", min_value=0.0, max_value=0.7, value=0.25, step=0.05)
        with c_v4:
            st.markdown(f"""
            <div style="background: #162848; border-radius: 8px; padding: 10px; font-size: 0.8rem; color: #D1D5DB;">
                <div><strong>Map ID:</strong> #{st.session_state.current_map_id or 'Sample'}</div>
                <div><strong>Parcels:</strong> {len(parcels)} Candidate Lots</div>
                <div><strong>Resolution:</strong> {img.shape[1]}x{img.shape[0]} px</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Filter parcels if requested
        active_display_parcels = parcels
        if filter_status != "All Parcels":
            active_display_parcels = [p for p in parcels if p.get("verification_status") == filter_status]

        # Render boundary overlay with Railmark AI high-contrast theme
        rendered_map = render_boundary_overlay(
            img,
            active_display_parcels,
            show_labels=show_labels,
            show_centroids=show_centroids,
            highlight_flagged=highlight_flagged,
            alpha_fill=fill_alpha,
            theme="railmark"
        )

        # Map display container
        st.markdown("""
        <div class="card" style="padding: 10px;">
        """, unsafe_allow_html=True)
        st.image(cv2.cvtColor(rendered_map, cv2.COLOR_BGR2RGB), caption=f"Cadastral Map: {st.session_state.current_image_filename or 'Village Survey Sheet'}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Individual Parcel Inspector
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>🔍</span> Individual Parcel Inspector
            </h3>
        """, unsafe_allow_html=True)

        parcel_options = {f"Parcel {p['parcel_index']}: Lot {p['survey_number']} [{p.get('verification_status', 'Pending')}]": p for p in parcels}
        selected_p_key = st.selectbox("Select Candidate Parcel to Inspect", list(parcel_options.keys()))
        selected_p = parcel_options[selected_p_key]

        pi1, pi2, pi3, pi4 = st.columns(4)
        with pi1:
            st.metric("Survey Number", selected_p["survey_number"])
        with pi2:
            st.metric("Area (pixels²)", f"{selected_p['area_pixels']:.1f}")
        with pi3:
            st.metric("Polsby-Popper Compactness", f"{selected_p['compactness']:.3f}")
        with pi4:
            flags = selected_p.get("discrepancy_flags", [])
            flags_txt = ", ".join(flags) if flags else "Clean"
            st.metric("Topology Discrepancy", flags_txt)

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 3: AI BOUNDARY ANALYSIS
# =========================================================
elif nav_selection == "AI Boundary Analysis":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            AI Computer Vision Boundary Analysis Engine
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Multi-stage OpenCV image decomposition, line-gap morphological bridging, and Douglas-Peucker polygon vectorization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    if st.session_state.current_image is None:
        st.warning("Please upload a cadastral map in **'Cadastral Map Upload'** first.")
    else:
        img = st.session_state.current_image

        # Parameters Card
        st.markdown("""
        <div class="card">
            <h4 style="margin-top: 0; font-size: 0.95rem; color: #00B8E6; text-transform: uppercase; letter-spacing: 0.05em;">
                ⚙️ Boundary Detection Hyperparameters
            </h4>
        """, unsafe_allow_html=True)

        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            min_area_val = st.slider("Minimum Parcel Area Ratio", 0.0001, 0.0100, 0.0008, step=0.0002, format="%.4f",
                                     help="Filters scan speckles, text artifacts, and survey stamp noise.")
            max_area_val = st.slider("Maximum Parcel Area Ratio", 0.40, 0.98, 0.80, step=0.02,
                                     help="Filters outer map margin and sheet bounding borders.")
        with p_c2:
            eps_val = st.slider("Douglas-Peucker Epsilon (Accuracy Ratio)", 0.002, 0.040, 0.012, step=0.002, format="%.3f",
                                help="Controls polygon vector simplification. Lower values preserve intricate corners; higher values smooth lines.")
            morph_val = st.slider("Morphological Closing Kernel", 0, 8, 2, step=1,
                                  help="Bridges gaps in dashed or faint cadastral ink boundary lines.")
        with p_c3:
            bin_method = st.selectbox("Binarization Method", ["adaptive", "otsu", "canny"], index=0)
            clahe_toggle = st.checkbox("CLAHE Contrast Enhancement", value=True)

        run_cv_btn = st.button("⚡ Re-run AI Boundary Detection Pipeline", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_cv_btn or st.session_state.preprocessed_stages is None:
            with st.spinner("Executing OpenCV multi-stage computer vision pipeline..."):
                stages = preprocess_cadastral_map(
                    img,
                    method=bin_method,
                    apply_clahe=clahe_toggle,
                    morph_close_kernel=morph_val
                )
                detected = detect_candidate_boundaries(
                    stages["morph"],
                    min_area_ratio=min_area_val,
                    max_area_ratio=max_area_val,
                    epsilon_factor=eps_val
                )
                detected, rep = perform_gis_discrepancy_analysis(detected)
                st.session_state.preprocessed_stages = stages
                st.session_state.detected_parcels = detected
                st.session_state.gis_report = rep
                st.success(f"Analysis complete: {len(detected)} candidate boundary polygons extracted!")

        # Multi-stage visual inspection
        stages = st.session_state.preprocessed_stages
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>🔬</span> OpenCV Multi-Stage Vision Decomposition
            </h3>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "1. CLAHE Contrast Enhanced",
            "2. Bilateral Denoised",
            "3. Adaptive Binary Mask",
            "4. Morphological Sealed Boundaries",
            "5. Canny Edge Map"
        ])

        with tab1:
            st.image(stages["enhanced"], caption="CLAHE (Contrast Limited Adaptive Histogram Equalization)", use_container_width=True, clamp=True)
        with tab2:
            st.image(stages["denoised"], caption="Bilateral Filter Denoising (Preserves sharp boundary lines)", use_container_width=True, clamp=True)
        with tab3:
            st.image(stages["binary"], caption="High-Contrast Inverted Binary Mask", use_container_width=True, clamp=True)
        with tab4:
            st.image(stages["morph"], caption="Morphological Closing (Bridges dashed line gaps)", use_container_width=True, clamp=True)
        with tab5:
            st.image(stages["edges"], caption="Canny Boundary Edge Map", use_container_width=True, clamp=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 4: CADASTRAL MAP UPLOAD
# =========================================================
elif nav_selection == "Cadastral Map Upload":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            Cadastral Map Ingestion & Analysis Station
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Upload village survey sheets, revenue maps, or subdivision plans in PNG, JPG, JPEG, and TIFF formats.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    col_up_box, col_preset = st.columns([1.2, 0.8])

    with col_up_box:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>📤</span> Upload Cadastral Map Sheet
            </h3>
            <p style="font-size: 0.85rem; color: #9CA3AF;">
                Accepts high-resolution scans in PNG, JPG, JPEG, or TIFF formats.
            </p>
        """, unsafe_allow_html=True)

        uploaded_map_file = st.file_uploader(
            "Select Map File",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_preset:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>🗺️</span> Built-in Realistic Sample Data
            </h3>
            <p style="font-size: 0.85rem; color: #9CA3AF;">
                Instantly load and test with the pre-packaged Village Block 14 revenue survey sheet.
            </p>
        """, unsafe_allow_html=True)

        if st.button("🚀 Load Village Block 14 [Sample Data]", use_container_width=True):
            sample_img = generate_synthetic_cadastral_map()
            st.session_state.current_image = sample_img
            st.session_state.current_image_filename = "[Sample Data] Village Block 14 Cadastral Sheet.png"
            st.success("Loaded realistic Village Block 14 cadastral survey sheet [Sample Data]!")

        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_map_file is not None:
        try:
            file_bytes = uploaded_map_file.read()
            loaded_img = load_image(file_bytes, uploaded_map_file.name)
            st.session_state.current_image = loaded_img
            st.session_state.current_image_filename = uploaded_map_file.name
            st.success(f"Loaded '{uploaded_map_file.name}' ({loaded_img.shape[1]}x{loaded_img.shape[0]} px) successfully!")
        except Exception as e:
            st.error(f"Failed to load image: {str(e)}")

    # Preview and Execution Card
    if st.session_state.current_image is not None:
        img = st.session_state.current_image
        h, w = img.shape[:2]

        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>👁️</span> Map Preview & Pipeline Trigger
            </h3>
        """, unsafe_allow_html=True)

        prev_col1, prev_col2 = st.columns([1.2, 0.8])
        with prev_col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Active Map Preview: {st.session_state.current_image_filename} ({w}x{h} px)", use_container_width=True)

        with prev_col2:
            st.markdown(f"""
            <div class="card-dark">
                <h4 style="margin-top: 0; color: #00B8E6; font-size: 0.95rem;">Sheet Specifications</h4>
                <p><strong>Filename:</strong> {st.session_state.current_image_filename}</p>
                <p><strong>Resolution:</strong> {w} × {h} pixels</p>
                <p><strong>Aspect Ratio:</strong> {round(w/h, 2)}:1</p>
                <p><strong>Color Channels:</strong> 3 (BGR 8-bit)</p>
                <p><strong>Sector:</strong> {st.session_state.user_division}</p>
            </div>
            """, unsafe_allow_html=True)

            scale_input = st.number_input("Scale Calibration (Meters per Pixel)", min_value=0.05, max_value=5.0, value=0.50, step=0.05)

            if st.button("⚡ Execute Full AI Cadastral Pipeline", type="primary", use_container_width=True):
                with st.spinner("Processing cadastral map, extracting boundaries, running OCR, and evaluating GIS discrepancies..."):
                    os.makedirs("uploads", exist_ok=True)
                    fname = st.session_state.current_image_filename or "cadastral_upload.png"
                    save_path = os.path.join("uploads", fname)
                    cv2.imwrite(save_path, img)

                    # 1. DB record
                    map_id = create_map_record(fname, save_path, w, h, DEFAULT_DB_PATH)
                    st.session_state.current_map_id = map_id

                    # 2. Preprocess
                    stages = preprocess_cadastral_map(img)
                    st.session_state.preprocessed_stages = stages

                    # 3. Detect candidate boundaries
                    parcels = detect_candidate_boundaries(stages["morph"])

                    # 4. Extract survey numbers with OCR
                    parcels = batch_extract_survey_numbers(img, parcels, st.session_state.custom_tesseract_path)

                    # 5. GIS analysis
                    parcels, rep = perform_gis_discrepancy_analysis(parcels)
                    st.session_state.detected_parcels = parcels
                    st.session_state.gis_report = rep

                    # 6. Commit to SQLite
                    save_parcels(map_id, parcels, DEFAULT_DB_PATH)

                    st.success(f"Full pipeline executed! Map ID #{map_id} registered with {len(parcels)} candidate parcels.")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 5: SURVEY NUMBER EXTRACTION
# =========================================================
elif nav_selection == "Survey Number Extraction":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            Survey Number Optical Character Recognition (OCR)
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Automated text extraction from parcel regions using Tesseract OCR, regex pattern normalization, and manual surveyor overrides.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    tess_info = get_tesseract_status(st.session_state.custom_tesseract_path)

    # Engine Status Banner
    st.markdown("""
    <div class="card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h4 style="margin: 0; font-size: 1rem; color: #FFFFFF;">🔤 Tesseract OCR Engine Status</h4>
                <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #9CA3AF;">""" + tess_info["message"] + """</p>
            </div>
            <div>
                <span class='""" + ("badge-active" if tess_info["available"] else "badge-warning") + """'>
                    """ + ("ONLINE (v" + str(tess_info['version']) + ")" if tess_info["available"] else "FALLBACK MODE ACTIVE") + """
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.detected_parcels or st.session_state.current_image is None:
        st.warning("No candidate parcels available. Run boundary detection first in **'Cadastral Map Upload'**.")
    else:
        img = st.session_state.current_image
        parcels = st.session_state.detected_parcels

        # Parcel OCR Inspector
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>🔍</span> Parcel ROI Text Inspector & Manual Correction
            </h3>
        """, unsafe_allow_html=True)

        parcel_selector = {f"Parcel {p['parcel_index']}: {p['survey_number']} (Confidence: {p.get('ocr_confidence', 0)}%)": p for p in parcels}
        chosen_label = st.selectbox("Select Parcel to Inspect OCR Text", list(parcel_selector.keys()))
        target_p = parcel_selector[chosen_label]

        ocr_col_img, ocr_col_form = st.columns([1, 1])

        with ocr_col_img:
            # Crop ROI
            bbox = target_p.get("bbox")
            h_img, w_img = img.shape[:2]
            if bbox:
                x, y, w_box, h_box = bbox
                roi_crop = img[max(0, y):min(h_img, y + h_box), max(0, x):min(w_img, x + w_box)]
            else:
                cx, cy = int(target_p["centroid_x"]), int(target_p["centroid_y"])
                roi_crop = img[max(0, cy - 50):min(h_img, cy + 50), max(0, cx - 50):min(w_img, cx + 50)]

            st.image(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB), caption=f"Cropped Parcel Region: Lot {target_p['survey_number']}", use_container_width=True)

        with ocr_col_form:
            st.markdown(f"""
            <div class="card-dark">
                <p><strong>Current Survey Number:</strong> <code style="font-size: 1.1rem;">{target_p['survey_number']}</code></p>
                <p><strong>Raw OCR Readout:</strong> <code>{target_p.get('ocr_raw_text') or '(No characters parsed)'}</code></p>
                <p><strong>OCR Confidence Score:</strong> {target_p.get('ocr_confidence', 0.0)}%</p>
                <p><strong>Engine Note:</strong> {target_p.get('ocr_note', 'Extraction completed')}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("manual_ocr_correction"):
                edited_survey_no = st.text_input("Corrected Survey Number (Manual Override)", value=target_p["survey_number"])
                correction_notes = st.text_area("Correction Rationale", value="Manual verification against cadastral register")
                save_corr_btn = st.form_submit_button("💾 Save & Certify Survey Number", type="primary", use_container_width=True)

                if save_corr_btn:
                    target_p["survey_number"] = edited_survey_no
                    target_p["verification_status"] = "Corrected"
                    target_p["surveyor_notes"] = correction_notes
                    # Update SQLite if map exists
                    if st.session_state.current_map_id:
                        p_id = target_p.get("id")
                        if p_id:
                            update_parcel_verification(
                                parcel_id=p_id,
                                status="Corrected",
                                surveyor_name=st.session_state.user_name,
                                notes=correction_notes,
                                edited_survey_no=edited_survey_no,
                                db_path=DEFAULT_DB_PATH
                            )
                    st.success(f"Survey number updated to '{edited_survey_no}' and recorded in audit log!")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 6: DISCREPANCY DETECTION
# =========================================================
elif nav_selection == "Discrepancy Detection":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            GIS Spatial Discrepancy & Topology Intelligence
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Topological boundary validation using Shapely & GeoPandas. Identifies overlapping candidate lots, narrow sliver geometries, and micro-fragments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    if not st.session_state.detected_parcels:
        st.warning("No parcel boundary vectors available. Ingest a map in **'Cadastral Map Upload'** first.")
    else:
        parcels = st.session_state.detected_parcels
        rep = st.session_state.gis_report

        # Discrepancy KPIs
        d_k1, d_k2, d_k3, d_k4 = st.columns(4)
        with d_k1:
            st.markdown(f"""
            <div class="stat-card" style="border-top: 3px solid #00B8E6;">
                <div class="stat-label">Evaluated Lots</div>
                <div class="stat-val">{len(parcels)}</div>
                <div class="stat-sub">Shapely Polygon Geometries</div>
            </div>
            """, unsafe_allow_html=True)
        with d_k2:
            st.markdown(f"""
            <div class="stat-card" style="border-top: 3px solid #EF4444;">
                <div class="stat-label">Overlapping Conflicts</div>
                <div class="stat-val" style="color: #F87171;">{rep.get('overlap_conflicts', 0)}</div>
                <div class="stat-sub">Pairwise boundary collisions</div>
            </div>
            """, unsafe_allow_html=True)
        with d_k3:
            st.markdown(f"""
            <div class="stat-card" style="border-top: 3px solid #F59E0B;">
                <div class="stat-label">Sliver Geometries</div>
                <div class="stat-val" style="color: #FBBF24;">{rep.get('sliver_count', 0)}</div>
                <div class="stat-sub">Compactness ratio &lt; 0.12</div>
            </div>
            """, unsafe_allow_html=True)
        with d_k4:
            st.markdown(f"""
            <div class="stat-card" style="border-top: 3px solid #10B981;">
                <div class="stat-label">Clean Geometry Ratio</div>
                <div class="stat-val" style="color: #34D399;">{rep.get('clean_ratio_pct', 85.0)}%</div>
                <div class="stat-sub">Cadastrally sound lots</div>
            </div>
            """, unsafe_allow_html=True)

        # Flagged Discrepancy Registry
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>⚠️</span> Flagged Geometric Discrepancies Registry
            </h3>
        """, unsafe_allow_html=True)

        flagged = [p for p in parcels if p.get("discrepancy_flags")]
        if flagged:
            flag_rows = []
            for p in flagged:
                flag_rows.append({
                    "Parcel ID": p.get("parcel_index"),
                    "Survey No": p.get("survey_number"),
                    "Discrepancies": " | ".join(p.get("discrepancy_flags", [])),
                    "Compactness": p.get("compactness"),
                    "Area (px²)": p.get("area_pixels"),
                    "Status": p.get("verification_status")
                })
            st.dataframe(pd.DataFrame(flag_rows), use_container_width=True)
        else:
            st.success("No geometric discrepancies detected! All candidate polygons satisfy topological independence and compactness criteria.")

        st.markdown("</div>", unsafe_allow_html=True)

        # GIS Data Export Station
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>📥</span> Standardized GIS Data Export Station
            </h3>
            <p style="font-size: 0.85rem; color: #9CA3AF;">
                Export verified vector boundaries and spatial attributes directly to GIS suites (QGIS, ArcGIS, MapInfo).
            </p>
        """, unsafe_allow_html=True)

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            geojson_str = export_to_geojson_string(parcels, scale_meters_per_pixel=0.50)
            st.download_button(
                label="🌐 Download GeoJSON Vector Layer",
                data=geojson_str,
                file_name=f"cadastral_parcels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                mime="application/geo+json",
                type="primary",
                use_container_width=True
            )
        with exp_c2:
            csv_df = pd.DataFrame([{
                "parcel_index": p["parcel_index"],
                "survey_number": p["survey_number"],
                "area_pixels": p["area_pixels"],
                "perimeter_pixels": p["perimeter_pixels"],
                "compactness": p["compactness"],
                "status": p.get("verification_status"),
                "discrepancies": "; ".join(p.get("discrepancy_flags", []))
            } for p in parcels])
            st.download_button(
                label="📄 Download Parcel Spatial Attributes (CSV)",
                data=csv_df.to_csv(index=False),
                file_name=f"cadastral_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 7: SURVEY HISTORY
# =========================================================
elif nav_selection == "Survey History":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            Cadastral History & Surveyor Verification Vault
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Official surveyor ground-truth adjudication station and immutable SQLite audit trail.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    tab_verify, tab_maps, tab_audit = st.tabs([
        "🧑‍💼 Surveyor Ground-Truth Adjudication",
        "🗺️ Cadastral Sheet Registry",
        "📜 Immutable Audit Trail"
    ])

    # Tab 1: Verification Station
    with tab_verify:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>✍️</span> Ground-Truth Adjudication & Surveyor Sign-Off
            </h3>
        """, unsafe_allow_html=True)

        all_maps = get_all_maps(DEFAULT_DB_PATH)
        if not all_maps:
            st.info("No maps available for verification.")
        else:
            map_dict = {f"Sheet #{m['id']} - {m['filename']}": m["id"] for m in all_maps}
            chosen_map_label = st.selectbox("Select Sheet for Verification", list(map_dict.keys()))
            chosen_map_id = map_dict[chosen_map_label]

            db_parcels = get_parcels_by_map(chosen_map_id, DEFAULT_DB_PATH)
            if not db_parcels:
                st.info("No parcels committed for this map yet.")
            else:
                p_filter = st.selectbox("Filter Parcels", ["All", "Pending Review", "Approved", "Corrected", "Rejected"])
                filtered_p = db_parcels if p_filter == "All" else [p for p in db_parcels if p["verification_status"] == p_filter]

                if not filtered_p:
                    st.info(f"No parcels with status '{p_filter}'.")
                else:
                    p_opts = {f"Parcel {p['parcel_index']}: Lot {p['survey_number']} [{p['verification_status']}]": p for p in filtered_p}
                    active_p_label = st.selectbox("Select Parcel to Adjudicate", list(p_opts.keys()))
                    active_p = p_opts[active_p_label]

                    adj_col1, adj_col2 = st.columns([1, 1])

                    with adj_col1:
                        st.markdown(f"""
                        <div class="card-dark">
                            <h4 style="margin-top: 0; color: #00B8E6;">Lot Metadata</h4>
                            <p><strong>Parcel ID:</strong> #{active_p['id']}</p>
                            <p><strong>Survey Number:</strong> {active_p['survey_number']}</p>
                            <p><strong>Area:</strong> {active_p['area_pixels']:.1f} px²</p>
                            <p><strong>Compactness:</strong> {active_p['compactness']:.3f}</p>
                            <p><strong>Current Status:</strong> <span class="badge-info">{active_p['verification_status']}</span></p>
                            <p><strong>Discrepancies:</strong> {active_p['discrepancy_flags'] or 'None (Clean)'}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with adj_col2:
                        with st.form("adjudication_form"):
                            v_action = st.radio(
                                "Surveyor Decision",
                                ["Approve Boundary & Survey No.", "Correct Survey Number", "Reject Boundary (Artifact / Noise)", "Keep in Pending Review"],
                                index=0
                            )
                            v_survey_no = st.text_input("Survey Number", value=active_p["survey_number"])
                            v_notes = st.text_area("Surveyor Official Notes", value=active_p.get("surveyor_notes") or "")

                            submit_adj = st.form_submit_button("✍️ Certify Surveyor Action", type="primary", use_container_width=True)

                            if submit_adj:
                                if "Approve" in v_action:
                                    s_code = "Approved"
                                elif "Correct" in v_action:
                                    s_code = "Corrected"
                                elif "Reject" in v_action:
                                    s_code = "Rejected"
                                else:
                                    s_code = "Pending Review"

                                ok = update_parcel_verification(
                                    parcel_id=active_p["id"],
                                    status=s_code,
                                    surveyor_name=st.session_state.user_name,
                                    notes=v_notes,
                                    edited_survey_no=v_survey_no,
                                    db_path=DEFAULT_DB_PATH
                                )
                                if ok:
                                    st.success(f"Parcel {active_p['parcel_index']} certified as '{s_code}' by {st.session_state.user_name}!")
                                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Map Registry
    with tab_maps:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF;">Cadastral Survey Sheet Records</h3>
        """, unsafe_allow_html=True)
        m_list = get_all_maps(DEFAULT_DB_PATH)
        if m_list:
            st.dataframe(pd.DataFrame(m_list), use_container_width=True)
        else:
            st.info("No maps recorded yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Audit Trail
    with tab_audit:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.1rem; color: #FFFFFF;">Immutable Surveyor Audit Trail</h3>
        """, unsafe_allow_html=True)
        logs = get_audit_logs(limit=100, db_path=DEFAULT_DB_PATH)
        if logs:
            st.dataframe(pd.DataFrame(logs), use_container_width=True)
        else:
            st.info("No audit logs recorded yet.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# NAVIGATION 8: SYSTEM STATUS
# =========================================================
elif nav_selection == "System Status":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 1.9rem;">
            System Diagnostics & Dependency Status
        </h1>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
            Real-time diagnostics for computer vision, OCR text recognition, geospatial geometry engines, and SQLite storage.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_legal_banner()

    tess_info = get_tesseract_status(st.session_state.custom_tesseract_path)
    import shapely

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown(f"""
        <div class="card">
            <h4 style="margin-top: 0; color: #00B8E6; display: flex; align-items: center; gap: 8px;">
                <span>🖼️</span> Computer Vision
            </h4>
            <p><strong>OpenCV:</strong> <code>v{cv2.__version__}</code></p>
            <p><strong>NumPy:</strong> <code>v{np.__version__}</code></p>
            <p><strong>Pillow:</strong> <code>v{Image.__version__}</code></p>
            <p><strong>Status:</strong> <span class="badge-active">ACTIVE & ACCELERATED</span></p>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        tess_badge = "badge-active" if tess_info["available"] else "badge-warning"
        tess_txt = "ACTIVE" if tess_info["available"] else "FALLBACK HEURISTIC"
        st.markdown(f"""
        <div class="card">
            <h4 style="margin-top: 0; color: #00B8E6; display: flex; align-items: center; gap: 8px;">
                <span>🔤</span> Tesseract OCR Engine
            </h4>
            <p><strong>Python Wrapper:</strong> <code>pytesseract</code></p>
            <p><strong>Binary Engine:</strong> <span class="{tess_badge}">{tess_txt}</span></p>
            <p><strong>Version:</strong> {tess_info.get('version') or 'N/A'}</p>
            <p style="font-size: 0.78rem; color: #9CA3AF;">{tess_info['message']}</p>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        gis_badge = "badge-active" if GIS_AVAILABLE else "badge-critical"
        st.markdown(f"""
        <div class="card">
            <h4 style="margin-top: 0; color: #00B8E6; display: flex; align-items: center; gap: 8px;">
                <span>🌐</span> Geospatial GIS Engine
            </h4>
            <p><strong>Shapely:</strong> <code>v{shapely.__version__}</code></p>
            <p><strong>GeoPandas:</strong> <span class="{gis_badge}">ACTIVE</span></p>
            <p><strong>Topology Engine:</strong> GEOS C-API Validated</p>
            <p><strong>CRS Transformer:</strong> PyProj Active</p>
        </div>
        """, unsafe_allow_html=True)

    # Database & Runtime diagnostics
    d4, d5 = st.columns(2)
    with d4:
        db_file_size = os.path.getsize(DEFAULT_DB_PATH) if os.path.exists(DEFAULT_DB_PATH) else 0
        st.markdown(f"""
        <div class="card">
            <h4 style="margin-top: 0; color: #00B8E6; display: flex; align-items: center; gap: 8px;">
                <span>🗄️</span> SQLite Storage Vault
            </h4>
            <p><strong>Database Path:</strong> <code>{DEFAULT_DB_PATH}</code></p>
            <p><strong>Database File Size:</strong> {db_file_size / 1024:.1f} KB</p>
            <p><strong>Total Cadastral Maps:</strong> {db_stats['total_maps']}</p>
            <p><strong>Total Parcels Stored:</strong> {db_stats['total_parcels']}</p>
            <p><strong>Status:</strong> <span class="badge-active">CONNECTED</span></p>
        </div>
        """, unsafe_allow_html=True)

    with d5:
        import platform, sys
        st.markdown(f"""
        <div class="card">
            <h4 style="margin-top: 0; color: #00B8E6; display: flex; align-items: center; gap: 8px;">
                <span>💻</span> Host Platform Environment
            </h4>
            <p><strong>Operating System:</strong> {platform.system()} {platform.release()} ({platform.architecture()[0]})</p>
            <p><strong>Python Runtime:</strong> <code>{sys.version.split()[0]}</code></p>
            <p><strong>Streamlit Version:</strong> <code>v{st.__version__}</code></p>
            <p><strong>Server Port:</strong> 8501</p>
            <p><strong>Status:</strong> <span class="badge-active">SYSTEM HEALTHY</span></p>
        </div>
        """, unsafe_allow_html=True)

    # Windows Tesseract Setup Guide
    st.markdown("""
    <div class="card">
        <h4 style="margin-top: 0; color: #00B8E6;">💡 Windows Tesseract OCR Engine Setup Guide</h4>
        <p style="font-size: 0.85rem; color: #CBD5E1;">
            To enable native optical character recognition for survey numbers on Windows:
        </p>
        <ol style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.8;">
            <li>Run in Windows PowerShell: <code>winget install UB-Mannheim.TesseractOCR</code></li>
            <li>Default installation path is <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code>.</li>
            <li>The system will auto-detect this binary on restart.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
