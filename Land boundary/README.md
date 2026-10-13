# AI-Assisted Land Boundary Verification & Cadastral Intelligence System

A decision-support system designed for cadastral land surveyors, revenue departments, and municipal land administration agencies. The system integrates computer vision (OpenCV), optical character recognition (Tesseract OCR), geospatial geometry analysis (Shapely & GeoPandas), and an SQLite audit vault to accelerate cadastral boundary digitizing, identify topological discrepancies, and provide surveyor ground-truth verification workflows.

> [!IMPORTANT]
> **Cadastral & Legal Notice:**
> Automatically detected boundaries and OCR survey numbers are computer vision and machine learning decision-support aids. They do **not** constitute legally verified or binding cadastral property boundaries until reviewed, verified, and officially certified by an authorized licensed land surveyor.

---

## 🏛️ System Architecture & Workflow

```
[Cadastral Map: TIFF / PNG / JPG]
                │
                ▼
   [OpenCV Preprocessing Pipeline]
   ├── CLAHE Contrast Enhancement
   ├── Adaptive Gaussian Binarization
   └── Morphological Line Gap Closing
                │
                ▼
   [Candidate Boundary Detection]
   ├── Contour Topology Extraction
   ├── Douglas-Peucker Polygon Approximation
   └── Polsby-Popper Compactness Scoring
                │
                ├─────────────────────────────────────────┐
                ▼                                         ▼
   [Tesseract OCR Text Extraction]          [GIS Spatial Discrepancy Engine]
   ├── Parcel ROI Parsing                  ├── Shapely Polygon Topology Validation
   ├── Regex Survey Number Parsing         ├── Overlap Conflict Detection
   └── Manual Entry Fallback               └── Sliver & Aspect Ratio Detection
                │                                         │
                └───────────────────┬─────────────────────┘
                                    ▼
                     [Surveyor Verification Station]
                     ├── 🟢 Approve Boundary & Survey No.
                     ├── 🟡 Correct Survey No. & Boundary
                     ├── 🔴 Reject Artifact / Scan Noise
                     └── 🔵 Mark for Ground Resurvey
                                    │
                                    ▼
                     [SQLite Audit & Cadastral Vault]
                     ├── Ingested Map Registry
                     ├── Parcel Boundary Geometry (JSON)
                     ├── Immutable Audit Trail
                     └── Export to GeoJSON & CSV
```

---

## 📂 Project Structure

```
Land boundary/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python package dependencies
├── README.md                   # System documentation & execution instructions
├── .gitignore                  # Git ignore rules for virtualenv, uploads, and DB
├── sample_maps/                # Directory containing sample cadastral maps
│   └── village_block_14.png    # Realistic synthetic cadastral survey sheet
└── src/
    ├── __init__.py
    ├── image_processing.py     # OpenCV preprocessing & Douglas-Peucker boundary detection
    ├── ocr_processing.py       # Tesseract OCR extraction with manual fallback
    ├── gis_analysis.py         # Shapely & GeoPandas topology & discrepancy detection
    └── database.py             # SQLite cadastral storage, parcel records & audit logs
```

---

## 🚀 Windows Installation & Execution Guide

### 1. Prerequisites
- **Operating System:** Windows 10 or 11 (64-bit)
- **Python:** Python 3.10+ (tested with Python 3.14)

### 2. Navigate to Project Directory
Open **PowerShell** or **Command Prompt** and navigate to the project directory:
```powershell
cd "c:\Users\dhanuja\Downloads\Land boundary"
```

### 3. (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 5. (Optional) Install Tesseract OCR Engine on Windows
While `pytesseract` Python library is included in `requirements.txt`, the underlying Tesseract C++ engine can be installed via Windows Package Manager:
```powershell
winget install UB-Mannheim.TesseractOCR
```
*Note: If Tesseract OCR is not installed, the application automatically runs in fallback mode with automatic sequential cadastral numbering and allows full manual editing by the surveyor.*

### 6. Launch the Streamlit Web Application
Run the Streamlit application using the Python module launcher:
```powershell
python -m streamlit run app.py
```
Or if Streamlit is directly on your PATH:
```powershell
streamlit run app.py
```

The application will launch and open in your default browser at:
`http://localhost:8501`

---

## 🌟 Key Application Features (Railmark AI UI/UX Redesign)

### 1. High-Tech Sign-In Portal
- Centered glassmorphism modal with glowing cyan accent border (`#00B8E6`), radial gradients, and subtle grid background.
- Role selector: Chief Cadastral Surveyor, Revenue Geographic Officer, GIS Verification Specialist, Field Inspector.
- Jurisdiction sector selector.
- 1-Click **"Quick Demo Sign-In"** button for instant evaluation with realistic sample credentials.

### 2. Left Sidebar Navigation (8 Modules)
1. **Dashboard:** High-impact metric cards (`.stat-card`), verification bar chart, cadastral sheet registry, and system workflow summary.
2. **Land Boundary Map:** High-resolution interactive cadastral map viewer with Railmark high-contrast vector overlays, status filtering (Approved, Pending, Flagged, Corrected), layer toggles, and individual parcel inspection.
3. **AI Boundary Analysis:** Step-by-step visual inspection of OpenCV pipeline (CLAHE Contrast, Bilateral Denoising, Adaptive Binarization, Morphological Closing, Vector Polygons) with live parameter tuning.
4. **Cadastral Map Upload:** Dedicated map upload zone for PNG, JPG, JPEG, and TIFF files, instant preview, scale calibration, and one-click execution of the full AI cadastral pipeline. Also features instant loading of realistic sample data (`Village Block 14`).
5. **Survey Number Extraction:** Tesseract OCR engine status, ROI parcel cropping, text confidence scores, regex parsing, and inline manual surveyor correction form.
6. **Discrepancy Detection:** GIS spatial topology checks via Shapely & GeoPandas. Detects boundary overlap collisions, narrow slivers, and micro-fragments. Export to standardized GeoJSON and CSV.
7. **Survey History:** Official cadastral registry, ground-truth surveyor adjudication workstation (Approve, Correct, Reject, Review with digital signatures), and immutable SQLite audit trail.
8. **System Status:** Real-time diagnostics dashboard for OpenCV, Tesseract OCR, Shapely, GeoPandas, SQLite database, and host environment.

---

## 🎨 Design System (Railmark AI Specification)
- **Background:** Deep Obsidian Navy (`#080F1C`) with 36px geometric grid overlay
- **Cards & Surfaces:** Dark Navy (`#0D1B30`) with fine borders (`#1E3560`) and subtle shadows
- **Elevated Surfaces:** Dark Slate (`#0F1623` / `#162848`)
- **Brand Accents:** Electric Cyan (`#00B8E6` / `#1ACEFF`) and Royal Blue (`#0070E0` / `#0058B0`)
- **Typography:** `Inter` for interface, `JetBrains Mono` for metrics and code
- **Status Badges:** Emerald Green (Approved), Amber Gold (Warning/Corrected), Crimson Red (Flagged/Rejected), Royal Blue (Pending/Info)

