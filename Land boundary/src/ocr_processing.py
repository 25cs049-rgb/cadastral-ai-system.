"""
AI-Assisted Land Boundary Verification & Cadastral Intelligence System
OCR Processing Module (Tesseract OCR & Fallbacks)

Extracts cadastral survey numbers from parcel image regions using Tesseract OCR,
with robust regex pattern extraction, confidence scoring, and seamless manual fallback.

DISCLAIMER: OCR extracted survey numbers are candidate suggestions requiring
surveyor review and verification before legal record entry.
"""

import os
import re
import shutil
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Try importing pytesseract
try:
    import pytesseract
    PYTESSERACT_INSTALLED = True
except ImportError:
    pytesseract = None
    PYTESSERACT_INSTALLED = False

# Common Windows installation locations for Tesseract OCR
STANDARD_TESSERACT_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]


def resolve_tesseract_path(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Search for Tesseract executable in PATH, standard Windows directories, or custom user path.
    """
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    system_path = shutil.which("tesseract")
    if system_path and os.path.exists(system_path):
        return system_path

    for candidate in STANDARD_TESSERACT_WINDOWS_PATHS:
        if os.path.exists(candidate):
            return candidate

    return None


def get_tesseract_status(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check availability of Tesseract OCR Python wrapper and underlying engine binary.
    """
    if not PYTESSERACT_INSTALLED:
        return {
            "available": False,
            "pytesseract_installed": False,
            "binary_path": None,
            "version": None,
            "message": "pytesseract Python library is not installed."
        }

    resolved_bin = resolve_tesseract_path(custom_path)
    if not resolved_bin:
        return {
            "available": False,
            "pytesseract_installed": True,
            "binary_path": None,
            "version": None,
            "message": "Tesseract OCR engine (tesseract.exe) was not found in PATH or standard directories. Please install Tesseract or specify custom path."
        }

    try:
        pytesseract.pytesseract.tesseract_cmd = resolved_bin
        version = str(pytesseract.get_tesseract_version())
        return {
            "available": True,
            "pytesseract_installed": True,
            "binary_path": resolved_bin,
            "version": version,
            "message": f"Tesseract OCR v{version} active at: {resolved_bin}"
        }
    except Exception as e:
        return {
            "available": False,
            "pytesseract_installed": True,
            "binary_path": resolved_bin,
            "version": None,
            "message": f"Found binary at {resolved_bin} but verification failed: {str(e)}"
        }


def preprocess_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped parcel region for optimal OCR readability:
    - Upscaling with bicubic interpolation
    - Grayscale conversion
    - Contrast enhancement and Otsu binarization
    - Noise clearance
    """
    if roi.size == 0:
        return roi

    # Upscale if small
    h, w = roi.shape[:2]
    if h < 60 or w < 60:
        scale = max(2.5, 80.0 / max(h, 1))
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Gaussian blur & Otsu thresholding
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Border padding
    padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded


def extract_survey_number_from_text(raw_text: str) -> Optional[str]:
    """
    Parse candidate survey numbers from OCR string output using regex patterns.
    Matches formats like '104', '104/1', '104/1A', '45-B', '72', '23A'.
    """
    if not raw_text:
        return None

    cleaned = raw_text.strip().replace("\n", " ")

    # Cadastral survey number patterns (e.g. 102, 102/1, 102-A, 45/2B)
    patterns = [
        r"\b\d{1,5}[/-][A-Za-z0-9]{1,4}\b",  # e.g., 104/1, 45-B, 12/A
        r"\b\d{1,5}[A-Za-z]\b",               # e.g., 104A, 42B
        r"\b\d{1,5}\b",                        # pure numbers: 104, 72, 8
    ]

    for pat in patterns:
        matches = re.findall(pat, cleaned)
        if matches:
            # Pick the most substantial match
            return matches[0]

    return None


def extract_survey_number_for_parcel(
    image: np.ndarray,
    parcel: Dict[str, Any],
    custom_tesseract_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract survey number for a single parcel using Tesseract OCR,
    with automatic fallback to default label if OCR is unavailable or fails.
    """
    status_info = get_tesseract_status(custom_tesseract_path)
    idx = parcel.get("parcel_index", 1)
    fallback_no = f"CAD-{idx:03d}"

    if not status_info["available"]:
        return {
            "survey_number": parcel.get("survey_number") or fallback_no,
            "ocr_confidence": 0.0,
            "ocr_raw_text": "",
            "is_fallback": True,
            "note": "OCR engine unavailable; default placeholder assigned (manual entry available)"
        }

    # Crop parcel region around centroid or bounding box
    bbox = parcel.get("bbox")
    h_img, w_img = image.shape[:2]

    if bbox:
        x, y, w, h = bbox
        # Add slight margin inside parcel
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        x1 = max(0, x + pad_x)
        y1 = max(0, y + pad_y)
        x2 = min(w_img, x + w - pad_x)
        y2 = min(h_img, y + h - pad_y)
        roi = image[y1:y2, x1:x2]
    else:
        cx, cy = int(parcel.get("centroid_x", 0)), int(parcel.get("centroid_y", 0))
        r = 40
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w_img, cx + r), min(h_img, cy + r)
        roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            "survey_number": fallback_no,
            "ocr_confidence": 0.0,
            "ocr_raw_text": "",
            "is_fallback": True,
            "note": "Empty parcel ROI"
        }

    processed_roi = preprocess_roi_for_ocr(roi)

    try:
        # Tesseract configuration for sparse digits / alphanumeric codes
        tess_config = "--psm 6 -c tessedit_char_whitelist=0123456789/-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        data = pytesseract.image_to_data(processed_roi, config=tess_config, output_type=pytesseract.Output.DICT)

        texts = []
        confidences = []
        for text, conf in zip(data["text"], data["conf"]):
            if text.strip() and int(conf) > 0:
                texts.append(text.strip())
                confidences.append(float(conf))

        raw_str = " ".join(texts)
        extracted = extract_survey_number_from_text(raw_str)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        if extracted:
            return {
                "survey_number": extracted,
                "ocr_confidence": round(avg_conf, 1),
                "ocr_raw_text": raw_str,
                "is_fallback": False,
                "note": f"Extracted via OCR ({round(avg_conf, 1)}% confidence)"
            }
        else:
            return {
                "survey_number": fallback_no,
                "ocr_confidence": round(avg_conf, 1),
                "ocr_raw_text": raw_str,
                "is_fallback": True,
                "note": "No distinct survey number parsed from OCR text; placeholder assigned"
            }

    except Exception as e:
        return {
            "survey_number": fallback_no,
            "ocr_confidence": 0.0,
            "ocr_raw_text": "",
            "is_fallback": True,
            "note": f"OCR extraction error: {str(e)}"
        }


def batch_extract_survey_numbers(
    image: np.ndarray,
    parcels: List[Dict[str, Any]],
    custom_tesseract_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process all candidate parcels, attempting OCR extraction for each.
    Updates parcel dicts in-place and returns the list.
    """
    for p in parcels:
        ocr_result = extract_survey_number_for_parcel(image, p, custom_tesseract_path)
        p["survey_number"] = ocr_result["survey_number"]
        p["ocr_confidence"] = ocr_result["ocr_confidence"]
        p["ocr_raw_text"] = ocr_result["ocr_raw_text"]
        p["ocr_note"] = ocr_result["note"]
        p["is_ocr_fallback"] = ocr_result["is_fallback"]

    return parcels
