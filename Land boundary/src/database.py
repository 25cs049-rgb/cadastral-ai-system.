"""
AI-Assisted Land Boundary Verification & Cadastral Intelligence System
Database Module (SQLite)

Handles persistent storage for cadastral maps, candidate parcel boundaries,
surveyor verification records, discrepancy logs, and audit trails.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cadastral_system.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a database connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database tables if they do not exist."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Maps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_width INTEGER,
                image_height INTEGER,
                file_path TEXT,
                total_parcels INTEGER DEFAULT 0,
                approved_parcels INTEGER DEFAULT 0,
                flagged_parcels INTEGER DEFAULT 0,
                status TEXT DEFAULT 'Uploaded',
                notes TEXT
            )
        """)

        # Parcels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parcels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                map_id INTEGER NOT NULL,
                parcel_index INTEGER NOT NULL,
                survey_number TEXT,
                polygon_json TEXT NOT NULL,
                area_pixels REAL,
                perimeter_pixels REAL,
                compactness REAL,
                centroid_x REAL,
                centroid_y REAL,
                ocr_confidence REAL DEFAULT 0.0,
                ocr_raw_text TEXT,
                verification_status TEXT DEFAULT 'Pending Review',
                discrepancy_flags TEXT,
                surveyor_notes TEXT,
                verified_by TEXT,
                verified_at TIMESTAMP,
                FOREIGN KEY (map_id) REFERENCES maps(id) ON DELETE CASCADE
            )
        """)

        # Audit logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                map_id INTEGER,
                parcel_id INTEGER,
                action TEXT NOT NULL,
                previous_value TEXT,
                new_value TEXT,
                actor TEXT DEFAULT 'Surveyor',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                comments TEXT
            )
        """)

        conn.commit()


def create_map_record(filename: str, file_path: str, width: int, height: int, db_path: str = DEFAULT_DB_PATH) -> int:
    """Create a new map ingestion record and return its ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO maps (filename, file_path, image_width, image_height, status)
            VALUES (?, ?, ?, ?, 'Uploaded')
        """, (filename, file_path, width, height))
        conn.commit()
        map_id = cursor.lastrowid
        
        log_audit(
            map_id=map_id,
            parcel_id=None,
            action="MAP_UPLOADED",
            prev_val=None,
            new_val=filename,
            actor="System",
            comments=f"Uploaded cadastral map ({width}x{height}px)",
            db_path=db_path
        )
        return map_id


def save_parcels(map_id: int, parcels_data: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH) -> None:
    """Save or replace detected candidate parcels for a map."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Clear any prior detected parcels for this map if re-processing
        cursor.execute("DELETE FROM parcels WHERE map_id = ?", (map_id,))

        flagged_count = 0
        for p in parcels_data:
            flags = p.get("discrepancy_flags", [])
            flags_str = ", ".join(flags) if isinstance(flags, list) else str(flags or "")
            if flags_str.strip():
                flagged_count += 1

            polygon_json = json.dumps(p.get("polygon_coords", []))
            cursor.execute("""
                INSERT INTO parcels (
                    map_id, parcel_index, survey_number, polygon_json,
                    area_pixels, perimeter_pixels, compactness,
                    centroid_x, centroid_y, ocr_confidence, ocr_raw_text,
                    verification_status, discrepancy_flags, surveyor_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                map_id,
                p.get("parcel_index", 0),
                p.get("survey_number", f"P-{p.get('parcel_index', 0)}"),
                polygon_json,
                p.get("area_pixels", 0.0),
                p.get("perimeter_pixels", 0.0),
                p.get("compactness", 0.0),
                p.get("centroid_x", 0.0),
                p.get("centroid_y", 0.0),
                p.get("ocr_confidence", 0.0),
                p.get("ocr_raw_text", ""),
                p.get("verification_status", "Pending Review"),
                flags_str,
                p.get("surveyor_notes", "")
            ))

        total_parcels = len(parcels_data)
        cursor.execute("""
            UPDATE maps
            SET total_parcels = ?,
                flagged_parcels = ?,
                status = 'Processed'
            WHERE id = ?
        """, (total_parcels, flagged_count, map_id))

        conn.commit()

        log_audit(
            map_id=map_id,
            parcel_id=None,
            action="PARCELS_DETECTED",
            prev_val=None,
            new_val=f"{total_parcels} candidate parcels",
            actor="CV Pipeline",
            comments=f"Detected {total_parcels} candidates; {flagged_count} geometry flags identified.",
            db_path=db_path
        )


def get_all_maps(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all ingested maps."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM maps ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_map_by_id(map_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve details for a single map."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM maps WHERE id = ?", (map_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_parcels_by_map(map_id: int, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all parcels belonging to a map."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM parcels WHERE map_id = ? ORDER BY parcel_index ASC", (map_id,))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["polygon_coords"] = json.loads(d["polygon_json"])
            except Exception:
                d["polygon_coords"] = []
            results.append(d)
        return results


def get_parcel_by_id(parcel_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single parcel by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM parcels WHERE id = ?", (parcel_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["polygon_coords"] = json.loads(d["polygon_json"])
        except Exception:
            d["polygon_coords"] = []
        return d


def update_parcel_verification(
    parcel_id: int,
    status: str,
    surveyor_name: str,
    notes: str = "",
    edited_survey_no: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Update verification status for a specific parcel.
    Actions supported: Approved, Corrected, Rejected, Under Review.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT map_id, survey_number, verification_status FROM parcels WHERE id = ?", (parcel_id,))
        current = cursor.fetchone()
        if not current:
            return False

        map_id = current["map_id"]
        old_status = current["verification_status"]
        old_survey_no = current["survey_number"]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_survey_no = edited_survey_no if edited_survey_no is not None and edited_survey_no.strip() else old_survey_no

        cursor.execute("""
            UPDATE parcels
            SET verification_status = ?,
                survey_number = ?,
                surveyor_notes = ?,
                verified_by = ?,
                verified_at = ?
            WHERE id = ?
        """, (status, new_survey_no, notes, surveyor_name, now, parcel_id))

        # Recompute approved count for the parent map
        cursor.execute("""
            SELECT COUNT(*) AS approved_count
            FROM parcels
            WHERE map_id = ? AND verification_status = 'Approved'
        """, (map_id,))
        approved_count = cursor.fetchone()["approved_count"]

        cursor.execute("""
            SELECT COUNT(*) AS total_count
            FROM parcels
            WHERE map_id = ?
        """, (map_id,))
        total_count = cursor.fetchone()["total_count"]

        # If all parcels approved or reviewed, update map status accordingly
        new_map_status = "In Review"
        if approved_count == total_count and total_count > 0:
            new_map_status = "Fully Verified"

        cursor.execute("""
            UPDATE maps
            SET approved_parcels = ?,
                status = ?
            WHERE id = ?
        """, (approved_count, new_map_status, map_id))

        conn.commit()

        # Audit log
        action_detail = f"Status: {old_status} -> {status}"
        if new_survey_no != old_survey_no:
            action_detail += f" | Survey No: {old_survey_no} -> {new_survey_no}"

        log_audit(
            map_id=map_id,
            parcel_id=parcel_id,
            action=f"PARCEL_{status.upper()}",
            prev_val=f"{old_survey_no} ({old_status})",
            new_val=f"{new_survey_no} ({status})",
            actor=surveyor_name or "Surveyor",
            comments=notes or action_detail,
            db_path=db_path
        )
        return True


def update_map_status(map_id: int, status: str, notes: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> None:
    """Update general status of a cadastral map."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE maps SET status = ?, notes = COALESCE(?, notes) WHERE id = ?", (status, notes, map_id))
        conn.commit()
        log_audit(
            map_id=map_id,
            parcel_id=None,
            action="MAP_STATUS_UPDATE",
            prev_val=None,
            new_val=status,
            actor="Surveyor",
            comments=notes or f"Map status updated to {status}",
            db_path=db_path
        )


def log_audit(
    map_id: Optional[int],
    parcel_id: Optional[int],
    action: str,
    prev_val: Optional[str],
    new_val: Optional[str],
    actor: str = "Surveyor",
    comments: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Record an audit trail event."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (map_id, parcel_id, action, previous_value, new_value, actor, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (map_id, parcel_id, action, prev_val, new_val, actor, comments))
            conn.commit()
    except Exception as e:
        # Avoid crashing app if logging fails
        print(f"Audit log failed: {e}")


def get_audit_logs(map_id: Optional[int] = None, limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve audit history logs."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if map_id:
            cursor.execute("""
                SELECT * FROM audit_logs
                WHERE map_id = ?
                ORDER BY id DESC LIMIT ?
            """, (map_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM audit_logs
                ORDER BY id DESC LIMIT ?
            """, (limit,))
        return [dict(r) for r in cursor.fetchall()]


def get_system_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculate key system statistics for dashboard."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) AS total_maps FROM maps")
        total_maps = cursor.fetchone()["total_maps"]

        cursor.execute("SELECT COUNT(*) AS total_parcels FROM parcels")
        total_parcels = cursor.fetchone()["total_parcels"]

        cursor.execute("SELECT COUNT(*) AS approved_parcels FROM parcels WHERE verification_status = 'Approved'")
        approved_parcels = cursor.fetchone()["approved_parcels"]

        cursor.execute("SELECT COUNT(*) AS corrected_parcels FROM parcels WHERE verification_status = 'Corrected'")
        corrected_parcels = cursor.fetchone()["corrected_parcels"]

        cursor.execute("SELECT COUNT(*) AS rejected_parcels FROM parcels WHERE verification_status = 'Rejected'")
        rejected_parcels = cursor.fetchone()["rejected_parcels"]

        cursor.execute("SELECT COUNT(*) AS pending_parcels FROM parcels WHERE verification_status = 'Pending Review'")
        pending_parcels = cursor.fetchone()["pending_parcels"]

        cursor.execute("SELECT COUNT(*) AS flagged_parcels FROM parcels WHERE discrepancy_flags IS NOT NULL AND discrepancy_flags != ''")
        flagged_parcels = cursor.fetchone()["flagged_parcels"]

        return {
            "total_maps": total_maps,
            "total_parcels": total_parcels,
            "approved_parcels": approved_parcels,
            "corrected_parcels": corrected_parcels,
            "rejected_parcels": rejected_parcels,
            "pending_parcels": pending_parcels,
            "flagged_parcels": flagged_parcels,
        }


def seed_sample_cadastral_data(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Populate realistic sample cadastral records if database has no parcels,
    clearly labeled as sample data as required by the user instructions.
    """
    import numpy as np
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS c FROM parcels")
        if cursor.fetchone()["c"] > 0:
            return  # Already has parcels

    try:
        from src.image_processing import generate_synthetic_cadastral_map, preprocess_cadastral_map, detect_candidate_boundaries
        from src.gis_analysis import perform_gis_discrepancy_analysis
        import cv2

        sample_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_maps", "village_block_14.png")
        if not os.path.exists(sample_map_path):
            os.makedirs(os.path.dirname(sample_map_path), exist_ok=True)
            img = generate_synthetic_cadastral_map()
            cv2.imwrite(sample_map_path, img)
        else:
            img = cv2.imread(sample_map_path)

        if img is None:
            return

        h, w = img.shape[:2]
        filename = "[Sample Data] Village Block 14 Cadastral Sheet.png"
        map_id = create_map_record(filename, sample_map_path, w, h, db_path)

        # Preprocess and detect candidate boundaries
        stages = preprocess_cadastral_map(img)
        parcels = detect_candidate_boundaries(stages["morph"], min_area_ratio=0.001, max_area_ratio=0.7)
        parcels, rep = perform_gis_discrepancy_analysis(parcels)

        sample_survey_numbers = [
            "101", "102/1", "102/2", "103", "104-A", "104-B",
            "105", "106/A", "106/B", "107", "108-A", "108-B", "109"
        ]
        statuses = [
            "Approved", "Approved", "Approved", "Pending Review",
            "Corrected", "Pending Review", "Approved", "Pending Review",
            "Rejected", "Pending Review", "Approved", "Pending Review", "Pending Review"
        ]

        for i, p in enumerate(parcels):
            if i < len(sample_survey_numbers):
                p["survey_number"] = sample_survey_numbers[i]
            if i < len(statuses):
                p["verification_status"] = statuses[i]
            p["ocr_confidence"] = float(np.random.randint(88, 98)) if p["verification_status"] == "Approved" else float(np.random.randint(68, 86))
            p["ocr_raw_text"] = f"Survey Lot #{p['survey_number']}"
            if p["verification_status"] == "Approved":
                p["verified_by"] = "Chief Surveyor #4082"
                p["surveyor_notes"] = "[Sample Data] Boundary verified against revenue boundary records."
            elif p["verification_status"] == "Corrected":
                p["verified_by"] = "Chief Surveyor #4082"
                p["surveyor_notes"] = "[Sample Data] Survey number manually corrected from OCR noise."
            elif p["verification_status"] == "Rejected":
                p["verified_by"] = "Chief Surveyor #4082"
                p["surveyor_notes"] = "[Sample Data] Rejected false boundary from scanner crease."

        save_parcels(map_id, parcels, db_path)
        update_map_status(map_id, "In Review", notes="[Sample Data] Demonstration survey sheet.", db_path=db_path)

    except Exception as e:
        print(f"Seed sample data error: {e}")

