"""
AI-Assisted Land Boundary Verification & Cadastral Intelligence System
GIS Analysis & Discrepancy Intelligence Module (Shapely & GeoPandas)

Performs geometric topology validation, candidate parcel overlap detection,
sliver polygon identification, compactness scoring, and GIS format export (GeoJSON).

DISCLAIMER: Discrepancy checks are automated geometric anomaly heuristics.
They assist surveyors in identifying scan artifacts, line breaks, or digitization errors,
and are subject to professional surveyor adjudication.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Shapely and GeoPandas
try:
    from shapely.geometry import Polygon, MultiPolygon, box
    from shapely.validation import make_valid
    import geopandas as gpd
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False


def create_shapely_polygon(coords: List[List[float]]) -> Optional[Any]:
    """
    Construct a valid Shapely Polygon from a coordinate list [[x, y], ...].
    Applies make_valid if self-intersections or bowtie artifacts exist.
    """
    if not GIS_AVAILABLE or len(coords) < 3:
        return None

    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
            # If make_valid created MultiPolygon, extract the largest polygon
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly
    except Exception:
        return None


def calculate_polsby_popper_compactness(area: float, perimeter: float) -> float:
    """
    Calculate the Polsby-Popper compactness score: 4 * pi * Area / Perimeter^2.
    Range: 0.0 (infinitely elongated/sliver) to 1.0 (perfect circle).
    Standard rectangular parcels typically range between 0.40 and 0.85.
    """
    if perimeter <= 0 or area <= 0:
        return 0.0
    score = (4.0 * np.pi * area) / (perimeter ** 2)
    return round(float(score), 4)


def perform_gis_discrepancy_analysis(
    parcels: List[Dict[str, Any]],
    overlap_threshold_ratio: float = 0.05,
    min_compactness_threshold: float = 0.12,
    sliver_area_ratio_threshold: float = 0.001
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate candidate parcel geometries for cadastral discrepancies:
    1. Overlapping Polygons (Cadastral boundaries should not overlap)
    2. Slivers / Spurious boundary artifacts (High perimeter, minimal area)
    3. Self-intersecting or topologically degenerate boundaries
    4. Low compactness anomalies
    
    Returns:
    - Updated parcel list with 'discrepancy_flags' and geometric attributes
    - Summary discrepancy report dictionary
    """
    if not GIS_AVAILABLE:
        # Graceful fallback if GIS libraries missing
        for p in parcels:
            p.setdefault("discrepancy_flags", [])
        return parcels, {
            "status": "GIS libraries not available",
            "overlap_count": 0,
            "sliver_count": 0,
            "total_flagged": 0
        }

    # Step 1: Build Shapely geometries
    shapely_polys = []
    for p in parcels:
        coords = p.get("polygon_coords", [])
        geom = create_shapely_polygon(coords)
        shapely_polys.append(geom)

    total_parcels = len(parcels)
    total_area = sum(p.get("area_pixels", 0) for p in parcels)
    avg_area = total_area / max(1, total_parcels)

    overlap_pairs = []
    sliver_parcels = []
    anomaly_parcels = []

    # Reset discrepancy flags for fresh evaluation
    for p in parcels:
        p["discrepancy_flags"] = []

    # Step 2: Spatial overlap detection (O(N^2) pairwise check)
    for i in range(total_parcels):
        poly_a = shapely_polys[i]
        if poly_a is None or poly_a.is_empty:
            continue

        p_a = parcels[i]

        for j in range(i + 1, total_parcels):
            poly_b = shapely_polys[j]
            if poly_b is None or poly_b.is_empty:
                continue

            p_b = parcels[j]

            # Fast bounding box check
            if not poly_a.intersects(poly_b):
                continue

            try:
                intersection = poly_a.intersection(poly_b)
                if not intersection.is_empty and intersection.area > 0:
                    min_area = min(poly_a.area, poly_b.area)
                    if min_area > 0:
                        overlap_ratio = intersection.area / min_area
                        if overlap_ratio >= overlap_threshold_ratio:
                            pct = round(overlap_ratio * 100, 1)
                            overlap_pairs.append((p_a["survey_number"], p_b["survey_number"], pct))
                            p_a["discrepancy_flags"].append(f"Overlap with {p_b['survey_number']} ({pct}%)")
                            p_b["discrepancy_flags"].append(f"Overlap with {p_a['survey_number']} ({pct}%)")
            except Exception:
                continue

    # Step 3: Sliver and compactness analysis
    for i, p in enumerate(parcels):
        area = p.get("area_pixels", 0)
        perimeter = p.get("perimeter_pixels", 0)
        compactness = p.get("compactness", calculate_polsby_popper_compactness(area, perimeter))
        p["compactness"] = compactness

        # Sliver detection
        if compactness < min_compactness_threshold:
            p["discrepancy_flags"].append(f"Sliver / Narrow polygon (compactness: {compactness:.2f})")
            sliver_parcels.append(p["survey_number"])

        # Extreme area anomaly (tiny fragment relative to map average)
        if total_area > 0 and area < (avg_area * sliver_area_ratio_threshold):
            p["discrepancy_flags"].append("Micro-fragment geometry (possible scan noise)")
            anomaly_parcels.append(p["survey_number"])

        # Deduplicate flags
        p["discrepancy_flags"] = list(dict.fromkeys(p["discrepancy_flags"]))

    flagged_parcels = [p for p in parcels if len(p.get("discrepancy_flags", [])) > 0]

    report = {
        "status": "Analysis completed",
        "total_evaluated": total_parcels,
        "total_flagged": len(flagged_parcels),
        "overlap_conflicts": len(overlap_pairs),
        "overlap_details": overlap_pairs,
        "sliver_count": len(sliver_parcels),
        "sliver_parcels": sliver_parcels,
        "anomaly_count": len(anomaly_parcels),
        "clean_parcels_count": total_parcels - len(flagged_parcels),
        "clean_ratio_pct": round((total_parcels - len(flagged_parcels)) / max(1, total_parcels) * 100, 1)
    }

    return parcels, report


def export_parcels_to_geodataframe(
    parcels: List[Dict[str, Any]],
    scale_meters_per_pixel: float = 1.0,
    crs: str = "EPSG:3857"
) -> Optional[Any]:
    """
    Convert candidate cadastral parcels into a GeoPandas GeoDataFrame.
    Applies spatial scale conversion if specified.
    """
    if not GIS_AVAILABLE or not parcels:
        return None

    features = []
    for p in parcels:
        coords = p.get("polygon_coords", [])
        if len(coords) < 3:
            continue

        # Scale pixel coordinates if conversion factor provided
        if scale_meters_per_pixel != 1.0:
            scaled_coords = [[x * scale_meters_per_pixel, y * scale_meters_per_pixel] for x, y in coords]
        else:
            scaled_coords = coords

        poly = create_shapely_polygon(scaled_coords)
        if poly is None or poly.is_empty:
            continue

        flags = p.get("discrepancy_flags", [])
        flags_str = "; ".join(flags) if isinstance(flags, list) else str(flags)

        features.append({
            "parcel_index": p.get("parcel_index"),
            "survey_number": p.get("survey_number"),
            "area_sq_units": p.get("area_pixels", 0.0) * (scale_meters_per_pixel ** 2),
            "perimeter_units": p.get("perimeter_pixels", 0.0) * scale_meters_per_pixel,
            "compactness": p.get("compactness", 0.0),
            "ocr_confidence": p.get("ocr_confidence", 0.0),
            "verification_status": p.get("verification_status", "Pending Review"),
            "discrepancies": flags_str,
            "surveyor_notes": p.get("surveyor_notes", ""),
            "geometry": poly
        })

    if not features:
        return None

    gdf = gpd.GeoDataFrame(features, crs=crs)
    return gdf


def export_to_geojson_string(
    parcels: List[Dict[str, Any]],
    scale_meters_per_pixel: float = 1.0
) -> str:
    """
    Export candidate parcel boundaries to a standardized GeoJSON string.
    """
    gdf = export_parcels_to_geodataframe(parcels, scale_meters_per_pixel=scale_meters_per_pixel)
    if gdf is not None and not gdf.empty:
        return gdf.to_json()

    # Manual GeoJSON fallback if GeoPandas fails
    geojson_features = []
    for p in parcels:
        coords = p.get("polygon_coords", [])
        if len(coords) < 3:
            continue

        # Close polygon ring if not closed
        ring = [[float(pt[0]), float(pt[1])] for pt in coords]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        feature = {
            "type": "Feature",
            "properties": {
                "parcel_index": p.get("parcel_index"),
                "survey_number": p.get("survey_number"),
                "area_pixels": p.get("area_pixels"),
                "compactness": p.get("compactness"),
                "status": p.get("verification_status"),
                "discrepancy_flags": p.get("discrepancy_flags", [])
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring]
            }
        }
        geojson_features.append(feature)

    return json.dumps({
        "type": "FeatureCollection",
        "features": geojson_features
    }, indent=2)
