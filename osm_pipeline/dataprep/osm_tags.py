"""OSM tag → 6-class mapping.

Classes (must match training set):
    road=0, water=1, foliage=2, building=3, grass=4, ground=5

Each entry below is consumed by `osmnx.features_from_bbox` (tag dict) and the
returned GeoDataFrames are then unioned per class. `ground` is computed
implicitly as (city_bbox - union(other 5)).
"""
from __future__ import annotations

# osmnx tag queries per class. Values can be True (any) or list of strings.
# NOTE: foliage is NOT fetched from OSM -- OSM tree data is too sparse in
# suburbs. Foliage is produced by SAM3 directly on the RGB satellite tile
# (see scripts/sam3_foliage.py and dataprep/raster_utils.py:compose_seg).
TAG_QUERIES: dict[str, dict] = {
    "grass": {
        "landuse": ["grass", "meadow", "recreation_ground", "village_green",
                    "cemetery", "allotments"],
        "leisure": ["park", "garden", "pitch", "playground", "golf_course"],
        "natural": ["grassland", "heath"],
    },
    "water": {
        "natural": ["water", "bay", "strait", "coastline"],
        "waterway": ["river", "stream", "canal", "riverbank"],
        "landuse": ["reservoir", "basin"],
    },
    "building": {
        "building": True,
    },
    # roads are fetched as a graph (LineString) and buffered separately.
}

# Highway types we keep for the road class. Anything else is dropped.
ROAD_HIGHWAY_KEEP = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "service", "unclassified", "motorway_link",
    "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "living_street",
}

# Class-paint priority when polygons overlap (higher = drawn on top in raster).
# building > road > water > grass > foliage > ground
CLASS_PRIORITY = {
    "ground":   0,
    "foliage":  1,
    "grass":    2,
    "water":    3,
    "road":     4,
    "building": 5,
}

CLASS_IDS = {
    "road":     0,
    "water":    1,
    "foliage":  2,
    "building": 3,
    "grass":    4,
    "ground":   5,
}
