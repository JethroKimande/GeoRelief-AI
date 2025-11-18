"""
Configuration management for GeoRelief-AI
Manages all file paths and settings for easy scaling to new countries/disasters
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Input Data Paths (Raw data will be placed here)
DATA_DIR = BASE_DIR / "raw_data"
ADMIN_BOUNDARIES_DIR = DATA_DIR / "admin_boundaries"
POPULATION_DIR = DATA_DIR / "population"  # WorldPop GeoTIFFs
INFRASTRUCTURE_DIR = DATA_DIR / "infrastructure"  # HDX/OSM data
DISASTER_EVENTS_DIR = DATA_DIR / "disaster_events"  # Flood/Conflict/Drought data
VULNERABILITY_DIR = DATA_DIR / "vulnerability"  # Water access, displacement data
LAND_COVER_DIR = DATA_DIR / "land_cover"  # Land use/land cover data

# Processed Data Paths
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MASTER_DATASET_PATH = PROCESSED_DATA_DIR / "global_master_dataset.geojson"

# Model Path
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "priority_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Feature names (standardized across all countries)
STANDARD_FEATURES = [
    'population_density',
    'health_facility_density',
    'vulnerability_index',
    'flooded_population',  # Disaster-specific magnitude
    'alluvial_plain_pct'  # Land cover: percentage of county covered by alluvial plains (flood-prone)
]

# Target variable
TARGET_VARIABLE = 'PriorityScore'

