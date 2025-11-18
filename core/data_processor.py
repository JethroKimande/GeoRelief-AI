"""
Modular data processing for GeoRelief-AI
Standardizes data from different countries into common features
"""

import geopandas as gpd
import pandas as pd # type: ignore
import rasterio # type: ignore
from rasterio.mask import mask
from rasterstats import zonal_stats
import numpy as np
from pathlib import Path
import json
from typing import Optional
from . import config


def find_data_file(directory: Path, country_iso: str, extensions: Optional[list[str]] = None, pattern: Optional[str] = None) -> Optional[Path]:
    """
    Helper function to find data files with flexible naming.
    
    Args:
        directory: Base directory to search
        country_iso: Country code
        extensions: List of file extensions to search for (e.g., ['.shp', '.geojson'])
        pattern: Optional pattern to match in filename
    
    Returns:
        Path to found file or None
    """
    if extensions is None:
        extensions = ['.shp', '.geojson', '.gpkg']
    
    country_dir = directory / country_iso.lower()
    
    if not country_dir.exists():
        return None
    
    # Search for files with given extensions
    for ext in extensions:
        files = list(country_dir.rglob(f"*{ext}"))
        if files:
            if pattern:
                # Filter by pattern if provided
                files = [f for f in files if pattern.lower() in f.name.lower()]
                # Sort to prioritize ADM1 over ADM0, ADM2, etc.
                if "adm" in pattern.lower():
                    files.sort(key=lambda x: x.name)
            if files:
                return files[0]  # Return first match
    
    return None


def load_and_process_data(country_iso="KEN", admin_level=1):
    """
    Main function to process all data for a given country
    and return a standardized GeoDataFrame.
    
    Args:
        country_iso: ISO country code (e.g., "KEN" for Kenya)
        admin_level: Administrative level (1=states/provinces, 2=counties/districts)
    
    Returns:
        GeoDataFrame with standardized features
    """
    
    # 1. Load Admin Boundaries - try multiple naming conventions
    admin_path = None
    
    # Try specific filename first (for Kenya)
    if country_iso.upper() == "KEN":
        admin_path = config.ADMIN_BOUNDARIES_DIR / f"{country_iso.lower()}/ken_admbnda_adm1_iebc_20191031.shp"
    
    # Try GADM naming convention - prefer ADM1 over ADM0
    if not admin_path or not admin_path.exists():
        country_dir = config.ADMIN_BOUNDARIES_DIR / country_iso.lower()
        if country_dir.exists():
            # Look for ADM1 files first (states/provinces) - GADM format: gadm41_KEN_1.shp
            adm1_files = list(country_dir.glob("*_1.shp")) + list(country_dir.glob("*adm1*.shp")) + list(country_dir.glob("*adm1*.gpkg"))
            if adm1_files:
                admin_path = adm1_files[0]
            else:
                # Fall back to ADM0 (country level) - GADM format: gadm41_KEN_0.shp
                adm0_files = list(country_dir.glob("*_0.shp")) + list(country_dir.glob("*adm0*.shp")) + list(country_dir.glob("*adm0*.gpkg"))
                if adm0_files:
                    admin_path = adm0_files[0]
    
    # Try any shapefile in the directory
    if not admin_path:
        admin_path = find_data_file(
            config.ADMIN_BOUNDARIES_DIR,
            country_iso,
            extensions=['.shp', '.geojson', '.gpkg']
        )
    
    if not admin_path or not admin_path.exists():
        print(f"Warning: Admin boundaries not found for {country_iso}")
        print(f"  Searched in: {config.ADMIN_BOUNDARIES_DIR / country_iso.lower()}")
        print("Creating sample data structure...")
        gdf_admin = _create_sample_admin_boundaries()
    else:
        print(f"Loading admin boundaries from: {admin_path}")
        gdf_admin = gpd.read_file(admin_path)
        
        # Ensure CRS is set (default to EPSG:4326 if not)
        if gdf_admin.crs is None:
            print("Warning: No CRS found, assuming EPSG:4326")
            gdf_admin.set_crs('EPSG:4326', inplace=True)
    
    # 2. Standardize Features
    # --- Population Density ---
    # Try multiple naming conventions for population data
    pop_path = None
    
    # Try WorldPop naming convention
    for year in [2020, 2019, 2021, 2018]:
        test_path = config.POPULATION_DIR / f"{country_iso.lower()}/{country_iso.lower()}_ppp_{year}_1km.tif"
        if test_path.exists():
            pop_path = test_path
            break
    
    # Try any .tif file in the directory
    if not pop_path:
        pop_dir = config.POPULATION_DIR / country_iso.lower()
        if pop_dir.exists():
            tif_files = list(pop_dir.glob("*.tif"))
            if tif_files:
                pop_path = tif_files[0]
    
    if pop_path and pop_path.exists():
        print(f"Loading population data from: {pop_path}")
        # Ensure admin boundaries and population raster are in same CRS
        with rasterio.open(pop_path) as src:
            pop_crs = src.crs
            if gdf_admin.crs != pop_crs:
                print(f"Reprojecting admin boundaries from {gdf_admin.crs} to {pop_crs}")
                gdf_admin = gdf_admin.to_crs(pop_crs)
        
        gdf_admin['population_density'] = calculate_zonal_stats(pop_path, gdf_admin, 'mean')
    else:
        print(f"Warning: Population data not found in {config.POPULATION_DIR / country_iso.lower()}")
        print("  Expected format: [country]_ppp_[year]_1km.tif (e.g., ken_ppp_2020_1km.tif)")
        gdf_admin['population_density'] = np.random.uniform(50, 500, len(gdf_admin))
    
    # --- Health Facility Density ---
    # Try multiple naming conventions
    infra_path = find_data_file(
        config.INFRASTRUCTURE_DIR,
        country_iso,
        extensions=['.shp', '.geojson', '.gpkg'],
        pattern='health'
    )
    
    if not infra_path:
        # Try any file in infrastructure directory
        infra_path = find_data_file(
            config.INFRASTRUCTURE_DIR,
            country_iso,
            extensions=['.shp', '.geojson', '.gpkg']
        )
    
    if infra_path and infra_path.exists():
        print(f"Loading infrastructure data from: {infra_path}")
        gdf_infra = gpd.read_file(infra_path)
        
        # Ensure same CRS
        if gdf_infra.crs != gdf_admin.crs:
            gdf_infra = gdf_infra.to_crs(gdf_admin.crs)
        
        gdf_admin['health_facility_count'] = count_points_in_polygons(gdf_infra, gdf_admin)
        gdf_admin['health_facility_density'] = (
            gdf_admin['health_facility_count'] / (gdf_admin['population_density'] + 1)
        )
    else:
        print(f"Warning: Infrastructure data not found in {config.INFRASTRUCTURE_DIR / country_iso.lower()}")
        print("  Place health facility shapefiles/geojson files in this directory")
        gdf_admin['health_facility_count'] = np.random.randint(0, 20, len(gdf_admin))
        gdf_admin['health_facility_density'] = (
            gdf_admin['health_facility_count'] / (gdf_admin['population_density'] + 1)
        )
    
    # --- Vulnerability Index ---
    # Calculate composite vulnerability index from:
    # 1. Water Access (33% weight) - Lower access = Higher vulnerability
    # 2. Health Access (33% weight) - Lower access = Higher vulnerability  
    # 3. Internal Displacement (33% weight) - Higher displacement = Higher vulnerability
    
    print("\n=== Calculating Vulnerability Index ===")
    
    # 1. Water Access Component
    water_counts = process_water_data(gdf_admin, country_iso)
    
    # Calculate water points per capita (need population for this)
    if 'population_density' in gdf_admin.columns and gdf_admin['population_density'].sum() > 0:
        # Estimate total population from density (rough approximation)
        # For per-capita, we'll use water points per 1000 people
        # If we don't have exact population, use density as proxy
        gdf_admin['water_points'] = water_counts
        # Normalize water access (0-1, where 1 = most access, 0 = least access)
        if water_counts.max() > 0:
            water_access_normalized = (water_counts - water_counts.min()) / (water_counts.max() - water_counts.min() + 1e-10)
        else:
            water_access_normalized = pd.Series([0.5] * len(gdf_admin))  # Default if no data
    else:
        # Fallback: normalize by count only
        if water_counts.max() > 0:
            water_access_normalized = (water_counts - water_counts.min()) / (water_counts.max() - water_counts.min() + 1e-10)
        else:
            water_access_normalized = pd.Series([0.5] * len(gdf_admin))
    
    # 2. Health Access Component (already calculated above)
    if 'health_facility_density' in gdf_admin.columns:
        health_density = gdf_admin['health_facility_density']
        # Normalize health access (0-1, where 1 = most access)
        if health_density.max() > health_density.min():
            health_access_normalized = (health_density - health_density.min()) / (health_density.max() - health_density.min() + 1e-10)
        else:
            health_access_normalized = pd.Series([0.5] * len(gdf_admin))
    else:
        health_access_normalized = pd.Series([0.5] * len(gdf_admin))
    
    # 3. Displacement Component
    displacement_counts = process_displacement_data(gdf_admin, country_iso)
    gdf_admin['displacement'] = displacement_counts
    
    # Normalize displacement (0-1, where 1 = most displacement)
    if displacement_counts.max() > 0:
        displacement_normalized = (displacement_counts - displacement_counts.min()) / (displacement_counts.max() - displacement_counts.min() + 1e-10)
    else:
        displacement_normalized = pd.Series([0] * len(gdf_admin))
    
    # Calculate composite vulnerability index
    # Vulnerability = (1 - water_access) * 0.33 + (1 - health_access) * 0.33 + displacement * 0.33
    # Higher vulnerability = less access + more displacement
    vulnerability_index = (
        (1 - water_access_normalized) * 0.33 +
        (1 - health_access_normalized) * 0.33 +
        displacement_normalized * 0.33
    )
    
    # Ensure values are between 0 and 1
    vulnerability_index = vulnerability_index.clip(0, 1)
    
    gdf_admin['vulnerability_index'] = vulnerability_index
    
    print(f"Vulnerability index range: {vulnerability_index.min():.3f} - {vulnerability_index.max():.3f}")
    print(f"Vulnerability_index mean: {vulnerability_index.mean():.3f}")
    
    # --- Hydrology Data (River Risk and Soil Saturation) ---
    print("\n=== Loading Hydrology Data ===")
    
    # 1. Load River Risk Data
    river_risk_path = config.VULNERABILITY_DIR / country_iso.lower() / "hydrology" / "river_risk.json"
    river_risk_map = {}
    
    if river_risk_path.exists():
        try:
            with open(river_risk_path, 'r') as f:
                river_risk_map = json.load(f)
            print(f"Loaded river risk data for {len(river_risk_map)} stations")
        except Exception as e:
            print(f"Warning: Error loading river risk data: {e}")
            river_risk_map = {}
    else:
        print(f"Warning: River risk data not found at {river_risk_path}")
        print("  Run: python scripts/download_hydrology.py")
    
    # Map river stations to counties
    # Get admin name column
    admin_name_col = None
    for col in ['NAME_1', 'ADM1_EN', 'NAME', 'County', 'county']:
        if col in gdf_admin.columns:
            admin_name_col = col
            break
    
    if admin_name_col:
        # Create normalized county names for matching
        gdf_admin['county_normalized'] = gdf_admin[admin_name_col].apply(normalize_string)
        
        # River station to county mapping
        river_county_mapping = {
            "Tana_Garissa": ["garissa", "tana river", "tana"],
            "Nzoia_Webuye": ["busia", "siaya", "bungoma", "kakamega"],
            "Athi_Malindi": ["kilifi", "taita taveta", "machakos", "makueni"],
            "Turkwel_Lodwar": ["turkana", "west pokot"]
        }
        
        # Assign river risk scores to counties
        river_risk_scores = pd.Series([0.0] * len(gdf_admin))
        
        for station, risk_score in river_risk_map.items():
            if station in river_county_mapping:
                affected_counties = river_county_mapping[station]
                for idx, county_norm in enumerate(gdf_admin['county_normalized']):
                    if any(aff_county in county_norm for aff_county in affected_counties):
                        # Use maximum if county is affected by multiple rivers
                        river_risk_scores.iloc[idx] = max(river_risk_scores.iloc[idx], risk_score)
        
        gdf_admin['river_risk_score'] = river_risk_scores
        print(f"River risk scores assigned. Range: {river_risk_scores.min():.3f} - {river_risk_scores.max():.3f}")
    else:
        gdf_admin['river_risk_score'] = 0.0
        print("Warning: Could not find admin name column for river mapping")
    
    # 2. Fetch Soil Saturation Data
    print("Fetching soil saturation data...")
    
    # Import the function from download_hydrology script
    try:
        hydrology_script = Path(__file__).resolve().parent.parent / "scripts" / "download_hydrology.py"
        if hydrology_script.exists():
            # Import the function dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("download_hydrology", hydrology_script)
            if spec is None or spec.loader is None:
                raise ImportError("Could not create module spec")
            hydrology_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hydrology_module)
            get_soil_saturation = hydrology_module.get_soil_saturation
        else:
            raise ImportError("Hydrology script not found")
    except Exception as e:
        print(f"Warning: Could not import get_soil_saturation function: {e}")
        print("  Soil saturation will be set to 0. Run: python scripts/download_hydrology.py")
        get_soil_saturation = None
    
    soil_saturation_scores = []
    
    if get_soil_saturation:
        # Calculate centroids for each county
        gdf_admin_4326 = gdf_admin.to_crs('EPSG:4326')  # Ensure WGS84 for lat/lon
        
        for idx, row in gdf_admin_4326.iterrows():
            centroid = row.geometry.centroid
            lat = float(centroid.y)
            lon = float(centroid.x)
            
            try:
                saturation = get_soil_saturation(lat, lon)
                soil_saturation_scores.append(saturation)
                if (idx + 1) % 10 == 0:  # type: ignore
                    print(f"  Processed {idx + 1}/{len(gdf_admin)} counties...")
            except Exception as e:
                county_name = row.get(admin_name_col, str(idx)) if admin_name_col else str(idx)
                print(f"  Warning: Error fetching soil data for {county_name}: {e}")
                soil_saturation_scores.append(0.0)
        
        gdf_admin['soil_saturation'] = pd.Series(soil_saturation_scores)
        print(f"Soil saturation calculated. Range: {gdf_admin['soil_saturation'].min():.3f} - {gdf_admin['soil_saturation'].max():.3f}")
    else:
        gdf_admin['soil_saturation'] = 0.0
        print("  Soil saturation set to 0 (function not available)")
    
    # --- Disaster Magnitude (e.g., Flooded Population) ---
    # Try to find disaster event files (flood, drought, conflict, etc.)
    flood_path = None
    nga_floodplains_path = None
    disaster_dir = config.DISASTER_EVENTS_DIR / country_iso.lower()
    
    if disaster_dir.exists():
        # First, check for NGA floodplains (base risk layer)
        nga_floodplains_path = disaster_dir / "nga_floodplains.geojson"
        if not nga_floodplains_path.exists():
            # Also check for .shp version
            nga_shp = list(disaster_dir.glob("nga_floodplains.shp"))
            if nga_shp:
                nga_floodplains_path = nga_shp[0]
        
        # Prioritize maximum flood extent files (most comprehensive) for event-specific data
        priority_patterns = [
            '*MaximumFloodWaterExtent*',
            '*FloodExtent*',
            '*flood*',
            '*FL*',
            '*Flood*'
        ]
        
        for pattern in priority_patterns:
            flood_files = list(disaster_dir.rglob(f"{pattern}.shp")) + \
                         list(disaster_dir.rglob(f"{pattern}.geojson")) + \
                         list(disaster_dir.rglob(f"{pattern}.gpkg"))
            # Skip NGA floodplains - we handle it separately
            flood_files = [f for f in flood_files if 'nga_floodplains' not in f.name.lower()]
            if flood_files:
                flood_path = flood_files[0]
                print(f"Found flood event data: {flood_path.name}")
                break
        
        # If no flood-specific files, try any shapefile (except NGA)
        if not flood_path:
            all_shp = list(disaster_dir.rglob("*.shp"))
            all_shp = [f for f in all_shp if 'nga_floodplains' not in f.name.lower()]
            if all_shp:
                flood_path = all_shp[0]
                print(f"Using general disaster data: {flood_path.name}")
    
    # Calculate base flood risk from NGA floodplains (low-resolution, static risk)
    base_flood_risk = pd.Series([0.0] * len(gdf_admin))
    if nga_floodplains_path and nga_floodplains_path.exists() and pop_path and pop_path.exists():
        try:
            print(f"Loading NGA floodplains (base risk layer): {nga_floodplains_path.name}")
            gdf_nga = gpd.read_file(nga_floodplains_path)
            
            # Ensure same CRS
            if gdf_nga.crs != gdf_admin.crs:
                gdf_nga = gdf_nga.to_crs(gdf_admin.crs)
            
            # Calculate percentage of each county covered by floodplains
            for idx, admin_row in gdf_admin.iterrows():
                admin_geom = admin_row.geometry
                # Find overlapping floodplain polygons
                overlapping = gdf_nga[gdf_nga.geometry.intersects(admin_geom)]
                if len(overlapping) > 0:
                    # Calculate intersection area
                    intersection = overlapping.geometry.intersection(admin_geom)
                    total_intersection_area = intersection.area.sum()
                    admin_area = admin_geom.area
                    if admin_area > 0:
                        base_flood_risk.iloc[idx] = min(total_intersection_area / admin_area, 1.0)
            
            print(f"Base flood risk calculated. Range: {base_flood_risk.min():.3f} - {base_flood_risk.max():.3f}")
        except Exception as e:
            print(f"Warning: Error processing NGA floodplains: {e}")
    
    # Calculate event-specific flooded population
    if flood_path and flood_path.exists() and pop_path and pop_path.exists():
        gdf_flood = gpd.read_file(flood_path)
        
        # Clip population raster to flood extent
        try:
            with rasterio.open(pop_path) as src:
                flood_geom = [geom for geom in gdf_flood.geometry]
                if flood_geom:
                    out_image, out_transform = mask(src, flood_geom, crop=True)
                    # Get flooded population per county
                    gdf_admin['flooded_population'] = calculate_zonal_stats_from_array(
                        out_image[0], out_transform, gdf_admin, 'sum', nodata=src.nodata
                    )
                else:
                    gdf_admin['flooded_population'] = 0
        except Exception as e:
            print(f"Warning: Error processing flood event data: {e}")
            gdf_admin['flooded_population'] = np.random.uniform(0, 10000, len(gdf_admin))
    else:
        if flood_path:
            print(f"Warning: Disaster event data not found at {flood_path}")
        gdf_admin['flooded_population'] = np.random.uniform(0, 10000, len(gdf_admin))
    
    # Add base flood risk as a separate column (for use in priority calculation)
    gdf_admin['base_flood_risk'] = base_flood_risk
    
    # --- Land Cover Data ---
    print("\n=== Processing Land Cover Data ===")
    land_cover_path = None
    land_cover_dir = config.LAND_COVER_DIR / country_iso.lower()
    
    if land_cover_dir.exists():
        # Look for land cover files
        land_cover_files = list(land_cover_dir.glob("*.geojson")) + \
                          list(land_cover_dir.glob("*.shp")) + \
                          list(land_cover_dir.glob("*.gpkg"))
        if land_cover_files:
            land_cover_path = land_cover_files[0]
            print(f"Found land cover data: {land_cover_path.name}")
    
    # Calculate land cover statistics per county
    if land_cover_path and land_cover_path.exists():
        try:
            gdf_landcover = gpd.read_file(land_cover_path)
            
            # Ensure same CRS
            if gdf_landcover.crs != gdf_admin.crs:
                gdf_landcover = gdf_landcover.to_crs(gdf_admin.crs)
            
            # Initialize land cover columns
            gdf_admin['alluvial_plain_pct'] = 0.0
            gdf_admin['urban_area_pct'] = 0.0
            gdf_admin['agricultural_pct'] = 0.0
            
            # Calculate percentage of each county covered by different land cover types
            for idx, admin_row in gdf_admin.iterrows():
                admin_geom = admin_row.geometry
                admin_area = admin_geom.area
                
                if admin_area == 0:
                    continue
                
                # Find overlapping land cover polygons
                overlapping = gdf_landcover[gdf_landcover.geometry.intersects(admin_geom)]
                
                if len(overlapping) > 0:
                    # Calculate intersection area for each land cover type
                    alluvial_area = 0.0
                    urban_area = 0.0
                    agricultural_area = 0.0
                    
                    for _, lc_row in overlapping.iterrows():
                        intersection = lc_row.geometry.intersection(admin_geom)
                        intersection_area = intersection.area
                        
                        # Classify based on landform description
                        lc_desc = str(lc_row.get('l_form_des', '')).lower()
                        lc_label = str(lc_row.get('label', '')).lower()
                        
                        if 'alluvial' in lc_desc or 'alluvial' in lc_label:
                            alluvial_area += intersection_area
                        if 'urban' in lc_desc or 'urban' in lc_label or 'built' in lc_desc:
                            urban_area += intersection_area
                        if 'agricultural' in lc_desc or 'agricultural' in lc_label or 'crop' in lc_desc or 'farm' in lc_desc:
                            agricultural_area += intersection_area
                    
                    # Calculate percentages
                    gdf_admin.loc[idx, 'alluvial_plain_pct'] = min(alluvial_area / admin_area, 1.0)
                    gdf_admin.loc[idx, 'urban_area_pct'] = min(urban_area / admin_area, 1.0)
                    gdf_admin.loc[idx, 'agricultural_pct'] = min(agricultural_area / admin_area, 1.0)
            
            print(f"Land cover statistics calculated:")
            print(f"  Alluvial plain coverage: {gdf_admin['alluvial_plain_pct'].min():.3f} - {gdf_admin['alluvial_plain_pct'].max():.3f}")
            print(f"  Urban area coverage: {gdf_admin['urban_area_pct'].min():.3f} - {gdf_admin['urban_area_pct'].max():.3f}")
            print(f"  Agricultural coverage: {gdf_admin['agricultural_pct'].min():.3f} - {gdf_admin['agricultural_pct'].max():.3f}")
            
        except Exception as e:
            print(f"Warning: Error processing land cover data: {e}")
            import traceback
            traceback.print_exc()
            # Set defaults
            gdf_admin['alluvial_plain_pct'] = 0.0
            gdf_admin['urban_area_pct'] = 0.0
            gdf_admin['agricultural_pct'] = 0.0
    else:
        print("Land cover data not found - skipping land cover analysis")
        gdf_admin['alluvial_plain_pct'] = 0.0
        gdf_admin['urban_area_pct'] = 0.0
        gdf_admin['agricultural_pct'] = 0.0
    
    # 3. Clean and Save
    gdf_admin = gdf_admin.fillna(0)
    
    # Ensure all standard features exist
    for feature in config.STANDARD_FEATURES:
        if feature not in gdf_admin.columns:
            gdf_admin[feature] = 0
    
    # Create output directory if it doesn't exist
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    gdf_admin.to_file(config.MASTER_DATASET_PATH, driver="GeoJSON")
    
    print(f"Data processed and saved to {config.MASTER_DATASET_PATH}")
    return gdf_admin


def calculate_zonal_stats(raster_path, polygons, stat_type='mean', **kwargs):
    """
    Helper to calculate zonal statistics from a raster.
    
    Args:
        raster_path: Path to raster file
        polygons: GeoDataFrame with polygon geometries
        stat_type: Type of statistic ('mean', 'sum', 'max', 'min', etc.)
        **kwargs: Additional arguments for zonal_stats
    
    Returns:
        Series with statistics for each polygon
    """
    try:
        stats = zonal_stats(
            polygons.geometry,
            str(raster_path),
            stats=stat_type,
            **kwargs
        )
        return pd.Series([s.get(stat_type, 0) if s else 0 for s in stats])
    except Exception as e:
        print(f"Error calculating zonal stats: {e}")
        return pd.Series([0] * len(polygons))


def calculate_zonal_stats_from_array(array, transform, polygons, stat_type='mean', nodata=None):
    """
    Calculate zonal statistics from a numpy array.
    
    Args:
        array: 2D numpy array
        transform: Affine transform for the array
        polygons: GeoDataFrame with polygon geometries
        stat_type: Type of statistic
        nodata: NoData value
    
    Returns:
        Series with statistics for each polygon
    """
    try:
        stats = zonal_stats(
            polygons.geometry,
            array,
            affine=transform,
            stats=stat_type,
            nodata=nodata
        )
        return pd.Series([s.get(stat_type, 0) if s else 0 for s in stats])
    except Exception as e:
        print(f"Error calculating zonal stats from array: {e}")
        return pd.Series([0] * len(polygons))


def count_points_in_polygons(points, polygons):
    """
    Helper to count points in each polygon.
    
    Args:
        points: GeoDataFrame with point geometries
        polygons: GeoDataFrame with polygon geometries
    
    Returns:
        Series with count of points in each polygon
    """
    try:
        # Ensure CRS matches
        if points.crs != polygons.crs:
            points = points.to_crs(polygons.crs)
        
        # Spatial join
        joined = gpd.sjoin(points, polygons, how='inner', predicate='within')
        
        # Count points per polygon
        counts = joined.groupby('index_right').size()
        
        # Create full series with zeros for polygons with no points
        result = pd.Series(0, index=range(len(polygons)))
        result.loc[counts.index] = counts.values
        
        return result
    except Exception as e:
        print(f"Error counting points in polygons: {e}")
        return pd.Series([0] * len(polygons))


def normalize_string(s):
    """
    Normalize string for matching (lowercase, strip, remove special chars).
    
    Args:
        s: String to normalize
    
    Returns:
        Normalized string
    """
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    # Remove common special characters that cause mismatches
    s = s.replace("'", "").replace("-", " ").replace("_", " ")
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def process_water_data(gdf_admin, country_iso="KEN"):
    """
    Process water point data to calculate water access per admin region.
    
    Args:
        gdf_admin: GeoDataFrame with admin boundaries
        country_iso: Country ISO code
    
    Returns:
        Series with functional water point counts per admin region
    """
    water_path = config.VULNERABILITY_DIR / country_iso.lower() / "wpdx_enhanced.csv"
    
    if not water_path.exists():
        print(f"Warning: Water data not found at {water_path}")
        return pd.Series([0] * len(gdf_admin))
    
    try:
        print(f"Loading water data from: {water_path}")
        # Skip first row if it contains metadata headers
        df_water = pd.read_csv(water_path, skiprows=1, low_memory=False)
        
        # Check for required columns
        lat_col = None
        lon_col = None
        status_col = None
        
        # Try to find latitude/longitude columns
        # Handle both standard names (lat_deg, lon_deg) and HDX format (#geo+lat, #geo+lon)
        for col in df_water.columns:
            col_lower = col.lower()
            # Check for lat column
            if lat_col is None:
                if ('lat' in col_lower and ('deg' in col_lower or 'dd' in col_lower)) or \
                   ('#geo+lat' in col_lower or col == '#geo+lat'):
                    lat_col = col
            # Check for lon column
            if lon_col is None:
                if ('lon' in col_lower and ('deg' in col_lower or 'dd' in col_lower)) or \
                   ('#geo+lon' in col_lower or col == '#geo+lon'):
                    lon_col = col
            # Check for status column
            if status_col is None:
                if ('status' in col_lower and 'clean' in col_lower) or \
                   ('water_avail' in col_lower and 'detail' in col_lower) or \
                   col == '#indicator+water_avail_detail':
                    status_col = col
        
        if lat_col is None or lon_col is None:
            print(f"Warning: Could not find lat/lon columns in water data. Available columns: {list(df_water.columns)}")
            return pd.Series([0] * len(gdf_admin))
        
        # Convert coordinates to numeric, removing any non-numeric values
        df_water[lat_col] = pd.to_numeric(df_water[lat_col], errors='coerce')
        df_water[lon_col] = pd.to_numeric(df_water[lon_col], errors='coerce')
        
        # Remove rows with missing coordinates
        df_water = df_water.dropna(subset=[lat_col, lon_col])
        
        # Filter for functional water points
        if status_col and status_col in df_water.columns:
            # Try to identify functional status - check common values
            functional_keywords = ['yes', 'functional', 'working', 'operational', 'active', 'ok']
            df_water = df_water[
                df_water[status_col].astype(str).str.lower().str.contains('|'.join(functional_keywords), na=False)
            ]
        # If no status column or filtering removed all rows, use all points (existence = better than nothing)
        
        if len(df_water) == 0:
            print("Warning: No valid water points found after filtering")
            return pd.Series([0] * len(gdf_admin))
        
        # Create GeoDataFrame from water points
        gdf_water = gpd.GeoDataFrame(
            df_water,
            geometry=gpd.points_from_xy(df_water[lon_col], df_water[lat_col]),
            crs='EPSG:4326'
        )
        
        # Ensure same CRS as admin boundaries
        if gdf_admin.crs != gdf_water.crs:
            gdf_water = gdf_water.to_crs(gdf_admin.crs)
        
        # Count functional water points per admin region
        water_counts = count_points_in_polygons(gdf_water, gdf_admin)
        
        print(f"Found {len(gdf_water)} functional water points")
        return water_counts
        
    except Exception as e:
        print(f"Error processing water data: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series([0] * len(gdf_admin))


def process_displacement_data(gdf_admin, country_iso="KEN"):
    """
    Process displacement data to get displacement counts per admin region.
    
    Args:
        gdf_admin: GeoDataFrame with admin boundaries
        country_iso: Country ISO code
    
    Returns:
        Series with displacement counts per admin region
    """
    displacement_path = config.VULNERABILITY_DIR / country_iso.lower() / "event_data_ken.csv"
    
    if not displacement_path.exists():
        print(f"Warning: Displacement data not found at {displacement_path}")
        return pd.Series([0] * len(gdf_admin))
    
    try:
        print(f"Loading displacement data from: {displacement_path}")
        df_displacement = pd.read_csv(displacement_path)
        
        # Find the admin/area column
        admin_col = None
        displacement_col = None
        
        for col in df_displacement.columns:
            col_lower = col.lower()
            if 'admin' in col_lower or 'area' in col_lower or 'location' in col_lower or 'county' in col_lower:
                admin_col = col
            elif 'displacement' in col_lower or 'displaced' in col_lower:
                displacement_col = col
        
        if admin_col is None:
            print(f"Warning: Could not find admin/area column. Available columns: {list(df_displacement.columns)}")
            return pd.Series([0] * len(gdf_admin))
        
        if displacement_col is None:
            print(f"Warning: Could not find displacement column. Available columns: {list(df_displacement.columns)}")
            return pd.Series([0] * len(gdf_admin))
        
        # Group by admin area and sum displacement
        df_displacement[displacement_col] = pd.to_numeric(df_displacement[displacement_col], errors='coerce').fillna(0)
        displacement_by_area = df_displacement.groupby(admin_col)[displacement_col].sum().reset_index()
        displacement_by_area.columns = ['admin_name', 'displacement']
        
        # Normalize admin names for matching
        displacement_by_area['admin_name_normalized'] = displacement_by_area['admin_name'].apply(normalize_string)
        
        # Get admin names from gdf_admin (try multiple possible column names)
        admin_name_col = None
        for col in ['NAME_1', 'ADM1_EN', 'NAME', 'County', 'county']:
            if col in gdf_admin.columns:
                admin_name_col = col
                break
        
        if admin_name_col is None:
            print(f"Warning: Could not find admin name column in boundaries. Available: {list(gdf_admin.columns)}")
            return pd.Series([0] * len(gdf_admin))
        
        # Normalize admin boundary names
        gdf_admin_normalized = gdf_admin.copy()
        gdf_admin_normalized['admin_name_normalized'] = gdf_admin_normalized[admin_name_col].apply(normalize_string)
        
        # Merge displacement data with admin boundaries
        merged = gdf_admin_normalized.merge(
            displacement_by_area,
            on='admin_name_normalized',
            how='left'
        )
        
        displacement_counts = merged['displacement'].fillna(0)
        
        print(f"Found displacement data for {len(displacement_by_area)} areas")
        return displacement_counts
        
    except Exception as e:
        print(f"Error processing displacement data: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series([0] * len(gdf_admin))


def _create_sample_admin_boundaries():
    """
    Create a sample GeoDataFrame with admin boundaries for testing.
    This is used when actual data files are not available.
    """
    from shapely.geometry import Polygon
    
    # Create sample polygons (simplified Kenya counties)
    sample_polygons = [
        Polygon([(36.5, -1.0), (37.0, -1.0), (37.0, -0.5), (36.5, -0.5), (36.5, -1.0)]),
        Polygon([(37.0, -1.0), (37.5, -1.0), (37.5, -0.5), (37.0, -0.5), (37.0, -1.0)]),
        Polygon([(36.5, -0.5), (37.0, -0.5), (37.0, 0.0), (36.5, 0.0), (36.5, -0.5)]),
    ]
    
    gdf = gpd.GeoDataFrame({
        'ADM1_EN': ['Nairobi', 'Kiambu', 'Machakos'],
        'geometry': sample_polygons
    }, crs='EPSG:4326')
    
    return gdf


if __name__ == "__main__":
    print("Processing data...")
    result = load_and_process_data(country_iso="KEN")
    print(f"Data processed: {len(result)} regions")
    print(f"Features: {list(result.columns)}")

