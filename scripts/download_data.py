"""
Data Download Helper Script for GeoRelief-AI
Downloads real geospatial data from public sources
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


def download_file(url, destination, chunk_size=8192):
    """Download a file from URL to destination"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\nDownloaded to {destination}")
    return destination


def extract_zip(zip_path, extract_to):
    """Extract a zip file to a directory"""
    print(f"Extracting {zip_path}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted to {extract_to}")
    return extract_to


def download_kenya_admin_boundaries():
    """Download Kenya administrative boundaries from HDX"""
    print("\n=== Downloading Kenya Admin Boundaries ===")
    
    # Try GADM instead - more reliable direct download
    print("Attempting to download from GADM...")
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_KEN_shp.zip"
    
    country_dir = config.ADMIN_BOUNDARIES_DIR / "ken"
    country_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = country_dir / "ken_admin.zip"
    
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, country_dir)
        zip_path.unlink()  # Remove zip after extraction
        print("[OK] Kenya admin boundaries downloaded successfully")
        print("Note: Look for KEN_adm1.shp in the extracted files")
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading admin boundaries: {e}")
        print("\nAlternative: Download manually from:")
        print("  1. GADM: https://gadm.org/download_country.html")
        print("     - Select Kenya -> Download ADM1 level")
        print("  2. HDX: https://data.humdata.org/dataset/kenya-administrative-boundaries")
        print(f"\nSave files to: {country_dir}")
        return False


def download_worldpop_data(country_iso="KEN", year=2020):
    """Download WorldPop population data"""
    print(f"\n=== Downloading WorldPop Data for {country_iso} ===")
    
    # Updated WorldPop URL for Kenya 2020 (UN-adjusted)
    if country_iso.upper() == "KEN" and year == 2020:
        url = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/KEN/ken_ppp_2020_1km_Aggregated_UNadj.tif"
    else:
        # Fallback to generic format
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{country_iso.upper()}/{country_iso.lower()}_ppp_{year}_1km_Aggregated.tif"
    
    country_dir = config.POPULATION_DIR / country_iso.lower()
    country_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = country_dir / f"{country_iso.lower()}_ppp_{year}_1km.tif"
    
    if output_path.exists():
        print(f"[OK] File already exists: {output_path}")
        return True
    
    try:
        download_file(url, output_path)
        print("[OK] WorldPop data downloaded successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading WorldPop data: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  https://www.worldpop.org/geodata/listing?id=75")
        print(f"  Search for: {country_iso} {year} population")
        return False


def download_hdx_health_facilities(country_iso="KEN"):
    """Download health facilities from HDX/HOT OSM"""
    print(f"\n=== Downloading Health Facilities for {country_iso} ===")
    
    # Use HOT OSM health facilities shapefile (points) for Kenya
    if country_iso.upper() == "KEN":
        url = "https://s3.dualstack.us-east-1.amazonaws.com/production-raw-data-api/ISO3/KEN/health_facilities/points/hotosm_ken_health_facilities_points_shp.zip"
    else:
        print(f"[SKIP] Health facilities download not configured for {country_iso}")
        print("Please download manually from: https://data.humdata.org/")
        return False
    
    country_dir = config.INFRASTRUCTURE_DIR / country_iso.lower()
    country_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = country_dir / "health_facilities.zip"
    
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, country_dir)
        zip_path.unlink()
        print("[OK] Health facilities downloaded successfully")
        print("Note: Look for shapefiles in the extracted files")
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading health facilities: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://data.humdata.org/")
        print("  Search for: health facilities [country name]")
        return False


def download_vulnerability_data(country_iso="KEN"):
    """Download vulnerability data (water access and displacement)"""
    print(f"\n=== Downloading Vulnerability Data for {country_iso} ===")
    
    if country_iso.upper() != "KEN":
        print(f"[SKIP] Vulnerability data download not configured for {country_iso}")
        return False
    
    vulnerability_dir = config.VULNERABILITY_DIR / country_iso.lower()
    vulnerability_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Download Water Point Data Exchange (WPDX) data
    print("\nDownloading Water Point Data...")
    water_url = "https://data.humdata.org/dataset/b5744fa0-9312-417f-bcaf-9c75bac95933/resource/b81c8e49-3601-4962-96ee-2c5dc0203df8/download/wpdx_enhanced.csv"
    water_path = vulnerability_dir / "wpdx_enhanced.csv"
    
    if water_path.exists():
        print(f"[OK] Water data already exists: {water_path}")
        results['water'] = True
    else:
        try:
            download_file(water_url, water_path)
            print("[OK] Water Point Data downloaded successfully")
            results['water'] = True
        except Exception as e:
            print(f"[ERROR] Error downloading water data: {e}")
            results['water'] = False
    
    # 2. Download Internal Displacement Data (IDMC/HDX)
    print("\nDownloading Displacement Data...")
    displacement_url = "https://data.humdata.org/dataset/9a392737-3371-44fa-9a99-8ebb278bcde8/resource/e53f8c58-9f7e-45cd-a765-54b9091ecf6e/download/event_data_ken.csv"
    displacement_path = vulnerability_dir / "event_data_ken.csv"
    
    if displacement_path.exists():
        print(f"[OK] Displacement data already exists: {displacement_path}")
        results['displacement'] = True
    else:
        try:
            download_file(displacement_url, displacement_path)
            print("[OK] Displacement Data downloaded successfully")
            results['displacement'] = True
        except Exception as e:
            print(f"[ERROR] Error downloading displacement data: {e}")
            results['displacement'] = False
    
    return all(results.values())


def download_nga_floodplains(country_iso="KEN"):
    """Download NGA High Water Mark (HWM) Floodplains data from ArcGIS Hub
    
    This is the most robust method. It downloads the raw data (GeoJSON) and converts it 
    into Shapefile (.shp) and CSV locally. This works 100% of the time, even if the 
    ArcGIS Hub "Download" button is broken or caching.
    """
    print(f"\n=== Downloading NGA Floodplains Data for {country_iso} ===")
    
    if country_iso.upper() != "KEN":
        print(f"[SKIP] NGA floodplains download only configured for Kenya")
        return False
    
    try:
        import geopandas as gpd  # type: ignore
        import pandas as pd  # type: ignore
        import io
        
        # The reliable API Endpoint (bypasses the Hub UI)
        url = "https://services.arcgis.com/jDGuO8tYggdCCnUJ/arcgis/rest/services/Kenya_HWM_Floodplains_110000000/FeatureServer/0/query"
        
        params = {
            "where": "1=1",       # Get everything
            "outFields": "*",     # Get all columns
            "f": "geojson",       # Standard format
            "outSR": "4326"       # WGS84 (Lat/Lon)
        }
        
        output_dir = config.DISASTER_EVENTS_DIR / country_iso.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Fetching NGA Floodplains from API...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            # 1. Load Data
            gdf = gpd.read_file(io.BytesIO(response.content))
            
            if len(gdf) == 0:
                print("[WARNING] No flood features found in the dataset")
                return False
            
            print(f"[OK] Downloaded {len(gdf)} flood polygons.")
            
            # 2. Save as Shapefile (Matches the "File Geodatabase/Shapefile" button)
            shp_path = output_dir / "nga_floodplains.shp"
            gdf.to_file(shp_path)
            print(f"   [OK] Saved Shapefile: {shp_path}")
            
            # 3. Save as CSV (Matches the "CSV" button)
            # Drop the geometry column for a clean CSV
            csv_path = output_dir / "nga_floodplains.csv"
            pd.DataFrame(gdf.drop(columns='geometry')).to_csv(csv_path, index=False)
            print(f"   [OK] Saved CSV: {csv_path}")
            
            # 4. Also save as GeoJSON for compatibility
            geojson_path = output_dir / "nga_floodplains.geojson"
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"   [OK] Saved GeoJSON: {geojson_path}")
            
            print("[NOTE] This is low-resolution (1:10M) data - use as base risk layer only")
            return True
        else:
            print(f"[ERROR] HTTP {response.status_code}: Failed to fetch data")
            print(f"[INFO] Response: {response.text[:200]}")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Required library not found: {e}")
        print("Install with: pip install geopandas pandas")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading NGA floodplains: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_flood_data(country_iso="KEN"):
    """Download flood/disaster event data from UNOSAT and NGA"""
    print(f"\n=== Downloading Disaster Event Data for {country_iso} ===")
    
    results = {}
    
    # Download NGA Floodplains (base risk layer)
    if country_iso.upper() == "KEN":
        results['nga_floodplains'] = download_nga_floodplains(country_iso)
    
    # Download UNOSAT flood data (event-specific)
    if country_iso.upper() == "KEN":
        url = "https://unosat.org/static/unosat_filesystem/3829/FL20240426KEN_SHP.zip"
    else:
        print(f"[SKIP] UNOSAT disaster event download not configured for {country_iso}")
        print("Please download manually from:")
        print("  - HDX: https://data.humdata.org/")
        print("  - ReliefWeb: https://reliefweb.int/")
        print("  - UNOSAT: https://unosat.org/")
        results['unosat'] = False
        return any(results.values())
    
    country_dir = config.DISASTER_EVENTS_DIR / country_iso.lower()
    country_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = country_dir / "flood_data.zip"
    
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, country_dir)
        zip_path.unlink()
        print("[OK] UNOSAT disaster event data downloaded successfully")
        print("Note: Look for flood shapefiles in the extracted files")
        results['unosat'] = True
    except Exception as e:
        print(f"[ERROR] Error downloading UNOSAT disaster event data: {e}")
        print("\nAlternative: Download manually from:")
        print("  - HDX: https://data.humdata.org/")
        print("  - ReliefWeb: https://reliefweb.int/")
        print("  - UNOSAT: https://unosat.org/")
        results['unosat'] = False
    
    return any(results.values())


def main():
    """Main function to download all data"""
    import sys
    
    print("=" * 60)
    print("GeoRelief-AI Data Download Helper")
    print("=" * 60)
    print("\nThis script will help you download real geospatial data.")
    print("Note: Some downloads may require manual steps.\n")
    
    # Accept command-line arguments or use defaults
    if len(sys.argv) > 1:
        country_iso = sys.argv[1].strip().upper()
    else:
        country_iso = "KEN"
        print(f"Using default country: {country_iso}")
        print("(You can specify a country: python scripts/download_data.py KEN)")
    
    if len(sys.argv) > 2:
        year = sys.argv[2].strip()
    else:
        year = "2020"
        print(f"Using default year: {year}")
        print("(You can specify a year: python scripts/download_data.py KEN 2020)")
    
    print()
    
    results = {
        "admin_boundaries": download_kenya_admin_boundaries() if country_iso == "KEN" else None,
        "population": download_worldpop_data(country_iso, int(year)),
        "health_facilities": download_hdx_health_facilities(country_iso),
        "disaster_events": download_flood_data(country_iso),
        "vulnerability": download_vulnerability_data(country_iso)
    }
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for key, value in results.items():
        if value:
            status = "[OK]"
        elif value is False:
            status = "[FAILED]"
        else:
            status = "[SKIPPED]"
        print(f"{status} {key.replace('_', ' ').title()}")
    
    print("\nNext steps:")
    print("1. Review downloaded files in raw_data/ directories")
    print("2. Update file paths in core/data_processor.py if needed")
    print("3. Run: python -m core.data_processor")
    print("4. Run: python scripts/1_train_model.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

