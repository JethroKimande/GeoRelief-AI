"""
Download hydrology data for GeoRelief-AI
Fetches river discharge and soil saturation data from Open-Meteo API
"""

import openmeteo_requests  # type: ignore
import requests_cache  # type: ignore
from retry_requests import retry  # type: ignore
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # type: ignore
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)  # type: ignore
openmeteo = openmeteo_requests.Client(session=retry_session)  # type: ignore

# Add parent directory to path to import config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config

OUTPUT_DIR = config.VULNERABILITY_DIR / "ken" / "hydrology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Define Key River Monitor Points (Sentinel Stations)
# These coords are for the "mouths" or key flow points of basins
STATIONS = {
    "Tana_Garissa": {"lat": -0.456, "lon": 39.658},  # Tana River
    "Nzoia_Webuye": {"lat": 0.616, "lon": 34.763},   # Nzoia River
    "Athi_Malindi": {"lat": -3.220, "lon": 40.117},  # Athi-Galana-Sabaki
    "Turkwel_Lodwar": {"lat": 3.116, "lon": 35.600}  # Turkwel River
}


def get_river_data() -> Dict[str, float]:
    """Fetch river discharge data from GloFAS via Open-Meteo (10 years historical + 7 day forecast)"""
    print("Fetching River Discharge Data (GloFAS) - 10 years historical...")
    
    # Prepare lists for API call
    lats = [v["lat"] for v in STATIONS.values()]
    lons = [v["lon"] for v in STATIONS.values()]
    names = list(STATIONS.keys())

    # Use flood API - try to get maximum historical data
    # Note: The API may limit past_days, so we'll use 3650 days (10 years) and let it use the max available
    url = "https://flood-api.open-meteo.com/v1/flood"
    
    params = {
        "latitude": lats,
        "longitude": lons,
        "daily": "river_discharge_mean",
        "past_days": 3650,  # Request 10 years (API will use max available if less)
        "forecast_days": 7  # Look 1 week ahead
    }

    try:
        print("   Fetching historical and forecast data...")
        responses = openmeteo.weather_api(url, params=params)  # type: ignore
        
        river_risk_map: Dict[str, float] = {}

        for i, response in enumerate(responses):  # type: ignore
            station_name = names[i]
            
            # Get all data (historical + forecast)
            daily = response.Daily()  # type: ignore
            discharge_all = daily.Variables(0).ValuesAsNumpy()  # type: ignore
            
            # Filter out NaN values
            discharge_clean = discharge_all[~np.isnan(discharge_all)]  # type: ignore
            
            if len(discharge_clean) == 0:  # type: ignore
                print(f"   - {station_name}: No valid data (all NaN)")
                river_risk_map[station_name] = 0.0
                continue
            
            # Use most recent data point (last 7 days average for current flow)
            recent_data = discharge_clean[-7:] if len(discharge_clean) >= 7 else discharge_clean  # type: ignore
            current_flow = np.mean(recent_data) if len(recent_data) > 0 else discharge_clean[-1] if len(discharge_clean) > 0 else 0.0  # type: ignore
            peak_flow = max(discharge_clean) if len(discharge_clean) > 0 else 1.0  # type: ignore
            
            # Calculate number of days of data available
            days_available = len(discharge_clean)  # type: ignore
            
            # Normalize risk 0-1 based on historical maximum (up to 10 years)
            risk_score = current_flow / (peak_flow + 0.01) if peak_flow > 0 else 0.0  # Avoid div by zero  # type: ignore
            river_risk_map[station_name] = float(risk_score)  # type: ignore
            
            print(f"   - {station_name}: {current_flow:.2f} m3/s (peak: {peak_flow:.2f}, {days_available} days, Risk: {risk_score:.2f})")

        # Save to JSON for the Processor to read
        output_path = OUTPUT_DIR / "river_risk.json"
        with open(output_path, "w") as f:
            json.dump(river_risk_map, f, indent=2)
        
        print(f"[OK] River risk data saved to {output_path}")
        return river_risk_map
        
    except Exception as e:
        print(f"[ERROR] Error fetching river data: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_soil_saturation(lat: float, lon: float, start_date: Optional[str] = None, end_date: Optional[str] = None) -> float:
    """
    Fetch historical soil moisture and calculate saturation index.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date for historical data (default: 10 years ago)
        end_date: End date (default: yesterday, dynamically calculated)
    
    Returns:
        Saturation index (0-1) where 1 = maximum historical moisture
    """
    if end_date is None:
        # Default to yesterday
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    if start_date is None:
        # Default to 10 years ago
        start_date = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")
    
    # Use ERA5-Land API endpoint with correct parameter name
    # Try different parameter name formats for soil moisture
    url = "https://archive-api.open-meteo.com/v1/era5"
    
    # Try multiple parameter name variations
    param_variations = [
        "soil_moisture_0_7cm",  # Without "to"
        "volumetric_soil_water_layer_1",  # Alternative name
        "soil_moisture_0_to_7cm"  # Original
    ]
    
    for param_name in param_variations:
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": param_name
            }
            
            responses = openmeteo.weather_api(url, params=params)  # type: ignore
            hourly = responses[0].Hourly()  # type: ignore
            moisture = hourly.Variables(0).ValuesAsNumpy()  # type: ignore
            
            # Check if we got valid data (not all NaN)
            if len(moisture) > 0 and not np.all(np.isnan(moisture)):  # type: ignore
                # Get daily averages from hourly data
                hours_per_day = 24
                n_days = len(moisture) // hours_per_day  # type: ignore
                if n_days == 0:
                    continue
                
                # Reshape to (n_days, 24) and take mean along hours axis
                moisture_reshaped = moisture[:n_days * hours_per_day].reshape(n_days, hours_per_day)  # type: ignore
                daily_moisture = np.mean(moisture_reshaped, axis=1)  # type: ignore
                
                # Filter out NaN values
                daily_moisture_clean = daily_moisture[~np.isnan(daily_moisture)]  # type: ignore
                if len(daily_moisture_clean) == 0:  # type: ignore
                    continue
                
                current_moisture = daily_moisture_clean[-1]  # Most recent day
                max_moisture = max(daily_moisture_clean)     # "Saturation Point" proxy  # type: ignore
                
                if max_moisture == 0:
                    return 0.0
                
                saturation_index = float(current_moisture / max_moisture)  # type: ignore
                return saturation_index
        except Exception:
            # Try next parameter variation
            continue
    
    # If all parameter variations failed, return 0.0
    return 0.0


if __name__ == "__main__":
    print("=" * 60)
    print("GeoRelief-AI Hydrology Data Download")
    print("=" * 60)
    print()
    
    river_data = get_river_data()
    
    print()
    print("[OK] Hydrology data update complete.")
    print(f"River stations processed: {len(river_data)}")
    print(f"Output directory: {OUTPUT_DIR}")
