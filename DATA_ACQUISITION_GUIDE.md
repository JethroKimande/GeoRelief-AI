# GeoRelief-AI Data Acquisition Guide

This guide will help you download and integrate real geospatial data for GeoRelief-AI.

## 📋 Required Data Types

1. **Administrative Boundaries** - Country/region boundaries (Shapefile/GeoJSON)
2. **Population Data** - Population density rasters (GeoTIFF)
3. **Infrastructure Data** - Health facilities, roads, etc. (Shapefile/GeoJSON)
4. **Disaster Event Data** - Floods, conflicts, droughts (Shapefile/GeoJSON)

## 🌍 Data Sources

### 1. Administrative Boundaries

#### GADM (Global Administrative Areas)
- **URL**: https://gadm.org/download_country.html
- **Format**: Shapefile or GeoPackage
- **Steps**:
  1. Select your country
  2. Choose administrative level (ADM1 = states/provinces, ADM2 = counties/districts)
  3. Download and extract to `raw_data/admin_boundaries/[country_code]/`

#### HDX (Humanitarian Data Exchange)
- **URL**: https://data.humdata.org/
- **Search**: "[Country] administrative boundaries"
- **Example for Kenya**: https://data.humdata.org/dataset/kenya-administrative-boundaries

### 2. Population Data

#### WorldPop
- **URL**: https://www.worldpop.org/geodata/listing?id=75
- **Format**: GeoTIFF (1km resolution)
- **Steps**:
  1. Select country and year
  2. Download "Population Count" or "Population Density"
  3. Save to `raw_data/population/[country_code]/[country_code]_ppp_[year]_1km.tif`

#### Alternative: LandScan
- **URL**: https://landscan.ornl.gov/
- **Note**: Requires registration

### 3. Infrastructure Data

#### HDX (Health Facilities)
- **URL**: https://data.humdata.org/
- **Search**: "[Country] health facilities"
- **Example**: https://data.humdata.org/dataset/health-facilities-kenya
- **Save to**: `raw_data/infrastructure/[country_code]/`

#### OpenStreetMap (OSM)
- **URL**: https://www.openstreetmap.org/
- **Tools**: 
  - Overpass Turbo: https://overpass-turbo.eu/
  - Geofabrik: https://download.geofabrik.de/
- **Query Example** (for health facilities):
  ```
  [out:json][timeout:25];
  (
    node["amenity"="hospital"]({{bbox}});
    node["amenity"="clinic"]({{bbox}});
    way["amenity"="hospital"]({{bbox}});
    way["amenity"="clinic"]({{bbox}});
  );
  out body;
  >;
  out skel qt;
  ```

### 4. Disaster Event Data

#### HDX
- **URL**: https://data.humdata.org/
- **Search**: "[Country] flood" or "[Country] disaster"
- **Examples**:
  - Floods: https://data.humdata.org/dataset/floods-kenya
  - Conflicts: https://data.humdata.org/dataset/conflict-data

#### NASA MODIS Flood Mapping
- **URL**: https://modis.gsfc.nasa.gov/data/dataprod/mod44w.php
- **Format**: GeoTIFF

#### Copernicus Emergency Management Service
- **URL**: https://emergency.copernicus.eu/
- **Format**: Shapefile/GeoTIFF

#### ReliefWeb
- **URL**: https://reliefweb.int/
- **Note**: Often provides links to disaster datasets

## 📁 File Organization

After downloading, organize files as follows:

```
raw_data/
├── admin_boundaries/
│   └── ken/                          # Country code (lowercase)
│       ├── ken_admbnda_adm1.shp      # Admin level 1 (states/provinces)
│       ├── ken_admbnda_adm1.shx
│       ├── ken_admbnda_adm1.dbf
│       └── ken_admbnda_adm1.prj
├── population/
│   └── ken/
│       └── ken_ppp_2020_1km.tif      # WorldPop format
├── infrastructure/
│   └── ken/
│       ├── health_facilities.shp     # Health facilities
│       └── roads.shp                 # Optional: roads
└── disaster_events/
    └── ken/
        ├── flood_2023.shp            # Flood events
        └── drought_2023.shp          # Drought events
```

## 🔧 Using the Download Script

We've provided a helper script to automate some downloads:

```bash
python scripts/download_data.py
```

This script will:
- Download Kenya admin boundaries from HDX
- Download WorldPop population data
- Attempt to download health facilities
- Provide manual download links for disaster data

## 🔄 Updating the Data Processor

After downloading data, you may need to update `core/data_processor.py` to match your file names:

### Example: Different Admin Boundary File

If your admin boundaries file is named differently:

```python
# In core/data_processor.py, line ~30
admin_path = config.ADMIN_BOUNDARIES_DIR / f"{country_iso.lower()}/your_file_name.shp"
```

### Example: Multiple Population Files

To support multiple years:

```python
# In core/data_processor.py
year = 2020  # or make this a parameter
pop_path = config.POPULATION_DIR / f"{country_iso.lower()}/{country_iso.lower()}_ppp_{year}_1km.tif"
```

### Example: Auto-Discover Files

You can also make the processor auto-discover files:

```python
import glob

# Auto-find admin boundaries
admin_files = list(config.ADMIN_BOUNDARIES_DIR.glob(f"{country_iso.lower()}/**/*.shp"))
if admin_files:
    admin_path = admin_files[0]
```

## ✅ Verification Checklist

Before processing, verify:

- [ ] Admin boundaries file exists and has geometry column
- [ ] Population raster is in correct CRS (usually EPSG:4326 or UTM)
- [ ] Infrastructure data has point/polygon geometries
- [ ] Disaster event data covers the area of interest
- [ ] All files are in the correct CRS (reproject if needed)

## 🚀 Processing Your Data

Once files are in place:

1. **Process the data**:
   ```bash
   python -m core.data_processor
   ```

2. **Check the output**:
   ```bash
   # View the processed dataset
   python -c "import geopandas as gpd; from core import config; gdf = gpd.read_file(config.MASTER_DATASET_PATH); print(gdf.head())"
   ```

3. **Train the model**:
   ```bash
   python scripts/1_train_model.py
   ```

4. **Run the app**:
   ```bash
   python app/app.py
   ```

## 🌐 Country-Specific Examples

### Kenya
- **Admin**: GADM or HDX Kenya Admin Boundaries
- **Population**: WorldPop KEN_ppp_2020_1km_Aggregated.tif
- **Health**: HDX Kenya Health Facilities
- **Disasters**: HDX Kenya Floods, ReliefWeb

### Nigeria
- **Admin**: GADM NGA_adm1.shp
- **Population**: WorldPop NGA_ppp_2020_1km_Aggregated.tif
- **Health**: HDX Nigeria Health Facilities
- **Disasters**: HDX Nigeria Floods

### Bangladesh
- **Admin**: GADM BGD_adm1.shp
- **Population**: WorldPop BGD_ppp_2020_1km_Aggregated.tif
- **Health**: HDX Bangladesh Health Facilities
- **Disasters**: HDX Bangladesh Floods

## 📝 Notes

- **Coordinate Reference Systems (CRS)**: Ensure all data uses the same CRS, or the processor will handle reprojection
- **File Formats**: Supported formats include Shapefile (.shp), GeoJSON (.geojson), GeoTIFF (.tif)
- **File Sizes**: Population rasters can be large (100MB+). Ensure sufficient disk space
- **Licensing**: Most data sources are open, but check individual licenses

## 🆘 Troubleshooting

### "File not found" errors
- Check file paths in `core/data_processor.py`
- Verify files are in the correct directory structure
- Check file extensions (case-sensitive on Linux/Mac)

### CRS mismatch errors
- Reproject data to a common CRS (e.g., EPSG:4326)
- Use `gdf.to_crs('EPSG:4326')` in GeoPandas

### Memory errors with large rasters
- Use rasterio's windowed reading
- Consider resampling to lower resolution
- Process in chunks

## 📚 Additional Resources

- **GeoPandas Documentation**: https://geopandas.org/
- **Rasterio Documentation**: https://rasterio.readthedocs.io/
- **WorldPop API**: https://www.worldpop.org/geodata/summary?id=75
- **HDX API**: https://data.humdata.org/api

