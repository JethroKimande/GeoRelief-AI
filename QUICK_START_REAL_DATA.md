# Quick Start: Using Real Data

This is a quick reference guide for getting real data into GeoRelief-AI.

## 🚀 Fast Track (5 minutes)

### Option 1: Use the Download Script

```bash
# Install requests if not already installed
pip install requests

# Run the download helper
python scripts/download_data.py
```

Follow the prompts to download data for your country.

### Option 2: Manual Download

1. **Admin Boundaries** (Required)
   - Go to: https://gadm.org/download_country.html
   - Select your country → Download ADM1 level
   - Extract to: `raw_data/admin_boundaries/[country_code]/`

2. **Population Data** (Required)
   - Go to: https://www.worldpop.org/geodata/listing?id=75
   - Select country and year (2020 recommended)
   - Download "Population Count" GeoTIFF
   - Save to: `raw_data/population/[country_code]/[country_code]_ppp_2020_1km.tif`

3. **Health Facilities** (Optional but recommended)
   - Go to: https://data.humdata.org/
   - Search: "[Country] health facilities"
   - Download shapefile/geojson
   - Save to: `raw_data/infrastructure/[country_code]/`

4. **Disaster Events** (Optional)
   - Go to: https://data.humdata.org/ or https://reliefweb.int/
   - Search for recent flood/drought/conflict data
   - Save to: `raw_data/disaster_events/[country_code]/`

## 📋 File Naming

The system now auto-detects files, but preferred naming:

- **Admin**: `[country]_adm1.shp` or any `.shp`/`.geojson` in the directory
- **Population**: `[country]_ppp_[year]_1km.tif` (e.g., `ken_ppp_2020_1km.tif`)
- **Infrastructure**: Any file with "health" in the name, or any `.shp`/`.geojson`
- **Disasters**: Any file with "flood" in the name, or any `.shp`/`.geojson`

## ✅ Verify Your Data

After downloading, verify files exist:

```bash
# Check admin boundaries
ls raw_data/admin_boundaries/[country_code]/*.shp

# Check population
ls raw_data/population/[country_code]/*.tif

# Check infrastructure
ls raw_data/infrastructure/[country_code]/*.shp

# Check disasters
ls raw_data/disaster_events/[country_code]/*.shp
```

## 🔄 Process Your Data

```bash
# Process the data (will auto-detect your files)
python -m core.data_processor

# Train the model
python scripts/1_train_model.py

# Run the app
python app/app.py
```

## 🌍 Example: Kenya

```bash
# 1. Download admin boundaries
# From GADM: https://gadm.org/download_country.html
# Save to: raw_data/admin_boundaries/ken/KEN_adm1.shp

# 2. Download population
# From WorldPop: https://www.worldpop.org/geodata/listing?id=75
# Save to: raw_data/population/ken/ken_ppp_2020_1km.tif

# 3. Download health facilities
# From HDX: https://data.humdata.org/dataset/health-facilities-kenya
# Save to: raw_data/infrastructure/ken/health_facilities.shp

# 4. Process
python -m core.data_processor
python scripts/1_train_model.py
python app/app.py
```

## 🆘 Troubleshooting

**"File not found" errors:**
- Check file paths match the directory structure
- Ensure files are extracted from zip files
- Check file extensions (.shp, .tif, .geojson)

**CRS errors:**
- The system auto-handles CRS reprojection
- If issues persist, ensure files have valid CRS metadata

**Memory errors:**
- Population rasters can be large
- Consider using smaller regions or resampling

## 📚 More Information

See [DATA_ACQUISITION_GUIDE.md](DATA_ACQUISITION_GUIDE.md) for detailed instructions and alternative data sources.

