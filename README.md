# GeoRelief-AI 🌍

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/AI-TensorFlow-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](https://github.com/)

> **A scalable, modular AI platform to optimize humanitarian resource allocation for any disaster, anywhere in the world.**

GeoRelief-AI moves beyond simple flood mapping. By integrating **Hydrology triggers** (GloFAS), **Soil Physics** (ERA5-Land), and **Social Vulnerability** (WPDX/IDMC), it calculates a dynamic **"Priority Score"**. This allows aid organizations to target the *most vulnerable people*, not just the wettest ground.

---

## 🇺🇳 Alignment with UN Sustainable Development Goals (SDGs)

This project directly supports the UN 2030 Agenda by leveraging AI for climate resilience:

| SDG | Goal Name | How GeoRelief-AI Contributes |
| :--- | :--- | :--- |
| <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-11.jpg" width="50"/> | **SDG 11: Sustainable Cities** | Enhances disaster resilience in urban settlements by identifying high-risk infrastructure. |
| <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-13.jpg" width="50"/> | **SDG 13: Climate Action** | Uses historical climate data (GloFAS/ERA5-Land) to predict and adapt to extreme weather events. |
| <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-06.jpg" width="50"/> | **SDG 6: Clean Water** | Prioritizes regions with compromised water points (WPDX data) during flood contamination events. |
| <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-03.jpg" width="50"/> | **SDG 3: Good Health** | Factors in healthcare density to ensure medical aid reaches isolated communities. |

---

## 🏗️ System Architecture

GeoRelief-AI operates on a **Three-Layer Intelligence Model**:

1. **Hazard Layer (The Trigger):**
   - *Real-time:* River Discharge (GloFAS), Soil Saturation (ERA5-Land), Satellite Flood Extent (UNOSAT).
   - *Static:* Historical Floodplains (NGA), Alluvial Plains.

2. **Exposure Layer (The Impact):**
   - *Population:* High-resolution density (WorldPop).
   - *Infrastructure:* Health facilities, Urban vs. Agricultural land cover.

3. **Vulnerability Layer (The Human Factor):**
   - *Social:* Internal Displacement (IDMC), Water Security (WPDX).

---

## 📊 The Science: Priority Score Engine

Unlike standard density maps which suffer from "Urban Bias," our model uses a weighted, log-normalized formula to identify severity in both urban and rural contexts.

$$
\text{Priority Score} = (\text{Hazard} \times 0.6) + (\text{Log(Pop)} \times 0.2) + (\text{Vulnerability} \times 0.2)
$$

Where the **Hazard Component** is dynamically composed of:

$$
\text{Hazard} = (0.35 \times \text{RealTime Flood}) + (0.30 \times \text{River Discharge}) + (0.20 \times \text{Soil Saturation}) + (0.15 \times \text{Hist. Floodplain})
$$

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GDAL (Required for spatial processing)

```bash
pip install -r requirements.txt
```

**Note:** If you encounter dependency conflicts (especially with `protobuf`), this is because TensorFlow 2.14 requires `protobuf<5.0.0`. The `requirements.txt` file pins the protobuf version to be compatible with TensorFlow. If you have other packages that require newer protobuf versions, consider using a virtual environment to isolate dependencies.

### Installation

1. Clone or download this repository
2. Install dependencies (see above)
3. Set up your data directory structure in `raw_data/`

### Data Setup

Place your geospatial data in the following structure:

```
raw_data/
├── admin_boundaries/
│   └── ken/                    # Country code (lowercase)
│       └── ken_admbnda_adm1_iebc_20191031.shp
├── population/
│   └── ken/
│       └── ken_ppp_2020_1km.tif
├── infrastructure/
│   └── ken/
│       └── ke-health-facilities.shp
├── disaster_events/
│   └── ken/
│       ├── ken-flood-fl-20231121.shp
│       └── nga_floodplains.geojson  # NGA base flood risk layer
├── vulnerability/
│   └── ken/
│       ├── wpdx_enhanced.csv        # Water point data
│       ├── event_data_ken.csv       # Displacement data
│       └── hydrology/
│           └── river_risk.json      # River discharge risk scores
└── land_cover/
    └── ken/
        └── land_cover.geojson       # Land use/land cover data
```

**Note:** If data files are not available, the system will generate sample data for testing purposes.

### 1. Automated Data Acquisition

We provide utility scripts to fetch public humanitarian data (HDX, Open-Meteo, GADM).

```bash
# 1. Download Vector Data (Admin, Health, Population, NGA Floodplains)
python scripts/download_data.py

# 2. Fetch Real-time Hydrology (River Flow & Soil Moisture)
python scripts/download_hydrology.py
```

This will guide you through downloading:
- Administrative boundaries from HDX/GADM
- Population data from WorldPop
- Health facilities from HDX
- Vulnerability data (water points, displacement)
- Disaster event data (UNOSAT, NGA floodplains)
- Land cover data (optional)

**Hydrology Data:**
- River discharge data from GloFAS (via Open-Meteo) - 10 years historical
- Soil saturation data from ERA5-Land (via Open-Meteo) - 10 years historical

### 2. Data Processing Pipeline

Standardizes distinct CRS, Rasters, and Vectors into a unified GeoJSON Master Dataset.

```bash
python -m core.data_processor
```

This will:
- Load and process geospatial data
- Calculate vulnerability index from water access, health access, and displacement
- Fetch and integrate hydrology data (river risk, soil saturation)
- Process land cover data (alluvial plains, urban areas, agricultural land)
- Calculate base flood risk from NGA floodplains
- Standardize features across countries
- Save processed data to `processed_data/global_master_dataset.geojson`

### 3. Train the Model

Trains the neural network to calibrate risk weights based on historical events.

```bash
python scripts/1_train_model.py
```

This will:
- Load the processed dataset
- Calculate priority scores using the comprehensive formula
- Train a neural network model (5 input features including alluvial_plain_pct)
- Save the model and scaler to `models/`

### 4. Launch the Platform

Starts the Flask API and Interactive Leaflet Map.

```bash
python app/app.py
```

Access the dashboard at `http://localhost:5000`

---

## 📂 Project Structure

```
GeoRelief-AI/
├── app/                  # Flask Web Application
│   ├── static/           # JS/CSS assets
│   └── templates/        # HTML Frontend
├── core/                 # Intelligence Core
│   ├── data_processor.py # geospatial ETL pipeline
│   ├── model_engine.py   # Tensorflow/Scikit-Learn logic
│   └── config.py         # Global settings
├── scripts/              # Utilities
│   ├── download_data.py  # Data fetchers (HDX/Open-Meteo APIs)
│   ├── download_hydrology.py  # Hydrology data download
│   └── 1_train_model.py  # Training entry point
├── notebooks/            # Jupyter notebooks for exploration
├── models/               # Trained models (created after training)
├── processed_data/       # Cleaned Master Datasets
└── raw_data/             # Input Datasets (GitIgnored)
    ├── admin_boundaries/
    ├── population/
    ├── infrastructure/
    ├── disaster_events/
    ├── vulnerability/
    └── land_cover/
```

---

## 📊 Features

### Standardized Features

The system uses standardized features that work across all countries:

- **population_density**: Population per square kilometer (log-transformed to reduce urban bias)
- **health_facility_density**: Health facilities per population
- **vulnerability_index**: Composite vulnerability score (0-1) based on:
  - Water access (33% weight)
  - Health access (33% weight)
  - Internal displacement (33% weight)
- **flooded_population**: Disaster-affected population (disaster-specific)
- **alluvial_plain_pct**: Percentage of county covered by alluvial plains (flood-prone areas)

### Additional Data Features

The system also processes and stores additional features for enhanced analysis:

- **river_risk_score**: River discharge risk from GloFAS model (0-1, based on 10-year historical data)
- **soil_saturation**: Soil moisture saturation index from ERA5-Land (0-1, based on 10-year historical data)
- **base_flood_risk**: Base flood risk from NGA floodplains (percentage coverage, static risk layer)
- **urban_area_pct**: Percentage of county covered by urban areas
- **agricultural_pct**: Percentage of county covered by agricultural land

---

## 🔧 Configuration

All configuration is managed in `core/config.py`, including:
- Data directory paths
- Model paths
- Feature names
- Target variable

---

## 🌍 Scaling to New Countries

To add a new country:

1. Add data files to `raw_data/` following the directory structure
2. Update country-specific mappings in `core/data_processor.py` if needed (e.g., river station mappings)
3. Run data processing: `python -m core.data_processor`
4. Download hydrology data: `python scripts/download_hydrology.py` (if applicable)
5. Train the model: `python scripts/1_train_model.py`
6. The model will automatically work with the new country's data

**Note:** The system is designed to be modular and automatically detects data files with flexible naming conventions.

---

## 📝 API Endpoints

- `GET /` - Main web interface
- `GET /api/get_priority_scores` - Returns GeoJSON with priority scores
- `GET /api/health` - Health check endpoint

---

## 🛠️ Tech Stack

**Geospatial:** GeoPandas, Rasterio, Rasterstats, Shapely

**Machine Learning:** TensorFlow (Keras), Scikit-Learn

**Backend:** Python, Flask

**Frontend:** Leaflet.js, Bootstrap

**APIs Integrated:** Open-Meteo (GloFAS, ERA5-Land), ArcGIS REST API, HDX

**Data Sources:**
- **Population:** WorldPop
- **Administrative Boundaries:** GADM, HDX
- **Infrastructure:** HDX, OpenStreetMap
- **Vulnerability:** WPDX (Water Points), IDMC (Displacement)
- **Hydrology:** Open-Meteo (GloFAS, ERA5-Land)
- **Flood Data:** UNOSAT, NGA Floodplains
- **Land Cover:** User-provided or public datasets

---

## 📄 License

Distributed under the MIT License. See LICENSE for more information.

---

## 🤝 Contributing

We welcome contributions, particularly in adding new country data connectors or refining the vulnerability index. Please open an issue to discuss proposed changes.

---

## 📧 Support

For issues or questions, please open an issue on the project repository.
