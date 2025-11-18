"""
Main Flask application for GeoRelief-AI
Serves the web interface and API endpoints for priority score predictions
"""

from flask import Flask, render_template, jsonify
import geopandas as gpd
import sys
from pathlib import Path

# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.model_engine import get_model_engine

app = Flask(__name__)

# Load all assets on startup
model_engine = get_model_engine()
master_gdf = None

try:
    model_loaded = model_engine.load_model()
    if config.MASTER_DATASET_PATH.exists():
        master_gdf = gpd.read_file(config.MASTER_DATASET_PATH)
        print(f"Data loaded: {len(master_gdf)} regions")
    else:
        print(f"Warning: Master dataset not found at {config.MASTER_DATASET_PATH}")
        print("Please run data processing first.")
    print("Application initialized successfully.")
except Exception as e:
    print(f"Error loading assets: {e}")
    model_loaded = False


@app.route('/')
def index():
    """Render the main map page."""
    return render_template('index.html')


@app.route('/api/get_priority_scores')
def get_priority_scores():
    """
    This API runs the model on the latest data and returns
    the results as GeoJSON.
    """
    if not model_loaded or master_gdf is None:
        return jsonify({"error": "Model or data not loaded"}), 500
    
    try:
        # 1. Prepare data for the model (using standardized features)
        X = master_gdf[config.STANDARD_FEATURES].fillna(0)
        
        # 2. Get predictions
        predictions = model_engine.predict(X)
        
        # 3. Add predictions to the GeoDataFrame
        results_gdf = master_gdf.copy()
        results_gdf['Predicted_Priority_Score'] = predictions
        
        # 4. Normalize scores for display (0-100 scale)
        if predictions.max() > predictions.min():
            results_gdf['Priority_Score_Normalized'] = (
                (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 100
            )
        else:
            results_gdf['Priority_Score_Normalized'] = 50.0
        
        # 5. Return as GeoJSON
        return results_gdf.to_json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "data_loaded": master_gdf is not None,
        "regions_count": len(master_gdf) if master_gdf is not None else 0
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

