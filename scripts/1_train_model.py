"""
Model training script for GeoRelief-AI
Trains a neural network on standardized features for priority score prediction
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


def train_model():
    """Train the priority score prediction model"""
    
    # Load the master dataset
    if not config.MASTER_DATASET_PATH.exists():
        print(f"Error: Master dataset not found at {config.MASTER_DATASET_PATH}")
        print("Please run data processing first (core/data_processor.py)")
        return
    
    print(f"Loading data from {config.MASTER_DATASET_PATH}...")
    gdf = gpd.read_file(config.MASTER_DATASET_PATH)
    
    if len(gdf) == 0:
        print("Error: Dataset is empty")
        return
    
    print(f"Loaded {len(gdf)} regions")
    
    # 1. Define Features (X) and Target (y)
    # These are our STANDARDIZED features, applicable to any country
    FEATURES = config.STANDARD_FEATURES
    
    # Check that all features exist
    missing_features = [f for f in FEATURES if f not in gdf.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        print("Filling with zeros...")
        for feature in missing_features:
            gdf[feature] = 0
    
    # The target is what we want to predict: PriorityScore
    # A high score means high need for humanitarian assistance.
    # 
    # Updated formula: Priority = (Hazard * 0.6) + (Pop_Score * 0.2) + (Vuln * 0.2)
    # Where Hazard = (event_flood * 0.35) + (base_flood_risk * 0.15) + (river_risk * 0.3) + (soil_saturation * 0.2)
    # Note: base_flood_risk is from NGA floodplains (static, low-res) - used as baseline risk
    
    # Normalize flooded population (event-specific, 0-1)
    if 'flooded_population' in gdf.columns:
        if gdf['flooded_population'].max() > 0:
            flood_extent_normalized = (gdf['flooded_population'] - gdf['flooded_population'].min()) / (gdf['flooded_population'].max() - gdf['flooded_population'].min() + 1e-10)
        else:
            flood_extent_normalized = pd.Series([0.0] * len(gdf))
    else:
        flood_extent_normalized = pd.Series([0.0] * len(gdf))
    
    # Get base flood risk (NGA floodplains - already 0-1 normalized as percentage coverage)
    base_flood_risk = gdf.get('base_flood_risk', pd.Series([0.0] * len(gdf)))
    
    # Get river risk and soil saturation (already 0-1 normalized)
    river_risk = gdf.get('river_risk_score', pd.Series([0.0] * len(gdf)))
    soil_saturation = gdf.get('soil_saturation', pd.Series([0.0] * len(gdf)))
    
    # Calculate Hazard component (weighted sum of event flood, base flood risk, river, soil)
    # Event flood gets higher weight (0.35) as it's current/real-time
    # Base flood risk gets lower weight (0.15) as it's static/low-res baseline
    hazard = (
        flood_extent_normalized * 0.35 +      # Event-specific flood extent
        base_flood_risk * 0.15 +              # Base flood risk (NGA floodplains)
        river_risk * 0.3 +                    # River discharge risk
        soil_saturation * 0.2                 # Soil saturation
    )
    
    # Population score (use log to reduce urban bias)
    if 'population_density' in gdf.columns:
        pop_score = np.log1p(gdf['population_density'])  # log1p = log(1+x) to handle zeros
        if pop_score.max() > pop_score.min():
            pop_score = (pop_score - pop_score.min()) / (pop_score.max() - pop_score.min() + 1e-10)
        else:
            pop_score = pd.Series([0.5] * len(gdf))
    else:
        pop_score = pd.Series([0.5] * len(gdf))
    
    # Vulnerability (already 0-1)
    vulnerability = gdf.get('vulnerability_index', pd.Series([0.5] * len(gdf)))
    
    # Final Priority Score
    gdf[config.TARGET_VARIABLE] = (
        hazard * 0.6 +
        pop_score * 0.2 +
        vulnerability * 0.2
    )
    
    # Handle infinite or NaN values
    gdf[config.TARGET_VARIABLE] = gdf[config.TARGET_VARIABLE].replace([np.inf, -np.inf], 0)
    gdf[config.TARGET_VARIABLE] = gdf[config.TARGET_VARIABLE].fillna(0)
    
    X = gdf[FEATURES].fillna(0)
    y = gdf[config.TARGET_VARIABLE]
    
    print(f"\nFeatures: {FEATURES}")
    print(f"Target: {config.TARGET_VARIABLE}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"Target mean: {y.mean():.2f}")
    
    # 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Split data (if we have enough samples)
    if len(X) > 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
    else:
        print("\nWarning: Small dataset, using all data for training")
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = y, y
    
    # 4. Build Model
    print("\nBuilding model...")
    model = keras.Sequential([
        layers.Input(shape=[len(FEATURES)]),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu', name='dense_3'),
        layers.Dense(1, activation='linear', name='output')  # Regression output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    print(model.summary())
    
    # 5. Train Model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=min(8, len(X_train)),
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
        verbose=1
    )
    
    # 6. Evaluate Model
    if len(X_test) > 0:
        print("\nEvaluating model...")
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Make predictions for comparison
        predictions = model.predict(X_test, verbose=0)
        print(f"\nSample predictions vs actual:")
        for i in range(min(5, len(predictions))):
            print(f"  Predicted: {predictions[i][0]:.2f}, Actual: {y_test.iloc[i]:.2f}")
    
    # 7. Save Model and Scaler
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model.save(config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")
    
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("GeoRelief-AI Model Training")
    print("=" * 60)
    train_model()

