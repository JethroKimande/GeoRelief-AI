"""
Model engine for GeoRelief-AI
Handles model loading, prediction, and preprocessing
"""

import numpy as np
import joblib
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from . import config


class ModelEngine:
    """Wrapper class for model operations"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded = False
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if config.MODEL_PATH.exists():
                self.model = keras.models.load_model(config.MODEL_PATH)
                print(f"Model loaded from {config.MODEL_PATH}")
            else:
                print(f"Warning: Model not found at {config.MODEL_PATH}")
                return False
            
            if config.SCALER_PATH.exists():
                self.scaler = joblib.load(config.SCALER_PATH)
                print(f"Scaler loaded from {config.SCALER_PATH}")
            else:
                print(f"Warning: Scaler not found at {config.SCALER_PATH}")
                # Create a default scaler (will need to be fitted)
                self.scaler = StandardScaler()
            
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X):
        """
        Make predictions using the loaded model.
        
        Args:
            X: DataFrame or array with features
        
        Returns:
            Array of predictions
        """
        if not self.loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure we have the right features
        if hasattr(X, 'columns'):
            # It's a DataFrame
            X_features = X[config.STANDARD_FEATURES].fillna(0)
        else:
            # It's an array
            X_features = X
        
        # Scale the data
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X_features)
            except ValueError:
                # Scaler might not be fitted, try to fit it
                print("Warning: Scaler not fitted, fitting now...")
                self.scaler.fit(X_features)
                X_scaled = self.scaler.transform(X_features)
        else:
            X_scaled = X_features
        
        # Make predictions
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.loaded


# Global model engine instance
_model_engine = None


def get_model_engine():
    """Get or create the global model engine instance"""
    global _model_engine
    if _model_engine is None:
        _model_engine = ModelEngine()
    return _model_engine

