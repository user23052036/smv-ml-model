"""
Utility functions for model persistence and general helpers.
"""
import joblib
import os


def save_model(model, path):
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: The model object to save (sklearn model, encoder, etc.)
        path: File path where the model will be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✓ Model saved to {path}")


def load_model(path):
    """
    Load a trained model from disk.
    
    Args:
        path: File path to the saved model
        
    Returns:
        The loaded model object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
