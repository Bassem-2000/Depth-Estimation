#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for depth estimation models
"""

# Model configurations
MODELS = {
    # Monocular models
    "midas_large": {
        "type": "monocular",
        "module": "src.models.monocular.midas",
        "function": "create_midas_model",
        "args": {
            "model_type": "DPT_Large"
        },
        "description": "MiDaS DPT Large - high accuracy but slower"
    },
    "midas_hybrid": {
        "type": "monocular",
        "module": "src.models.monocular.midas",
        "function": "create_midas_model",
        "args": {
            "model_type": "DPT_Hybrid"
        },
        "description": "MiDaS DPT Hybrid - balanced accuracy and speed"
    },
    "midas_small": {
        "type": "monocular",
        "module": "src.models.monocular.midas",
        "function": "create_midas_model",
        "args": {
            "model_type": "MiDaS"
        },
        "description": "MiDaS Small - fast but less accurate"
    },
    
    # Stereo models
    "sgbm_fast": {
        "type": "stereo",
        "module": "src.models.stereo.sgbm",
        "function": "create_sgbm_preset",
        "args": {
            "preset": "fast"
        },
        "description": "SGBM Fast - faster but less accurate"
    },
    "sgbm_balanced": {
        "type": "stereo",
        "module": "src.models.stereo.sgbm",
        "function": "create_sgbm_preset",
        "args": {
            "preset": "balanced"
        },
        "description": "SGBM Balanced - good balance of speed and accuracy"
    },
    "sgbm_accurate": {
        "type": "stereo",
        "module": "src.models.stereo.sgbm",
        "function": "create_sgbm_preset",
        "args": {
            "preset": "accurate"
        },
        "description": "SGBM Accurate - high accuracy but slower"
    }
    
    # Add more models here
}

# Evaluation settings
EVALUATION = {
    "metrics": {
        "depth": ["abs_rel", "rmse", "rmse_log", "a1", "a2", "a3"],
        "disparity": ["bad_1.0", "bad_2.0", "bad_3.0", "mae", "rmse"]
    },
    "max_depth": 100.0,  # Maximum depth value for evaluation
    "max_disp": None     # Maximum disparity value for evaluation (None = auto)
}

# Visualization settings
VISUALIZATION = {
    "save_individual_results": True,  # Save individual depth maps
    "save_error_maps": True,          # Save error maps
    "create_comparison_grid": True,   # Create grid comparison of all models
    "colormap": "magma",              # Colormap for depth visualization
    "error_colormap": "hot",          # Colormap for error visualization
    "figure_dpi": 300                 # DPI for saved figures
}

# Default model groups
DEFAULT_MONO_MODELS = ["midas_large"]
DEFAULT_STEREO_MODELS = ["sgbm_balanced"]
DEFAULT_ALL_MODELS = DEFAULT_MONO_MODELS + DEFAULT_STEREO_MODELS

def get_model_names_by_type(model_type=None):
    """
    Get list of model names by type
    
    Args:
        model_type: Model type ('monocular', 'stereo', or None for all)
    
    Returns:
        List of model names
    """
    if model_type is None:
        return list(MODELS.keys())
    
    return [name for name, config in MODELS.items() 
            if config.get("type") == model_type]

def print_available_models():
    """Print information about available models"""
    print("\nAvailable Monocular Models:")
    for name in get_model_names_by_type("monocular"):
        print(f"  - {name}: {MODELS[name]['description']}")
    
    print("\nAvailable Stereo Models:")
    for name in get_model_names_by_type("stereo"):
        print(f"  - {name}: {MODELS[name]['description']}")

if __name__ == "__main__":
    print("Depth Estimation Models Configuration")
    print("====================================")
    print_available_models()