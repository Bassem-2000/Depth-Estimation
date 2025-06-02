"""
Depth Estimation Inference Script

This script runs monocular and stereo depth estimation models and evaluates their performance.
"""

import os
import sys
import argparse
import importlib
import time
import json
import torch
from tqdm import tqdm
import numpy as np

# Add parent directory to path to allow importing from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.utils.data_loader import load_dataset
from src.utils.evaluation import evaluate_depth, evaluate_disparity, aggregate_metrics, save_metrics
from src.utils.visualization import visualize_comparison, save_depth_as_colored_png
import config

def load_model(model_name):
    """
    Load a model by name from the configuration
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        model: The loaded model
    """
    if model_name not in config.MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = config.MODELS[model_name]
    
    # Import the module
    module = importlib.import_module(model_config["module"])
    
    # Get the function
    function = getattr(module, model_config["function"])
    
    # Create the model
    model = function(**model_config["args"])
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Depth Estimation Evaluation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Models to evaluate')
    parser.add_argument('--mono', action='store_true',
                        help='Evaluate all monocular models')
    parser.add_argument('--stereo', action='store_true',
                        help='Evaluate all stereo models')
    parser.add_argument('--list_models', action='store_true',
                        help='List all available models and exit')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        config.print_available_models()
        return
    
    # Determine models to evaluate
    models_to_evaluate = []
    
    if args.models:
        models_to_evaluate.extend(args.models)
    
    if args.mono:
        models_to_evaluate.extend(config.get_model_names_by_type("monocular"))
    
    if args.stereo:
        models_to_evaluate.extend(config.get_model_names_by_type("stereo"))
    
    # If no models specified, use defaults
    if not models_to_evaluate:
        models_to_evaluate = config.DEFAULT_ALL_MODELS
    
    # Remove duplicates
    models_to_evaluate = list(dict.fromkeys(models_to_evaluate))
    
    # Check if all models are valid
    for model_name in models_to_evaluate:
        if model_name not in config.MODELS:
            print(f"Error: Unknown model '{model_name}'")
            print("Available models:")
            config.print_available_models()
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load dataset
    try:
        dataset, camera_config = load_dataset(args.data_path, args.num_samples)
        print(f"Successfully loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create subdirectories for visualization
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create directory for individual results
    if config.VISUALIZATION["save_individual_results"]:
        for model_name in models_to_evaluate:
            model_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
    
    # Load models
    models = {}
    for model_name in models_to_evaluate:
        try:
            model = load_model(model_name)
            models[model_name] = model
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Initialize metrics storage
    metrics = {model_name: {'per_sample': []} for model_name in models}
    
    # Process each sample
    print(f"Processing {len(dataset)} samples with {len(models)} models...")
    for i, sample in enumerate(tqdm(dataset)):
        # Extract data
        left_img = sample['left']
        right_img = sample['right']
        gt_depth = sample['gt_depth']
        gt_disp = sample['gt_disp']
        filename = sample['filename']
        
        # Skip bad samples
        if left_img.size == 0 or right_img.size == 0:
            print(f"Skipping empty image: {filename}")
            continue
        
        # Dictionary to store all predictions for this sample
        predictions = {}
        
        # Process each model
        for model_name, model in models.items():
            model_type = config.MODELS[model_name]["type"]
            
            try:
                start_time = time.time()
                
                if model_type == "monocular":
                    # Monocular depth estimation
                    depth_map = model(left_img)
                    
                    # We don't have disparity for monocular models
                    disp_map = None
                elif model_type == "stereo":
                    # Stereo depth estimation
                    depth_map, disp_map = model(left_img, right_img, camera_config)
                else:
                    print(f"Unknown model type: {model_type}")
                    continue
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Store prediction
                predictions[model_name] = depth_map
                
                # Evaluate
                sample_metrics = {
                    'filename': filename,
                    'inference_time': inference_time
                }
                
                # Evaluate depth if available
                if depth_map is not None:
                    depth_metrics = evaluate_depth(depth_map, gt_depth, config.EVALUATION["max_depth"])
                    sample_metrics.update(depth_metrics)
                
                # Evaluate disparity if available
                if disp_map is not None and gt_disp is not None:
                    disp_metrics = evaluate_disparity(disp_map, gt_disp, config.EVALUATION["max_disp"])
                    sample_metrics.update({f"disp_{k}": v for k, v in disp_metrics.items()})
                
                # Add to metrics
                metrics[model_name]['per_sample'].append(sample_metrics)
                
                # Save individual depth map if requested
                if config.VISUALIZATION["save_individual_results"]:
                    model_out_dir = os.path.join(args.output_dir, model_name)
                    os.makedirs(model_out_dir, exist_ok=True)
                    
                    base_name = os.path.splitext(filename)[0]
                    if depth_map is not None:
                        depth_path = os.path.join(model_out_dir, f"{base_name}_depth.png")
                        save_depth_as_colored_png(depth_map, depth_path, config.VISUALIZATION["colormap"])
                    
                    if disp_map is not None:
                        disp_path = os.path.join(model_out_dir, f"{base_name}_disp.png")
                        save_depth_as_colored_png(disp_map, disp_path, config.VISUALIZATION["colormap"])
            
            except Exception as e:
                print(f"Error processing {filename} with model {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save visualization comparing all models
        if predictions:
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}.png")
            try:
                visualize_comparison(left_img, gt_depth, predictions, save_path=vis_path)
            except Exception as e:
                print(f"Error creating visualization for {filename}: {e}")
    
    # Compute average metrics for each model
    for model_name in models:
        # Aggregate depth metrics
        depth_metrics = [
            {k: v for k, v in m.items() if k in config.EVALUATION["metrics"]["depth"]}
            for m in metrics[model_name]['per_sample']
        ]
        metrics[model_name]['depth'] = aggregate_metrics(depth_metrics)
        
        # Aggregate disparity metrics if available
        disp_keys = [f"disp_{k}" for k in config.EVALUATION["metrics"]["disparity"]]
        if any(disp_key in m for m in metrics[model_name]['per_sample'] for disp_key in disp_keys):
            disp_metrics = [
                {k.replace("disp_", ""): v for k, v in m.items() if k in disp_keys}
                for m in metrics[model_name]['per_sample']
            ]
            metrics[model_name]['disparity'] = aggregate_metrics(disp_metrics)
    
    # Print results
    print("\nResults Summary:")
    for model_name in models:
        print(f"\n{model_name}:")
        
        if 'depth' in metrics[model_name]:
            print("  Depth Metrics:")
            for k, v in metrics[model_name]['depth'].items():
                print(f"    {k}: {v:.4f}" if not np.isnan(v) else f"    {k}: nan")
        
        if 'disparity' in metrics[model_name]:
            print("  Disparity Metrics:")
            for k, v in metrics[model_name]['disparity'].items():
                print(f"    {k}: {v:.4f}" if not np.isnan(v) else f"    {k}: nan")
    
    # Save metrics to file
    save_metrics(metrics, args.output_dir, list(models.keys()))
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Visualizations saved to {vis_dir}")

if __name__ == '__main__':
    main()