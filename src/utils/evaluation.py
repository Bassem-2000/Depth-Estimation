#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for depth estimation
"""

import numpy as np
import json
import os
import csv

def evaluate_depth(pred_depth, gt_depth, max_depth=100.0):
    """
    Calculate depth estimation evaluation metrics
    
    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        max_depth: Maximum depth value for evaluation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Create mask for valid pixels
    mask = (gt_depth > 0) & (gt_depth < max_depth) & (pred_depth > 0)
    
    # If no valid pixels, return NaN
    if not np.any(mask):
        return {metric: np.nan for metric in 
                ['abs_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    
    # Apply mask
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    
    # Ensure enough valid pixels for meaningful evaluation
    if len(pred) < 10:
        return {metric: np.nan for metric in 
                ['abs_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    
    # Scale prediction to have the same median as ground truth
    # This is a common approach for monocular methods that predict relative depth
    scale = np.median(gt) / np.median(pred)
    pred = pred * scale
    
    # Calculate metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    
    # Calculate error metrics with safety checks
    try:
        rmse = np.sqrt(np.mean((gt - pred) ** 2))
    except:
        rmse = np.nan
    
    try:
        # Avoid log(0)
        valid = (gt > 0) & (pred > 0)
        if np.any(valid):
            rmse_log = np.sqrt(np.mean((np.log(gt[valid]) - np.log(pred[valid])) ** 2))
        else:
            rmse_log = np.nan
    except:
        rmse_log = np.nan
    
    try:
        abs_rel = np.mean(np.abs(gt - pred) / gt)
    except:
        abs_rel = np.nan
    
    # Convert numpy values to Python native types
    return {
        'abs_rel': float(abs_rel) if not np.isnan(abs_rel) else float('nan'),
        'rmse': float(rmse) if not np.isnan(rmse) else float('nan'),
        'rmse_log': float(rmse_log) if not np.isnan(rmse_log) else float('nan'),
        'a1': float(a1) if not np.isnan(a1) else float('nan'),
        'a2': float(a2) if not np.isnan(a2) else float('nan'),
        'a3': float(a3) if not np.isnan(a3) else float('nan')
    }

def evaluate_disparity(pred_disp, gt_disp, max_disp=None):
    """
    Calculate disparity evaluation metrics
    
    Args:
        pred_disp: Predicted disparity map
        gt_disp: Ground truth disparity map
        max_disp: Maximum disparity value for evaluation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if max_disp is None:
        max_disp = np.max(gt_disp) * 1.5
    
    # Create mask for valid pixels
    mask = (gt_disp > 0) & (gt_disp < max_disp) & (pred_disp > 0)
    
    # If no valid pixels, return NaN
    if not np.any(mask):
        return {metric: np.nan for metric in 
                ['bad_1.0', 'bad_2.0', 'bad_3.0', 'mae', 'rmse']}
    
    # Apply mask
    pred = pred_disp[mask]
    gt = gt_disp[mask]
    
    # Ensure enough valid pixels for meaningful evaluation
    if len(pred) < 10:
        return {metric: np.nan for metric in 
                ['bad_1.0', 'bad_2.0', 'bad_3.0', 'mae', 'rmse']}
    
    # Calculate error
    error = np.abs(gt - pred)
    
    # Calculate metrics
    bad1 = (error > 1.0).mean()
    bad2 = (error > 2.0).mean()
    bad3 = (error > 3.0).mean()
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    
    # Convert numpy values to Python native types
    return {
        'bad_1.0': float(bad1) if not np.isnan(bad1) else float('nan'),
        'bad_2.0': float(bad2) if not np.isnan(bad2) else float('nan'),
        'bad_3.0': float(bad3) if not np.isnan(bad3) else float('nan'),
        'mae': float(mae) if not np.isnan(mae) else float('nan'),
        'rmse': float(rmse) if not np.isnan(rmse) else float('nan')
    }

def safe_mean(values):
    """Calculate mean of values, filtering out NaNs"""
    filtered = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    if filtered:
        return float(np.mean(filtered))  # Ensure it's a native Python float
    return float('nan')

def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple samples
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        avg_metrics: Dictionary of averaged metrics
    """
    # Initialize with all possible metrics
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    all_keys = [k for k in all_keys if k != 'filename']  # Exclude filename
    
    # Calculate average for each metric
    avg_metrics = {}
    for key in all_keys:
        values = [m.get(key, np.nan) for m in metrics_list]
        avg_metrics[key] = safe_mean(values)
    
    return avg_metrics

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif np.isnan(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def save_metrics(metrics, output_dir, models):
    """
    Save evaluation metrics to files
    
    Args:
        metrics: Dictionary of metrics per model
        output_dir: Output directory
        models: List of model names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison tables
    comparison_depth = {'Method': [model_name for model_name in models]}
    
    for metric in ['abs_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        comparison_depth[metric_display_name(metric)] = []
        for model_name in models:
            if model_name in metrics and 'depth' in metrics[model_name]:
                value = metrics[model_name]['depth'].get(metric, np.nan)
                comparison_depth[metric_display_name(metric)].append(
                    f"{value:.4f}" if not np.isnan(value) else "N/A"
                )
            else:
                comparison_depth[metric_display_name(metric)].append("N/A")
    
    # Save depth metrics as CSV
    with open(os.path.join(output_dir, 'comparison_depth.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(comparison_depth.keys())
        writer.writerows(zip(*comparison_depth.values()))
    
    # Create comparison for disparity if available
    if any('disparity' in metrics[model] for model in metrics if model in models):
        comparison_disp = {'Method': [model_name for model_name in models]}
        
        for metric in ['bad_1.0', 'bad_2.0', 'bad_3.0', 'mae', 'rmse']:
            comparison_disp[metric_display_name(metric)] = []
            for model_name in models:
                if model_name in metrics and 'disparity' in metrics[model_name]:
                    value = metrics[model_name]['disparity'].get(metric, np.nan)
                    comparison_disp[metric_display_name(metric)].append(
                        f"{value:.4f}" if not np.isnan(value) else "N/A"
                    )
                else:
                    comparison_disp[metric_display_name(metric)].append("N/A")
        
        # Save disparity metrics as CSV
        with open(os.path.join(output_dir, 'comparison_disparity.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(comparison_disp.keys())
            writer.writerows(zip(*comparison_disp.values()))
    
    # Convert metrics to native Python types and save as JSON
    try:
        results = {}
        for model_name in models:
            if model_name in metrics:
                results[model_name] = {}
                if 'depth' in metrics[model_name]:
                    results[model_name]['depth'] = metrics[model_name]['depth']
                if 'disparity' in metrics[model_name]:
                    results[model_name]['disparity'] = metrics[model_name]['disparity']
                if 'per_sample' in metrics[model_name]:
                    # Convert per_sample metrics
                    results[model_name]['per_sample'] = []
                    for sample in metrics[model_name]['per_sample']:
                        sample_dict = {}
                        for k, v in sample.items():
                            if k == 'filename':
                                sample_dict[k] = v
                            elif isinstance(v, (np.integer, np.floating, np.ndarray, np.bool_)):
                                sample_dict[k] = float(v) if not np.isnan(float(v)) else None
                            else:
                                sample_dict[k] = v
                        results[model_name]['per_sample'].append(sample_dict)
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving metrics.json: {e}")
        # Try a simpler approach
        with open(os.path.join(output_dir, 'metrics_simple.json'), 'w') as f:
            simple_results = {}
            for model_name in models:
                if model_name in metrics:
                    simple_results[model_name] = {}
                    if 'depth' in metrics[model_name]:
                        simple_results[model_name]['depth'] = {
                            k: str(v) for k, v in metrics[model_name]['depth'].items()
                        }
                    if 'disparity' in metrics[model_name]:
                        simple_results[model_name]['disparity'] = {
                            k: str(v) for k, v in metrics[model_name]['disparity'].items()
                        }
            json.dump(simple_results, f, indent=4)

def metric_display_name(metric):
    """Get a display-friendly name for each metric"""
    display_names = {
        'abs_rel': 'Abs Rel Error',
        'rmse': 'RMSE',
        'rmse_log': 'RMSE log',
        'a1': 'delta < 1.25',
        'a2': 'delta < 1.25^2',  # Changed from ² to ^2 to avoid encoding issues
        'a3': 'delta < 1.25^3',  # Changed from ³ to ^3 to avoid encoding issues
        'bad_1.0': 'Bad 1.0px (%)',
        'bad_2.0': 'Bad 2.0px (%)',
        'bad_3.0': 'Bad 3.0px (%)',
        'mae': 'MAE',
    }
    
    return display_names.get(metric, metric)

if __name__ == "__main__":
    # Test evaluation metrics
    import matplotlib.pyplot as plt
    
    # Create sample data
    gt = np.ones((100, 100)) * 10.0
    pred_good = gt * 1.1  # Good prediction (10% error)
    pred_bad = gt * 2.0   # Bad prediction (100% error)
    
    # Evaluate
    metrics_good = evaluate_depth(pred_good, gt)
    metrics_bad = evaluate_depth(pred_bad, gt)
    
    print("Good prediction metrics:")
    for k, v in metrics_good.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nBad prediction metrics:")
    for k, v in metrics_bad.items():
        print(f"  {k}: {v:.4f}")