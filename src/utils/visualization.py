#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for depth estimation results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_depth_map(depth_map, cmap='magma', vmin=None, vmax=None, title=None, save_path=None):
    """Visualize a single depth map"""
    plt.figure(figsize=(10, 8))
    
    # Filter out invalid values
    valid_mask = depth_map > 0
    
    # Set min/max values for visualization
    if vmin is None:
        vmin = 0
    if vmax is None and np.any(valid_mask):
        vmax = np.percentile(depth_map[valid_mask], 95)
    elif vmax is None:
        vmax = depth_map.max()
    
    plt.imshow(depth_map, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Depth')
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_error_map(pred_depth, gt_depth, mask=None, cmap='hot', title=None, save_path=None):
    """Visualize error map between predicted and ground truth depth"""
    plt.figure(figsize=(10, 8))
    
    # Create mask for valid pixels if not provided
    if mask is None:
        mask = (gt_depth > 0)
    
    # Calculate error
    error = np.zeros_like(gt_depth)
    error[mask] = np.abs(pred_depth[mask] - gt_depth[mask])
    
    # Set max error for visualization
    if np.any(mask):
        vmax = np.percentile(error[mask], 95)
    else:
        vmax = 1.0
    
    plt.imshow(error, cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(label='Absolute Error')
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_comparison(img, gt_depth, predictions, titles=None, save_path=None):
    """
    Visualize a comparison of multiple depth predictions
    
    Args:
        img: RGB input image
        gt_depth: Ground truth depth map
        predictions: Dictionary of model_name -> predicted depth map
        titles: Optional custom titles for each prediction
        save_path: Path to save visualization
    """
    num_models = len(predictions)
    models = list(predictions.keys())
    
    # Calculate total number of rows needed
    num_rows = 2 + (num_models + 1) // 3  # RGB+GT for first row, then models in groups of 3
    
    # Create figure with fixed figsize and spacing
    fig = plt.figure(figsize=(18, 6*num_rows))
    
    # Create GridSpec to manually manage layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(num_rows, 3, figure=fig, width_ratios=[1, 1, 1], 
                  wspace=0.1, hspace=0.2)
    
    # Show input RGB image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # Get valid mask and color range for depth maps
    valid_mask = gt_depth > 0
    if np.any(valid_mask):
        vmax = np.percentile(gt_depth[valid_mask], 95)
    else:
        vmax = np.max(gt_depth) if np.max(gt_depth) > 0 else 10.0
    
    # Show ground truth depth
    ax2 = fig.add_subplot(gs[0, 1])
    im_gt = ax2.imshow(gt_depth, cmap='magma', vmin=0, vmax=vmax)
    ax2.set_title('Ground Truth Depth')
    ax2.axis('off')
    
    # Add empty plot in third position of first row if needed
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Show each model prediction
    for i, model_name in enumerate(models):
        # Calculate position in grid
        row = 1 + i // 3
        col = i % 3
        
        if row >= num_rows:
            break  # Avoid IndexError if there are more models than grid cells
        
        # Scale prediction to have the same median as ground truth
        pred_depth = predictions[model_name]
        if np.any(valid_mask) and np.any(pred_depth > 0):
            scale = np.median(gt_depth[valid_mask]) / np.median(pred_depth[pred_depth > 0])
            scaled_pred = pred_depth * scale
        else:
            scaled_pred = pred_depth
        
        # Plot depth map
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(scaled_pred, cmap='magma', vmin=0, vmax=vmax)
        
        if titles and model_name in titles:
            ax.set_title(titles[model_name])
        else:
            ax.set_title(model_name)
        
        ax.axis('off')
    
    # Add colorbar to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax)
    cbar.set_label('Depth (m)')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def save_depth_as_colored_png(depth_map, save_path, cmap='magma', valid_mask=None):
    """
    Save depth map as a color-coded PNG image
    
    Args:
        depth_map: Depth map to visualize
        save_path: Path to save the image
        cmap: Colormap to use
        valid_mask: Optional mask for valid pixels
    """
    if valid_mask is None:
        valid_mask = depth_map > 0
    
    # Get colormap
    cmap_fn = plt.get_cmap(cmap)
    
    # Normalize depth values
    if np.any(valid_mask):
        vmin = depth_map[valid_mask].min()
        vmax = np.percentile(depth_map[valid_mask], 95)
        
        normalized = np.zeros_like(depth_map)
        normalized[valid_mask] = (depth_map[valid_mask] - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
    else:
        normalized = np.zeros_like(depth_map)
    
    # Apply colormap
    colored = cmap_fn(normalized)
    
    # Convert to uint8 RGB
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))

def save_error_as_colored_png(pred_depth, gt_depth, save_path, cmap='hot', mask=None):
    """
    Save error map as a color-coded PNG image
    
    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        save_path: Path to save the image
        cmap: Colormap to use
        mask: Optional mask for valid pixels
    """
    # Create mask for valid pixels if not provided
    if mask is None:
        mask = (gt_depth > 0) & (pred_depth > 0)
    
    # Calculate error
    error = np.zeros_like(gt_depth)
    if np.any(mask):
        error[mask] = np.abs(pred_depth[mask] - gt_depth[mask])
        vmax = np.percentile(error[mask], 95)
    else:
        vmax = 1.0
    
    # Normalize error values
    normalized = np.clip(error / vmax, 0, 1)
    
    # Get colormap
    cmap_fn = plt.get_cmap(cmap)
    
    # Apply colormap
    colored = cmap_fn(normalized)
    
    # Convert to uint8 RGB
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))

def create_comparison_visualization(results_dir, metrics, models, top_k=3):
    """
    Create comparison visualization for best and worst cases
    
    Args:
        results_dir: Directory with saved results
        metrics: Dictionary with metrics per model
        models: List of model names
        top_k: Number of best/worst samples to visualize
    """
    import pandas as pd
    
    # Create output directory
    vis_dir = os.path.join(results_dir, 'best_worst_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Collect per-sample metrics for each model
    df_list = []
    
    for model_name in models:
        if model_name in metrics and 'per_sample' in metrics[model_name]:
            per_sample = metrics[model_name]['per_sample']
            df = pd.DataFrame(per_sample)
            
            # Add model name
            df['model'] = model_name
            
            # Append to list
            df_list.append(df)
    
    if not df_list:
        print("No per-sample metrics found")
        return
    
    # Combine all dataframes
    all_metrics = pd.concat(df_list, ignore_index=True)
    
    # Get best and worst samples by RMSE for each model
    for model_name in models:
        model_metrics = all_metrics[all_metrics['model'] == model_name]
        
        if 'rmse' not in model_metrics.columns:
            continue
        
        # Remove rows with NaN values
        model_metrics = model_metrics.dropna(subset=['rmse'])
        
        if len(model_metrics) == 0:
            continue
        
        # Sort by RMSE
        best_samples = model_metrics.sort_values('rmse').head(top_k)
        worst_samples = model_metrics.sort_values('rmse', ascending=False).head(top_k)
        
        # Create visualizations
        print(f"Creating best/worst visualizations for {model_name}...")
        
        # Best samples
        for i, row in best_samples.iterrows():
            title = f"{model_name} - Best #{i+1} (RMSE: {row['rmse']:.4f})"
            save_path = os.path.join(vis_dir, f"{model_name}_best_{i+1}.png")
            
            # Need to load images from saved results
            # This depends on how the results are saved in your implementation
            # Example: visualize_from_saved_results(row['filename'], title, save_path)
        
        # Worst samples
        for i, row in worst_samples.iterrows():
            title = f"{model_name} - Worst #{i+1} (RMSE: {row['rmse']:.4f})"
            save_path = os.path.join(vis_dir, f"{model_name}_worst_{i+1}.png")
            
            # Example: visualize_from_saved_results(row['filename'], title, save_path)

if __name__ == "__main__":
    # Test visualization
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Visualization')
    parser.add_argument('--depth_map', type=str, help='Path to a depth map image')
    args = parser.parse_args()
    
    if args.depth_map:
        depth = cv2.imread(args.depth_map, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Error: Could not load depth map from {args.depth_map}")
        else:
            depth = depth.astype(np.float32)
            if len(depth.shape) == 3:
                depth = depth[:,:,0]  # Take first channel if multi-channel
            
            print(f"Depth map shape: {depth.shape}, dtype: {depth.dtype}")
            print(f"Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}")
            
            visualize_depth_map(depth, title="Test Depth Map")
    else:
        # Create test data
        depth = np.zeros((240, 320))
        y, x = np.mgrid[0:240, 0:320]
        depth = 10 + np.sqrt((x - 160)**2 + (y - 120)**2) / 10
        
        visualize_depth_map(depth, title="Test Depth Map")