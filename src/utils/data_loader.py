#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading utilities for depth estimation datasets
"""

import os
import json
import numpy as np
import cv2
from tqdm import tqdm

def load_dataset(dataset_path, num_samples=None):
    """
    Load dataset images and ground truth
    
    Args:
        dataset_path: Path to the dataset directory
        num_samples: Number of samples to load (None for all)
    
    Returns:
        dataset: List of dictionaries containing image data
        camera_config: Camera configuration
    """
    # Load camera configuration
    camera_config_path = os.path.join('data/sensor_config.json')
    if os.path.exists(camera_config_path):
        with open(camera_config_path, 'r') as f:
            camera_config = json.load(f)
    else:
        print(f"Warning: Camera config not found at {camera_config_path}")
        # Default camera config based on the assignment description
        camera_config = {
            "baseline": 0.4,
            "K": [
                [658.5570007600752, 0.0, 960.0],
                [0.0, 658.5570007600752, 600.0],
                [0.0, 0.0, 1.0]
            ],
            "image_w": 1920,
            "image_h": 1200
        }
    
    # Get list of all files
    left_path = os.path.join(dataset_path, 'left')
    left_files = sorted([f for f in os.listdir(left_path) if os.path.isfile(os.path.join(left_path, f))])
    
    if num_samples is not None:
        left_files = left_files[:num_samples]
    
    dataset = []
    
    print(f"Loading {len(left_files)} samples...")
    for filename in tqdm(left_files):
        # Extract base filename without extension
        base_name = os.path.splitext(filename)[0]
        
        # Load left and right images
        left_path = os.path.join(dataset_path, 'left', filename)
        right_path = os.path.join(dataset_path, 'right', filename)
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            print(f"Warning: Could not load image pair: {filename}")
            continue
        
        # Convert BGR to RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Load ground truth depth and disparity
        gt_depth_path = os.path.join(dataset_path, 'gt_depth', filename)
        gt_disp_path = os.path.join(dataset_path, 'gt_disp', filename)
        
        # Read the ground truth images
        gt_depth_img = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        gt_disp_img = cv2.imread(gt_disp_path, cv2.IMREAD_UNCHANGED)
        
        # Try alternative extensions if loading fails
        if gt_depth_img is None:
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                alt_path = os.path.join(dataset_path, 'gt_depth', base_name + ext)
                if os.path.exists(alt_path):
                    gt_depth_img = cv2.imread(alt_path, cv2.IMREAD_UNCHANGED)
                    if gt_depth_img is not None:
                        print(f"  Found alternative depth GT at {alt_path}")
                        break
        
        if gt_disp_img is None:
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                alt_path = os.path.join(dataset_path, 'gt_disp', base_name + ext)
                if os.path.exists(alt_path):
                    gt_disp_img = cv2.imread(alt_path, cv2.IMREAD_UNCHANGED)
                    if gt_disp_img is not None:
                        print(f"  Found alternative disparity GT at {alt_path}")
                        break
        
        # Create dummy arrays if ground truth still not found
        if gt_depth_img is None:
            print(f"  Creating dummy depth for {filename}")
            gt_depth_img = np.zeros_like(left_img[:,:,0], dtype=np.uint8)
        
        if gt_disp_img is None:
            print(f"  Creating dummy disparity for {filename}")
            gt_disp_img = np.zeros_like(left_img[:,:,0], dtype=np.uint8)
        
        # Convert ground truth images to float values
        gt_depth = gt_depth_img.astype(np.float32)
        gt_disp = gt_disp_img.astype(np.float32)
        
        # If ground truth is multi-channel, take first channel
        if len(gt_depth.shape) == 3:
            gt_depth = gt_depth[:,:,0]
        
        if len(gt_disp.shape) == 3:
            gt_disp = gt_disp[:,:,0]
        
        # Important: Scale values appropriately!
        # Based on the observed ranges (min=2.000, max=255.000),
        # the ground truth appears to be 8-bit images.
        # We'll convert to metric depth assuming a reasonable range.
        
        # 1. Scale depth values from [0-255] to reasonable metric range [0-100m]
        # If gt_depth is already in a good range, this won't change much
        if gt_depth.max() <= 255:
            # Assuming 8-bit images with values [0-255]
            # Scale to a reasonable depth range (e.g., 0-100 meters)
            # Zero values stay as zero (invalid)
            valid_mask = gt_depth > 0
            if np.any(valid_mask):
                gt_depth[valid_mask] = gt_depth[valid_mask] / 255.0 * 100.0
        
        # Print stats for first image to help debug
        if len(dataset) == 0:
            print(f"Ground truth depth stats: min={gt_depth.min():.3f}, max={gt_depth.max():.3f}, mean={gt_depth.mean():.3f}")
            print(f"Ground truth disparity stats: min={gt_disp.min():.3f}, max={gt_disp.max():.3f}, mean={gt_disp.mean():.3f}")
        
        dataset.append({
            'left': left_img,
            'right': right_img,
            'gt_depth': gt_depth,
            'gt_disp': gt_disp,
            'filename': filename
        })
    
    return dataset, camera_config

def inspect_dataset(dataset_path, num_samples=3):
    """Visualize sample images from the dataset"""
    import matplotlib.pyplot as plt
    
    # Get list of all files
    left_path = os.path.join(dataset_path, 'left')
    left_files = sorted([f for f in os.listdir(left_path) if os.path.isfile(os.path.join(left_path, f))])[:num_samples]
    
    for filename in left_files:
        # Load images
        left_img = cv2.imread(os.path.join(dataset_path, 'left', filename))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        
        right_img = cv2.imread(os.path.join(dataset_path, 'right', filename))
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        gt_depth_img = cv2.imread(os.path.join(dataset_path, 'gt_depth', filename), cv2.IMREAD_UNCHANGED)
        gt_disp_img = cv2.imread(os.path.join(dataset_path, 'gt_disp', filename), cv2.IMREAD_UNCHANGED)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Display RGB image
        axes[0].imshow(left_img)
        axes[0].set_title("Left Image")
        axes[0].axis('off')
        
        axes[1].imshow(right_img)
        axes[1].set_title("Right Image")
        axes[1].axis('off')
        
        # Display raw depth image
        if gt_depth_img is not None:
            if len(gt_depth_img.shape) == 3:
                # Show first channel if multi-channel
                axes[2].imshow(gt_depth_img[:,:,0], cmap='magma')
            else:
                axes[2].imshow(gt_depth_img, cmap='magma')
            axes[2].set_title(f"Depth GT\nShape: {gt_depth_img.shape}, Type: {gt_depth_img.dtype}")
        else:
            axes[2].text(0.5, 0.5, "Not Found", ha='center', va='center')
        axes[2].axis('off')
        
        # Display raw disparity image
        if gt_disp_img is not None:
            if len(gt_disp_img.shape) == 3:
                axes[3].imshow(gt_disp_img[:,:,0], cmap='magma')
            else:
                axes[3].imshow(gt_disp_img, cmap='magma')
            axes[3].set_title(f"Disparity GT\nShape: {gt_disp_img.shape}, Type: {gt_disp_img.dtype}")
        else:
            axes[3].text(0.5, 0.5, "Not Found", ha='center', va='center')
        axes[3].axis('off')
        
        plt.suptitle(f"Dataset Sample: {filename}")
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    # Test data loading
    import argparse
    parser = argparse.ArgumentParser(description='Test Dataset Loading')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()
    
    # Inspect the dataset
    inspect_dataset(args.data_path)
    
    # Test full dataset loading
    dataset, camera_config = load_dataset(args.data_path, num_samples=5)
    print(f"Loaded {len(dataset)} samples successfully")
    print(f"Camera config: {camera_config}")