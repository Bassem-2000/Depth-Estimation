#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Semi-Global Block Matching (SGBM) stereo depth estimation model
"""

import cv2
import numpy as np

class SGBMModel:
    """Semi-Global Block Matching stereo depth estimation model"""
    
    def __init__(self, min_disp=0, num_disp=192, block_size=11, p1=None, p2=None, 
                 disp12_max_diff=1, uniqueness_ratio=15, speckle_window_size=100,
                 speckle_range=32, pre_filter_cap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
        """
        Initialize the SGBM model
        
        Args:
            min_disp: Minimum disparity
            num_disp: Number of disparities
            block_size: Block size
            p1: First parameter controlling disparity smoothness
            p2: Second parameter controlling disparity smoothness
            disp12_max_diff: Maximum allowed difference in the left-right disparity check
            uniqueness_ratio: Margin in percentage by which the best (minimum) computed cost should win
            speckle_window_size: Maximum size of smooth disparity regions
            speckle_range: Maximum disparity variation within a connected component
            pre_filter_cap: Truncation value for the prefiltered image pixels
            mode: SGBM algorithm mode
        """
        self.min_disp = min_disp
        self.num_disp = num_disp
        self.block_size = block_size
        
        # Set default P1 and P2 if not provided
        if p1 is None:
            self.p1 = 8 * 3 * block_size**2
        else:
            self.p1 = p1
            
        if p2 is None:
            self.p2 = 32 * 3 * block_size**2
        else:
            self.p2 = p2
        
        self.disp12_max_diff = disp12_max_diff
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window_size = speckle_window_size
        self.speckle_range = speckle_range
        self.pre_filter_cap = pre_filter_cap
        self.mode = mode
        
        # Create SGBM object
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap,
            mode=self.mode
        )
    
    def __call__(self, left_img, right_img, camera_config=None):
        """
        Estimate depth from stereo image pair
        
        Args:
            left_img: Left RGB image (numpy array)
            right_img: Right RGB image (numpy array)
            camera_config: Optional camera configuration with baseline and focal length
        
        Returns:
            depth_map: Estimated depth map
            disparity_map: Estimated disparity map
        """
        return self.estimate_depth(left_img, right_img, camera_config)
    
    def estimate_depth(self, left_img, right_img, camera_config=None):
        """
        Estimate depth from stereo image pair
        
        Args:
            left_img: Left RGB image (numpy array)
            right_img: Right RGB image (numpy array)
            camera_config: Optional camera configuration with baseline and focal length
        
        Returns:
            depth_map: Estimated depth map
            disparity_map: Estimated disparity map
        """
        # Convert to grayscale
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Convert disparity to depth if camera config is provided
        if camera_config is not None:
            # Extract camera parameters
            if isinstance(camera_config, dict):
                baseline = camera_config.get('baseline', 0.4)
                focal_length = camera_config.get('K', [[1000, 0, 0], [0, 1000, 0], [0, 0, 1]])[0][0]
            else:
                # Assume camera_config is a tuple (baseline, focal_length)
                baseline, focal_length = camera_config
            
            # Convert disparity to depth
            depth = np.zeros_like(disparity)
            valid_mask = disparity > 0
            depth[valid_mask] = baseline * focal_length / disparity[valid_mask]
            
            return depth, disparity
        else:
            # Return only disparity if no camera config
            return None, disparity

# Create factory method for easier model instantiation
def create_sgbm_model(min_disp=0, num_disp=192, block_size=11, p1=None, p2=None, 
                      disp12_max_diff=1, uniqueness_ratio=15, speckle_window_size=100,
                      speckle_range=32, pre_filter_cap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
    """Create an SGBM model instance"""
    return SGBMModel(min_disp, num_disp, block_size, p1, p2, disp12_max_diff, 
                    uniqueness_ratio, speckle_window_size, speckle_range, 
                    pre_filter_cap, mode)

# Preset model configurations
def create_sgbm_preset(preset="accurate"):
    """
    Create an SGBM model with a preset configuration
    
    Args:
        preset: Preset configuration ('fast', 'balanced', or 'accurate')
    
    Returns:
        SGBMModel instance
    """
    if preset == "fast":
        return create_sgbm_model(
            num_disp=96,
            block_size=5,
            uniqueness_ratio=5,
            speckle_window_size=50,
            speckle_range=16,
            mode=cv2.STEREO_SGBM_MODE_SGBM
        )
    elif preset == "balanced":
        return create_sgbm_model(
            num_disp=128,
            block_size=9,
            uniqueness_ratio=10,
            speckle_window_size=100,
            speckle_range=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM
        )
    elif preset == "accurate":
        return create_sgbm_model(
            num_disp=192,
            block_size=11,
            uniqueness_ratio=15,
            speckle_window_size=150,
            speckle_range=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

if __name__ == "__main__":
    # Test SGBM model
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SGBM Model')
    parser.add_argument('--left', type=str, required=True, help='Left image path')
    parser.add_argument('--right', type=str, required=True, help='Right image path')
    parser.add_argument('--preset', type=str, default="balanced", 
                        choices=["fast", "balanced", "accurate"], help='SGBM preset')
    args = parser.parse_args()
    
    # Load images
    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    
    if left_img is None or right_img is None:
        print("Error: Could not load images")
        exit(1)
    
    # Convert to RGB
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    # Create model
    model = create_sgbm_preset(args.preset)
    
    # Define camera parameters (example values)
    camera_config = {
        'baseline': 0.4,
        'K': [[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]
    }
    
    # Estimate depth
    depth, disparity = model.estimate_depth(left_img, right_img, camera_config)
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(left_img)
    plt.title('Left Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(right_img)
    plt.title('Right Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(disparity, cmap='jet')
    plt.title('Disparity Map')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(depth, cmap='magma')
    plt.title('Depth Map')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()