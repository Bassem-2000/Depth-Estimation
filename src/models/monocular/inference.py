# #!/usr/bin/env python3
# """
# MiDaS Monocular Depth Estimation for CARLA Dataset
# Complete implementation for depth estimation assignment
# """

# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import time
# import json
# from tqdm import tqdm

# class MiDaSDepthEstimator:
#     """MiDaS monocular depth estimation specifically for CARLA data"""
    
#     def __init__(self, model_type='midas_v21', device=None):
#         """
#         Initialize MiDaS model
        
#         Args:
#             model_type: 'midas_v21' (default), 'midas_v21_small', 'dpt_large', 'dpt_hybrid'
#             device: torch device, auto-detected if None
#         """
#         self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model_type = model_type
        
#         print(f"Loading MiDaS model: {model_type}")
#         print(f"Using device: {self.device}")
        
#         # Load model and transform
#         self.model = self._load_model()
#         self.transform = self._load_transform()
        
#         print("✅ MiDaS model loaded successfully!")
    
#     def _load_model(self):
#         """Load MiDaS model from torch hub"""
#         try:
#             if self.model_type == 'midas_v21':
#                 model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
#             elif self.model_type == 'midas_v21_small':
#                 model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
#             elif self.model_type == 'dpt_large':
#                 model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
#             elif self.model_type == 'dpt_hybrid':
#                 model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
#             else:
#                 raise ValueError(f"Unknown model type: {self.model_type}")
            
#             model.to(self.device)
#             model.eval()
#             return model
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             print("Make sure you have internet connection for first-time download")
#             raise
    
#     def _load_transform(self):
#         """Load appropriate transform for the model"""
#         if 'dpt' in self.model_type:
#             transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
#         else:
#             transform = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform
#         return transform
    
#     def predict_depth(self, image, return_confidence=False):
#         """
#         Predict depth from a single RGB image
        
#         Args:
#             image: numpy array (H, W, 3) or PIL Image
#             return_confidence: whether to return confidence map
            
#         Returns:
#             depth_map: numpy array (H, W) with relative depth values
#         """
#         # Convert to PIL if numpy
#         if isinstance(image, np.ndarray):
#             image_pil = Image.fromarray(image)
#         else:
#             image_pil = image
        
#         original_size = image_pil.size  # (width, height)
        
#         # Preprocess
#         input_tensor = self.transform(image_pil).to(self.device)
        
#         # Inference
#         with torch.no_grad():
#             prediction = self.model(input_tensor)
            
#             # Resize to original dimensions
#             prediction = F.interpolate(
#                 prediction.unsqueeze(1),
#                 size=(original_size[1], original_size[0]),  # height, width
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
        
#         # Convert to numpy
#         depth_map = prediction.cpu().numpy()
        
#         return depth_map
    
#     def predict_depth_with_scaling(self, image, gt_depth, scaling_method='median'):
#         """
#         Predict depth and scale to match ground truth statistics
        
#         Args:
#             image: RGB image
#             gt_depth: Ground truth depth for scaling reference
#             scaling_method: 'median', 'mean', or 'least_squares'
            
#         Returns:
#             scaled_depth: Scaled depth map in metric units
#             scale_factor: Applied scale factor
#         """
#         # Get relative depth prediction
#         pred_depth = self.predict_depth(image)
        
#         # Apply scaling to match ground truth
#         scaled_depth, scale_factor = self._apply_scaling(
#             pred_depth, gt_depth, method=scaling_method
#         )
        
#         return scaled_depth, scale_factor
    
#     def _apply_scaling(self, pred_depth, gt_depth, method='median'):
#         """Apply scaling to align predicted depth with ground truth"""
#         # Create valid mask (positive depth values)
#         valid_gt = gt_depth > 0
#         valid_pred = pred_depth > 0
#         valid_mask = valid_gt & valid_pred
        
#         if not np.any(valid_mask):
#             print("Warning: No valid depth values for scaling")
#             return pred_depth, 1.0
        
#         gt_valid = gt_depth[valid_mask]
#         pred_valid = pred_depth[valid_mask]
        
#         # Compute scale factor based on method
#         if method == 'median':
#             scale = np.median(gt_valid) / np.median(pred_valid)
#         elif method == 'mean':
#             scale = np.mean(gt_valid) / np.mean(pred_valid)
#         elif method == 'least_squares':
#             # Least squares fit: gt = scale * pred
#             scale = np.sum(gt_valid * pred_valid) / np.sum(pred_valid ** 2)
#         else:
#             raise ValueError(f"Unknown scaling method: {method}")
        
#         # Apply scaling
#         scaled_depth = pred_depth * scale
        
#         return scaled_depth, scale


# def load_carla_sample(data_dir, sample_idx=0):
#     """Load a specific sample from CARLA dataset"""
#     # Get file list
#     left_dir = os.path.join(data_dir, 'left')
#     left_files = sorted([f for f in os.listdir(left_dir) 
#                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
#     if sample_idx >= len(left_files):
#         raise IndexError(f"Sample {sample_idx} not found. Only {len(left_files)} samples available.")
    
#     file_name = left_files[sample_idx]
    
#     # Load RGB image (use left camera)
#     rgb_path = os.path.join(data_dir, 'left', file_name)
#     rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
    
#     # Load ground truth depth
#     depth_path = os.path.join(data_dir, 'gt_depth', file_name)
#     if depth_path.endswith('.png'):
#         depth_img = np.array(Image.open(depth_path))
#         if depth_img.dtype == np.uint16:
#             # Convert CARLA depth encoding to meters
#             gt_depth = (depth_img.astype(np.float32) / 65535.0) * 1000.0
#         else:
#             gt_depth = depth_img.astype(np.float32)
#     else:
#         gt_depth = np.load(depth_path)
    
#     return rgb_image, gt_depth, file_name


# def compute_depth_metrics(gt_depth, pred_depth):
#     """Compute standard depth estimation metrics"""
#     # Valid mask
#     valid_mask = (gt_depth > 0) & (pred_depth > 0)
    
#     if not np.any(valid_mask):
#         return None
    
#     gt_valid = gt_depth[valid_mask]
#     pred_valid = pred_depth[valid_mask]
    
#     # Absolute relative error
#     abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    
#     # Squared relative error  
#     sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
#     # RMSE
#     rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    
#     # RMSE log
#     rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
#     # Accuracy thresholds
#     thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()
    
#     return {
#         'abs_rel': abs_rel,
#         'sq_rel': sq_rel,
#         'rmse': rmse,
#         'rmse_log': rmse_log,
#         'a1': a1,
#         'a2': a2,
#         'a3': a3
#     }


# def visualize_results(rgb_image, gt_depth, pred_depth, save_path=None):
#     """Create visualization comparing ground truth and predicted depth"""
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle('MiDaS Depth Estimation Results', fontsize=16)
    
#     # RGB input
#     axes[0, 0].imshow(rgb_image)
#     axes[0, 0].set_title('Input RGB Image')
#     axes[0, 0].axis('off')
    
#     # Ground truth depth
#     gt_vis = gt_depth.copy()
#     gt_vis[gt_vis == 0] = np.nan
#     im1 = axes[0, 1].imshow(gt_vis, cmap='plasma')
#     axes[0, 1].set_title('Ground Truth Depth (m)')
#     axes[0, 1].axis('off')
#     plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
#     # Predicted depth
#     pred_vis = pred_depth.copy()
#     pred_vis[pred_vis == 0] = np.nan
#     im2 = axes[1, 0].imshow(pred_vis, cmap='plasma')
#     axes[1, 0].set_title('MiDaS Predicted Depth (m)')
#     axes[1, 0].axis('off')
#     plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
#     # Error map
#     valid_mask = (gt_depth > 0) & (pred_depth > 0)
#     error_map = np.abs(gt_depth - pred_depth)
#     error_map[~valid_mask] = np.nan
#     im3 = axes[1, 1].imshow(error_map, cmap='hot')
#     axes[1, 1].set_title('Absolute Error (m)')
#     axes[1, 1].axis('off')
#     plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {save_path}")
    
#     return fig


# def run_midas_experiment(data_dir, output_dir, num_samples=None):
#     """Run complete MiDaS experiment on CARLA dataset"""
#     print("🚀 Starting MiDaS Depth Estimation Experiment")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize MiDaS
#     estimator = MiDaSDepthEstimator(model_type='midas_v21')
    
#     # Get sample list
#     left_dir = os.path.join(data_dir, 'left')
#     left_files = sorted([f for f in os.listdir(left_dir) 
#                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
#     if num_samples:
#         left_files = left_files[:num_samples]
    
#     print(f"Processing {len(left_files)} samples...")
    
#     # Storage for results
#     all_metrics = []
#     inference_times = []
#     scale_factors = []
    
#     # Process each sample
#     for i, file_name in enumerate(tqdm(left_files, desc="Processing samples")):
#         try:
#             # Load sample
#             rgb_image, gt_depth, _ = load_carla_sample(data_dir, i)
            
#             # Measure inference time
#             start_time = time.time()
#             pred_depth, scale_factor = estimator.predict_depth_with_scaling(rgb_image, gt_depth)
#             inference_time = time.time() - start_time
            
#             inference_times.append(inference_time)
#             scale_factors.append(scale_factor)
            
#             # Compute metrics
#             metrics = compute_depth_metrics(gt_depth, pred_depth)
#             if metrics:
#                 all_metrics.append(metrics)
            
#             # Save visualization for first few samples
#             if i < 5:
#                 vis_path = os.path.join(output_dir, f'result_{i:04d}.png')
#                 visualize_results(rgb_image, gt_depth, pred_depth, vis_path)
#                 plt.close()
                
#         except Exception as e:
#             print(f"Error processing sample {i}: {e}")
#             continue
    
#     # Aggregate results
#     if all_metrics:
#         avg_metrics = {}
#         for key in all_metrics[0].keys():
#             avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        
#         # Add timing and scaling statistics
#         avg_metrics['avg_inference_time'] = float(np.mean(inference_times))
#         avg_metrics['std_inference_time'] = float(np.std(inference_times))
#         avg_metrics['avg_scale_factor'] = float(np.mean(scale_factors))
#         avg_metrics['std_scale_factor'] = float(np.std(scale_factors))
        
#         # Save results
#         results_path = os.path.join(output_dir, 'midas_results.json')
#         with open(results_path, 'w') as f:
#             json.dump(avg_metrics, f, indent=4)
        
#         # Print summary
#         print("\n📊 Results Summary:")
#         print(f"Absolute Relative Error: {avg_metrics['abs_rel']:.4f}")
#         print(f"RMSE: {avg_metrics['rmse']:.4f} meters")
#         print(f"δ < 1.25: {avg_metrics['a1']:.4f}")
#         print(f"Average inference time: {avg_metrics['avg_inference_time']:.4f} seconds")
#         print(f"Average scale factor: {avg_metrics['avg_scale_factor']:.2f}")
        
#         print(f"\n✅ Experiment complete! Results saved to {output_dir}")
#         return avg_metrics
#     else:
#         print("❌ No valid results obtained")
#         return None


# def main():
#     # Configuration
#     data_dir = "/path/to/your/carla/dataset"  # Update this path
#     output_dir = "./midas_results"
#     num_samples = 10  # Set to None to process all samples
    
#     # Quick test with single sample
#     print("🧪 Testing MiDaS on single sample...")
#     try:
#         rgb_image, gt_depth, file_name = load_carla_sample(data_dir, 0)
#         estimator = MiDaSDepthEstimator()
#         pred_depth, scale = estimator.predict_depth_with_scaling(rgb_image, gt_depth)
        
#         metrics = compute_depth_metrics(gt_depth, pred_depth)
#         print(f"✅ Test successful! Sample metrics: {metrics}")
        
#         # Show visualization
#         visualize_results(rgb_image, gt_depth, pred_depth)
#         plt.show()
        
#     except Exception as e:
#         print(f"❌ Test failed: {e}")
#         print("Please check your data_dir path and dataset structure")
#         return
    
#     # Run full experiment
#     response = input("\nRun full experiment? (y/n): ").lower()
#     if response == 'y':
#         results = run_midas_experiment(data_dir, output_dir, num_samples)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Depth Estimation Evaluation Script (Fixed Version)

This script implements and compares monocular and stereo depth estimation 
methods on the CARLA simulator dataset with image-based ground truth.
"""

import os
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import csv

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(dataset_path, num_samples=None):
    """
    Load dataset images and ground truth from image files
    
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
        file_ext = os.path.splitext(filename)[1]
        
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
        # Assuming same file extension for ground truth
        gt_depth_path = os.path.join(dataset_path, 'gt_depth', filename)
        gt_disp_path = os.path.join(dataset_path, 'gt_disp', filename)
        
        # Read the ground truth images
        gt_depth_img = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        gt_disp_img = cv2.imread(gt_disp_path, cv2.IMREAD_UNCHANGED)
        
        # Check if image loading was successful
        if gt_depth_img is None:
            print(f"Warning: Could not load depth ground truth: {gt_depth_path}")
            # Try different extensions if needed
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                alt_path = os.path.join(dataset_path, 'gt_depth', base_name + ext)
                if os.path.exists(alt_path):
                    gt_depth_img = cv2.imread(alt_path, cv2.IMREAD_UNCHANGED)
                    if gt_depth_img is not None:
                        print(f"  Found alternative at {alt_path}")
                        break
            
            # If still None, create a dummy array
            if gt_depth_img is None:
                print(f"  Creating dummy depth for {filename}")
                gt_depth_img = np.zeros_like(left_img[:,:,0], dtype=np.uint8)
        
        if gt_disp_img is None:
            print(f"Warning: Could not load disparity ground truth: {gt_disp_path}")
            # Try different extensions if needed
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                alt_path = os.path.join(dataset_path, 'gt_disp', base_name + ext)
                if os.path.exists(alt_path):
                    gt_disp_img = cv2.imread(alt_path, cv2.IMREAD_UNCHANGED)
                    if gt_disp_img is not None:
                        print(f"  Found alternative at {alt_path}")
                        break
            
            # If still None, create a dummy array
            if gt_disp_img is None:
                print(f"  Creating dummy disparity for {filename}")
                gt_disp_img = np.zeros_like(left_img[:,:,0], dtype=np.uint8)
        
        # Convert ground truth images to actual depth/disparity values
        gt_depth = gt_depth_img.astype(np.float32)
        gt_disp = gt_disp_img.astype(np.float32)
        
        # If ground truth is RGB, might need to extract specific channel or apply formula
        if len(gt_depth.shape) == 3:
            # Use only one channel - in this case, we'll use the first channel
            gt_depth = gt_depth[:,:,0]
        
        if len(gt_disp.shape) == 3:
            gt_disp = gt_disp[:,:,0]
        
        # Print some statistics for the first image to help debug
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

# Monocular Depth Estimation Models
def load_midas():
    """Load MiDaS model for monocular depth estimation"""
    print("Loading MiDaS model...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model.eval()
    model.to(device)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform

def estimate_monocular_depth(model, image, transform, orig_size):
    """
    Estimate depth from a single image using MiDaS
    
    Args:
        model: MiDaS model
        image: Input RGB image (numpy array)
        transform: Preprocessing transform
        orig_size: Original image size (H, W)
    
    Returns:
        depth_map: Estimated depth map
    """
    # Convert numpy array to PIL Image
    input_image = Image.fromarray(image)
    
    # Apply transform
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
        # Resize to original resolution
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=orig_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy
    depth_map = prediction.cpu().numpy()
    
    # MiDaS outputs inverse depth, convert to metric depth
    # This requires calibration for each dataset
    depth_map = 1.0 / (depth_map + 1e-6)
    
    # Normalize to reasonable range (since we don't know the exact scaling)
    return depth_map

# Stereo Depth Estimation Methods
def setup_sgbm():
    """Setup Semi-Global Block Matching for stereo depth estimation"""
    window_size = 11
    min_disp = 0
    num_disp = 192 - min_disp
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    return stereo

def estimate_stereo_depth(stereo_matcher, left_img, right_img, camera_config):
    """
    Estimate depth using stereo matching
    
    Args:
        stereo_matcher: SGBM stereo matcher
        left_img: Left RGB image
        right_img: Right RGB image
        camera_config: Camera configuration
    
    Returns:
        depth_map: Estimated depth map
        disparity_map: Estimated disparity map
    """
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    
    # Compute disparity
    disparity = stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Extract camera parameters
    baseline = camera_config['baseline']
    focal_length = camera_config['K'][0][0]  # fx
    
    # Convert disparity to depth
    depth = np.zeros_like(disparity)
    valid_mask = disparity > 0
    depth[valid_mask] = baseline * focal_length / disparity[valid_mask]
    
    return depth, disparity

# Evaluation metrics
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
    
    # Ensure there are enough valid pixels for meaningful evaluation
    if len(pred) < 10:
        return {metric: np.nan for metric in 
                ['abs_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    
    # Scale prediction to have the same median as ground truth
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
    
    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }

def visualize_results(img, gt_depth, mono_depth, stereo_depth, filename=None):
    """
    Create visualization of depth estimation results
    
    Args:
        img: RGB image
        gt_depth: Ground truth depth map
        mono_depth: Monocular depth estimation
        stereo_depth: Stereo depth estimation
        filename: Output filename for saving visualization
    """
    # Scale monocular depth to have the same median as ground truth
    valid_mask = gt_depth > 0
    if np.any(valid_mask) and np.any(mono_depth > 0):
        scale = np.median(gt_depth[valid_mask]) / np.median(mono_depth[mono_depth > 0])
        mono_depth_scaled = mono_depth * scale
    else:
        mono_depth_scaled = mono_depth
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define common depth color map and range
    cmap = plt.cm.magma
    
    # Calculate a reasonable vmax based on non-zero depth values
    if np.any(gt_depth > 0):
        vmax = np.percentile(gt_depth[gt_depth > 0], 95)
    else:
        vmax = np.max(gt_depth) if np.max(gt_depth) > 0 else 10.0
    
    # RGB image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # Ground truth depth
    im_gt = axes[0, 1].imshow(gt_depth, cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 1].set_title('Ground Truth Depth')
    axes[0, 1].axis('off')
    
    # Monocular depth
    axes[0, 2].imshow(mono_depth_scaled, cmap=cmap, vmin=0, vmax=vmax)
    axes[0, 2].set_title('Monocular Depth')
    axes[0, 2].axis('off')
    
    # Stereo depth
    axes[1, 0].imshow(stereo_depth, cmap=cmap, vmin=0, vmax=vmax)
    axes[1, 0].set_title('Stereo Depth')
    axes[1, 0].axis('off')
    
    # Error maps
    mono_error = np.abs(mono_depth_scaled - gt_depth)
    stereo_error = np.abs(stereo_depth - gt_depth)
    
    # Calculate error range
    if np.any(gt_depth > 0):
        error_vmax = max(
            np.percentile(mono_error[gt_depth > 0], 95) if np.any(mono_error[gt_depth > 0]) else 1.0,
            np.percentile(stereo_error[gt_depth > 0], 95) if np.any(stereo_error[gt_depth > 0]) else 1.0
        )
    else:
        error_vmax = 1.0
    
    axes[1, 1].imshow(mono_error, cmap='hot', vmin=0, vmax=error_vmax)
    axes[1, 1].set_title('Monocular Error')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(stereo_error, cmap='hot', vmin=0, vmax=error_vmax)
    axes[1, 2].set_title('Stereo Error')
    axes[1, 2].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax)
    cbar.set_label('Depth (m)')
    
    # Save figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Depth Estimation Evaluation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Print some info about the data directory to help debug
    if args.debug:
        print(f"Data path: {args.data_path}")
        for subdir in ['left', 'right', 'gt_depth', 'gt_disp']:
            path = os.path.join(args.data_path, subdir)
            if os.path.exists(path):
                files = os.listdir(path)
                if files:
                    first_file = os.path.join(path, files[0])
                    if os.path.isfile(first_file):
                        try:
                            img = cv2.imread(first_file, cv2.IMREAD_UNCHANGED)
                            print(f"{subdir}: {len(files)} files, first file shape: {img.shape}, dtype: {img.dtype}")
                        except:
                            print(f"{subdir}: {len(files)} files, but couldn't read first file")
                    else:
                        print(f"{subdir}: {len(files)} files (first item is not a file)")
                else:
                    print(f"{subdir}: directory exists but is empty")
            else:
                print(f"{subdir}: directory not found")
    
    # Load dataset
    try:
        dataset, camera_config = load_dataset(args.data_path, args.num_samples)
        print(f"Successfully loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load models
    try:
        midas_model, midas_transform = load_midas()
        print("MiDaS model loaded successfully")
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        return
    
    stereo_matcher = setup_sgbm()
    print("SGBM stereo matcher set up successfully")
    
    # Initialize metrics storage
    mono_metrics = []
    stereo_metrics = []
    
    # Process each sample
    print("Processing samples...")
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
        
        # Monocular depth estimation
        mono_depth = estimate_monocular_depth(
            midas_model, left_img, midas_transform, left_img.shape[:2]
        )
        
        # Stereo depth estimation
        stereo_depth, stereo_disp = estimate_stereo_depth(
            stereo_matcher, left_img, right_img, camera_config
        )
        
        # Evaluate
        mono_result = evaluate_depth(mono_depth, gt_depth)
        stereo_result = evaluate_depth(stereo_depth, gt_depth)
        
        # Add filename to metrics for reference
        mono_result['filename'] = filename
        stereo_result['filename'] = filename
        
        mono_metrics.append(mono_result)
        stereo_metrics.append(stereo_result)
        
        # Visualize
        vis_path = os.path.join(args.output_dir, 'visualizations', 
                               f'{os.path.splitext(filename)[0]}.png')
        try:
            visualize_results(left_img, gt_depth, mono_depth, stereo_depth, vis_path)
        except Exception as e:
            print(f"Error visualizing results for {filename}: {e}")
        
        # Optional: Save individual depth maps
        if args.debug:
            # Save depth maps as normalized 16-bit PNGs for detailed inspection
            mono_dir = os.path.join(args.output_dir, 'mono_depth')
            stereo_dir = os.path.join(args.output_dir, 'stereo_depth')
            os.makedirs(mono_dir, exist_ok=True)
            os.makedirs(stereo_dir, exist_ok=True)
            
            # Normalize and save
            def save_normalized_depth(depth_map, save_path):
                # Clip to 99th percentile to handle outliers
                valid = depth_map > 0
                if np.any(valid):
                    # Scale to 0-65535 for 16-bit PNG
                    v_max = np.percentile(depth_map[valid], 99)
                    normalized = np.clip(depth_map, 0, v_max) / v_max * 65535
                    # Convert to 16-bit unsigned int
                    normalized = normalized.astype(np.uint16)
                    cv2.imwrite(save_path, normalized)
            
            mono_save_path = os.path.join(mono_dir, f'{os.path.splitext(filename)[0]}.png')
            stereo_save_path = os.path.join(stereo_dir, f'{os.path.splitext(filename)[0]}.png')
            
            try:
                save_normalized_depth(mono_depth, mono_save_path)
                save_normalized_depth(stereo_depth, stereo_save_path)
            except Exception as e:
                print(f"Error saving depth maps for {filename}: {e}")
    
    # Filter out NaN values for metric calculation
    def safe_mean(values):
        """Calculate mean of values, filtering out NaNs"""
        filtered = [v for v in values if not np.isnan(v)]
        if filtered:
            return np.mean(filtered)
        return float('nan')
    
    # Compute average metrics
    mono_avg = {metric: safe_mean([m[metric] for m in mono_metrics])
                for metric in ['abs_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    stereo_avg = {metric: safe_mean([m[metric] for m in stereo_metrics])
                for metric in ['abs_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
    
    # Print results
    print("\nMonocular Depth Estimation Results:")
    for k, v in mono_avg.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: nan")
    
    print("\nStereo Depth Estimation Results:")
    for k, v in stereo_avg.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: nan")
    
    # Create a comparison table
    comparison = {
        "Method": ["Monocular (MiDaS)", "Stereo (SGBM)"],
        "Abs Rel Error": [f"{mono_avg['abs_rel']:.4f}" if not np.isnan(mono_avg['abs_rel']) else "N/A", 
                          f"{stereo_avg['abs_rel']:.4f}" if not np.isnan(stereo_avg['abs_rel']) else "N/A"],
        "RMSE": [f"{mono_avg['rmse']:.4f}" if not np.isnan(mono_avg['rmse']) else "N/A", 
                 f"{stereo_avg['rmse']:.4f}" if not np.isnan(stereo_avg['rmse']) else "N/A"],
        "RMSE log": [f"{mono_avg['rmse_log']:.4f}" if not np.isnan(mono_avg['rmse_log']) else "N/A", 
                     f"{stereo_avg['rmse_log']:.4f}" if not np.isnan(stereo_avg['rmse_log']) else "N/A"],
        "delta < 1.25": [f"{mono_avg['a1']:.4f}" if not np.isnan(mono_avg['a1']) else "N/A", 
                         f"{stereo_avg['a1']:.4f}" if not np.isnan(stereo_avg['a1']) else "N/A"],
        "delta < 1.25^2": [f"{mono_avg['a2']:.4f}" if not np.isnan(mono_avg['a2']) else "N/A", 
                           f"{stereo_avg['a2']:.4f}" if not np.isnan(stereo_avg['a2']) else "N/A"],
        "delta < 1.25^3": [f"{mono_avg['a3']:.4f}" if not np.isnan(mono_avg['a3']) else "N/A", 
                           f"{stereo_avg['a3']:.4f}" if not np.isnan(stereo_avg['a3']) else "N/A"]
    }
    
    # Save comparison table as CSV
    with open(os.path.join(args.output_dir, 'comparison.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(comparison.keys())
        writer.writerows(zip(*comparison.values()))
    
    # Save detailed results to file
    results = {
        'monocular': {k: float(v) if not np.isnan(v) else None for k, v in mono_avg.items()},
        'stereo': {k: float(v) if not np.isnan(v) else None for k, v in stereo_avg.items()},
        'per_sample': {
            'monocular': [{k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else None 
                          for k, v in m.items() if k != 'filename'} for m in mono_metrics],
            'stereo': [{k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else None 
                       for k, v in m.items() if k != 'filename'} for m in stereo_metrics]
        }
    }
    
    try:
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving metrics.json: {e}")
        # Try a simpler approach
        with open(os.path.join(args.output_dir, 'metrics_simple.json'), 'w') as f:
            json.dump({
                'monocular': {k: str(v) for k, v in mono_avg.items()},
                'stereo': {k: str(v) for k, v in stereo_avg.items()}
            }, f, indent=4)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == '__main__':
    main()