# #!/usr/bin/env python3
# """
# Quick Stereo SGBM Test Script
# Simple script to test SGBM stereo matching on your CARLA dataset
# """

# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# import json

# # Step 1: Load camera configuration
# print("Loading camera configuration...")
# data_dir = "data/images"  # 👈 UPDATE THIS PATH

# config_path = os.path.join('data/sensor_config.json')
# if os.path.exists(config_path):
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     baseline = config.get('baseline', 0.4)  # meters
#     focal_length = config['K'][0][0]  # focal length in pixels
#     print(f"✅ Loaded camera config: baseline={baseline}m, focal_length={focal_length}px")
# else:
#     # Default values if no config found
#     baseline = 0.4  # meters
#     focal_length = 658.557  # pixels from your example
#     print(f"⚠️ Using default camera parameters: baseline={baseline}m, focal_length={focal_length}px")

# # Step 2: Load stereo images and ground truth
# print("Loading stereo sample...")
# try:
#     # Get first image
#     left_dir = os.path.join(data_dir, 'left')
#     image_files = sorted([f for f in os.listdir(left_dir) if f.endswith('.png')])
#     first_image = image_files[0]
    
#     # Load left and right images
#     left_path = os.path.join(data_dir, 'left', first_image)
#     right_path = os.path.join(data_dir, 'right', first_image)
    
#     left_img = np.array(Image.open(left_path).convert('RGB'))
#     right_img = np.array(Image.open(right_path).convert('RGB'))
    
#     # Load ground truth depth
#     depth_path = os.path.join(data_dir, 'gt_depth', first_image)
#     depth_img = np.array(Image.open(depth_path))
    
#     # Convert CARLA depth to meters
#     if depth_img.dtype == np.uint16:
#         gt_depth = (depth_img.astype(np.float32) / 65535.0) * 1000.0
#     else:
#         gt_depth = depth_img.astype(np.float32)
    
#     # Load ground truth disparity
#     disp_path = os.path.join(data_dir, 'gt_disp', first_image)
#     disp_img = np.array(Image.open(disp_path))
    
#     # Convert to float disparity in pixels
#     if disp_img.dtype == np.uint16:
#         gt_disp = disp_img.astype(np.float32) / 256.0
#     else:
#         gt_disp = disp_img.astype(np.float32)
    
#     print(f"✅ Loaded stereo images: {first_image}")
#     print(f"Left image shape: {left_img.shape}")
#     print(f"GT depth range: {gt_depth.min():.2f} - {gt_depth.max():.2f} meters")
#     print(f"GT disparity range: {gt_disp.min():.2f} - {gt_disp.max():.2f} pixels")
    
# except Exception as e:
#     print(f"❌ Error loading stereo data: {e}")
#     print("Please check your data_dir path and ensure the folder structure is correct")
#     exit(1)

# # Step 3: Configure and run SGBM
# print("Running SGBM stereo matching...")

# # Convert to grayscale
# left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
# right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

# # Configure SGBM
# window_size = 5
# num_disparities = 128  # Must be divisible by 16

# sgbm = cv2.StereoSGBM_create(
#     minDisparity=0,
#     numDisparities=num_disparities,
#     blockSize=window_size,
#     P1=8 * 3 * window_size**2,
#     P2=32 * 3 * window_size**2,
#     disp12MaxDiff=1,
#     uniquenessRatio=15,
#     speckleWindowSize=100,
#     speckleRange=2,
#     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
# )

# # Compute disparity
# disparity = sgbm.compute(left_gray, right_gray)
# disparity_float = disparity.astype(np.float32) / 16.0  # Convert to float and scale

# print(f"✅ Disparity computed: range {disparity_float.min():.2f} - {disparity_float.max():.2f} pixels")

# # Step 4: Convert disparity to depth
# print("Converting disparity to depth...")

# # Create empty depth map
# depth_map = np.zeros_like(disparity_float)

# # Apply depth = baseline * focal_length / disparity
# valid_mask = disparity_float > 0
# depth_map[valid_mask] = (baseline * focal_length) / disparity_float[valid_mask]

# print(f"✅ Depth map computed: range {depth_map[valid_mask].min():.2f} - {depth_map[valid_mask].max():.2f} meters")

# # Step 5: Compute metrics
# print("Computing metrics...")

# # For disparity
# disp_valid_mask = (gt_disp > 0) & (disparity_float > 0)
# if np.any(disp_valid_mask):
#     disp_abs_error = np.abs(gt_disp[disp_valid_mask] - disparity_float[disp_valid_mask])
#     disp_mae = disp_abs_error.mean()
#     bad_3px = (disp_abs_error > 3.0).mean() * 100
    
#     print(f"Disparity MAE: {disp_mae:.3f} pixels")
#     print(f"Bad 3px: {bad_3px:.2f}%")

# # For depth
# depth_valid_mask = (gt_depth > 0) & (depth_map > 0)
# if np.any(depth_valid_mask):
#     depth_abs_error = np.abs(gt_depth[depth_valid_mask] - depth_map[depth_valid_mask])
#     depth_mae = depth_abs_error.mean()
#     depth_rmse = np.sqrt(np.mean(depth_abs_error**2))
    
#     # Accuracy metric (δ < 1.25)
#     thresh = np.maximum(gt_depth[depth_valid_mask] / depth_map[depth_valid_mask], 
#                        depth_map[depth_valid_mask] / gt_depth[depth_valid_mask])
#     delta1 = (thresh < 1.25).mean()
    
#     print(f"Depth MAE: {depth_mae:.3f} meters")
#     print(f"Depth RMSE: {depth_rmse:.3f} meters")
#     print(f"Depth δ < 1.25: {delta1:.3f}")

# # Step 6: Visualize results
# print("Creating visualization...")

# fig, axes = plt.subplots(3, 2, figsize=(15, 12))
# fig.suptitle(f'Stereo SGBM Results - {first_image}', fontsize=16)

# # Input images
# axes[0, 0].imshow(left_img)
# axes[0, 0].set_title('Left Image')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(right_img)
# axes[0, 1].set_title('Right Image')
# axes[0, 1].axis('off')

# # Disparities
# gt_disp_vis = gt_disp.copy()
# im1 = axes[1, 0].imshow(gt_disp_vis, cmap='plasma')
# axes[1, 0].set_title('Ground Truth Disparity')
# axes[1, 0].axis('off')
# plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

# im2 = axes[1, 1].imshow(disparity_float, cmap='plasma')
# axes[1, 1].set_title('SGBM Disparity')
# axes[1, 1].axis('off')
# plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

# # Depths
# vmax = np.percentile(gt_depth[gt_depth > 0], 95)  # Clip to 95th percentile for better viz
# im3 = axes[2, 0].imshow(gt_depth, cmap='plasma', vmax=vmax)
# axes[2, 0].set_title('Ground Truth Depth')
# axes[2, 0].axis('off')
# plt.colorbar(im3, ax=axes[2, 0], fraction=0.046, pad=0.04)

# depth_vis = depth_map.copy()
# im4 = axes[2, 1].imshow(depth_vis, cmap='plasma', vmax=vmax)
# axes[2, 1].set_title('SGBM Depth')
# axes[2, 1].axis('off')
# plt.colorbar(im4, ax=axes[2, 1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.savefig('stereo_sgbm_result.png', dpi=300, bbox_inches='tight')
# print("📸 Visualization saved to stereo_sgbm_result.png")

# plt.show()

# print("\n🎯 Next steps:")
# print("1. Try different SGBM parameters: block_size, uniquenessRatio, numDisparities")
# print("2. Process multiple images to get average performance metrics")
# print("3. Compare with monocular depth estimation results")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Camera parameters from your configuration
baseline = 0.4  # meters
f = 658.5570007600752  # focal length in pixels

# Function to compute depth from disparity
def disparity_to_depth(disparity):
    # Avoid division by zero
    disparity[disparity == 0] = 0.1
    # Depth = (baseline * focal_length) / disparity
    depth = (baseline * f) / disparity
    return depth

# Load a stereo pair
left_img = cv2.imread('data/images/left/466.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('data/images/right/466.png', cv2.IMREAD_GRAYSCALE)

# Create stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # max disparity - min disparity
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute disparity
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# Convert disparity to depth
depth_map = disparity_to_depth(disparity)

# Visualize results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(left_img, 'gray'), plt.title('Left Image')
plt.subplot(132), plt.imshow(disparity, 'plasma'), plt.title('Disparity Map')
plt.subplot(133), plt.imshow(depth_map, 'plasma'), plt.title('Depth Map')
plt.tight_layout()
plt.show()