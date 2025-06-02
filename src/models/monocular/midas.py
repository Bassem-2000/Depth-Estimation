#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MiDaS monocular depth estimation model implementation
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class MidasModel:
    """MiDaS monocular depth estimation model"""
    
    def __init__(self, model_type="DPT_Large", device=None):
        """
        Initialize the MiDaS model
        
        Args:
            model_type: Model type to use ('DPT_Large', 'DPT_Hybrid', or 'MiDaS')
            device: Torch device to use
        """
        self.model_type = model_type
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading MiDaS model ({model_type})...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.eval()
        self.model.to(self.device)
        
        # Define transforms based on model type
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:  # MiDaS
            self.transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
            ])
    
    def __call__(self, image):
        """
        Estimate depth from a single image
        
        Args:
            image: Input RGB image (numpy array)
        
        Returns:
            depth_map: Estimated depth map
        """
        return self.estimate_depth(image)
    
    def estimate_depth(self, image):
        """
        Estimate depth from a single image
        
        Args:
            image: Input RGB image (numpy array)
        
        Returns:
            depth_map: Estimated depth map
        """
        # Convert numpy array to PIL Image
        input_image = Image.fromarray(image)
        orig_size = image.shape[:2]
        
        # Apply transform
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
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
        
        return depth_map

# Create factory method for easier model instantiation
def create_midas_model(model_type="DPT_Large", device=None):
    """Create a MiDaS model instance"""
    return MidasModel(model_type, device)

# List of available models
available_models = ["DPT_Large", "DPT_Hybrid", "MiDaS"]

if __name__ == "__main__":
    # Test MiDaS model
    import cv2
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MiDaS Model')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--model_type', type=str, default="DPT_Large", choices=available_models, help='MiDaS model type')
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create model
    model = create_midas_model(args.model_type)
    
    # Estimate depth
    depth = model.estimate_depth(img)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap='magma')
    plt.title('Estimated Depth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()