"""
These are full codes for Enhanced UNet for Canopy Height Training System with Comprehensive Visualization
- Saves feature maps/activations from all layers
- Generates and saves heatmaps
- Exports images, masks, and predictions as TIF (300 DPI o r whateverr you wish based on desired output resolution)
- Organized folder structure
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'matplotlib',
        'pandas',
        'tqdm',
        'scikit-learn',
        'seaborn',
        'pillow'
    ]
    
    print("Checking and installing required packages...")
    for package in packages:
        try:
            __import__(package.replace('-', '_').split('[')[0])
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except Exception as e:
                print(f"⚠ Failed to install {package}: {e}")

# Run installation
print("="*60)
print("INSTALLING DEPENDENCIES")
print("="*60)
install_requirements()

# Now import after installation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import shutil
import random
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# MODIFY THESE PATHS TO YOUR SETUP
# TILES_DIR = r"P:\for_UAS_data\Carbon_Market\codes\tiles_quality_filtered"     #...........Your data(raw-processed_data in their format either .png, .tif or jpeg) path
OUTPUT_DIR = r"P:\for_UAS_data\training_results"

CONFIG = {
    'TILE_SIZE': 256,
    'INPUT_CHANNELS': 3,
    'BATCH_SIZE': 8,
    'LEARNING_RATE': 0.0003,
    'EPOCHS': 300,
    'PATIENCE': 15,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'NUM_WORKERS': 0,
    'PIN_MEMORY': True if torch.cuda.is_available() else False,
    'SAVE_SAMPLE_COUNT': 10,  # Number of samples to visualize
    'DPI': 300,  # DPI for TIF exports
}

print(f"\nDevice: {CONFIG['DEVICE']}")
device = torch.device(CONFIG['DEVICE'])

# ============================================================================
# DATASET CLASS
# ============================================================================

class CanopyHeightDataset(Dataset):
    """Dataset for canopy height regression from quality-filtered tiles"""
    
    def __init__(self, images_dir, masks_dir, augment=False, return_path=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.return_path = return_path
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.png', '.jpg', '.tif'))])
        
        # Verify matching masks exist
        self.valid_pairs = []
        for img_file in self.image_files:
            mask_file = img_file
            mask_path = os.path.join(masks_dir, mask_file)
            img_path = os.path.join(images_dir, img_file)
            
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_path, mask_path, img_file))
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"No valid image-mask pairs found in {images_dir}")
        
        print(f"  Found {len(self.valid_pairs)} valid pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path, filename = self.valid_pairs[idx]
        
        # Load image (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Store original for saving
        original_image = image.copy()
        original_mask = mask.copy()
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        
        # Data augmentation (synchronized for image and mask)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            
            # Random vertical flip
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            
            # Random 90-degree rotations
            if random.random() > 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
            
            # Image-only augmentations
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 1)
            
            if random.random() > 0.5:
                mean = image.mean()
                factor = random.uniform(0.8, 1.2)
                image = np.clip((image - mean) * factor + mean, 0, 1)
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        if self.return_path:
            return image_tensor, mask_tensor, filename, original_image, original_mask
        else:
            return image_tensor, mask_tensor

# ============================================================================
# MODEL ARCHITECTURE WITH FEATURE EXTRACTION
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetCanopyHeight(nn.Module):
    """U-Net with feature extraction capability"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetCanopyHeight, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        # For storing activations
        self.activations = {}
    
    def forward(self, x, save_activations=False):
        # Encoder
        e1 = self.enc1(x)
        if save_activations:
            self.activations['enc1'] = e1
        p1 = self.pool1(e1)
        if save_activations:
            self.activations['pool1'] = p1
        
        e2 = self.enc2(p1)
        if save_activations:
            self.activations['enc2'] = e2
        p2 = self.pool2(e2)
        if save_activations:
            self.activations['pool2'] = p2
        
        e3 = self.enc3(p2)
        if save_activations:
            self.activations['enc3'] = e3
        p3 = self.pool3(e3)
        if save_activations:
            self.activations['pool3'] = p3
        
        e4 = self.enc4(p3)
        if save_activations:
            self.activations['enc4'] = e4
        p4 = self.pool4(e4)
        if save_activations:
            self.activations['pool4'] = p4
        
        # Bottleneck
        b = self.bottleneck(p4)
        if save_activations:
            self.activations['bottleneck'] = b
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        if save_activations:
            self.activations['dec4'] = d4
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        if save_activations:
            self.activations['dec3'] = d3
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        if save_activations:
            self.activations['dec2'] = d2
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        if save_activations:
            self.activations['dec1'] = d1
        
        out = self.out(d1)
        if save_activations:
            self.activations['output'] = out
        
        return out

# ============================================================================
# LOSS FUNCTION
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined MSE and L1 loss"""
    def __init__(self, mse_weight=1.0, l1_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.l1_weight * self.l1(pred, target)

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def save_tif_with_dpi(image_array, filepath, dpi=300):
    """Save image as TIF with specified DPI"""
    # Ensure uint8 format
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_array.shape) == 2:
        img = Image.fromarray(image_array, mode='L')
    else:
        img = Image.fromarray(image_array, mode='RGB')
    
    # Save with DPI info
    img.save(filepath, dpi=(dpi, dpi), compression='tiff_deflate')

def visualize_feature_map(feature_map, layer_name, save_dir):
    """Visualize and save a single feature map"""
    # feature_map shape: (batch, channels, height, width)
    # Take first batch and average across channels
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]  # Take first batch
    
    if feature_map.shape[0] > 1:
        # Average across channels
        feature_map = torch.mean(feature_map, dim=0)
    else:
        feature_map = feature_map[0]
    
    # Convert to numpy
    feature_map = feature_map.cpu().detach().numpy()
    
    # Normalize to 0-255
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    feature_map = (feature_map * 255).astype(np.uint8)
    
    # Save as PNG for visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(feature_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Feature Map: {layer_name}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return feature_map

def create_heatmap_overlay(original_image, heatmap, alpha=0.6):
    """Create heatmap overlay on original image"""
    # Ensure image is RGB uint8
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(original_image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

def save_all_visualizations(model, dataset, save_dir, num_samples=10):
    """Save comprehensive visualizations for selected samples"""
    
    print("\n" + "="*60)
    print("SAVING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Create directory structure
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'images': os.path.join(vis_dir, 'original_images'),
        'masks': os.path.join(vis_dir, 'ground_truth_masks'),
        'predictions': os.path.join(vis_dir, 'predictions'),
        'feature_maps': os.path.join(vis_dir, 'feature_maps'),
        'heatmaps': os.path.join(vis_dir, 'heatmaps'),
        'comparisons': os.path.join(vis_dir, 'comparisons')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    model.eval()
    
    # Select random samples
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"Processing {num_samples} samples...")
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Saving visualizations")):
            # Get data with original images
            if hasattr(dataset, 'return_path') and dataset.return_path:
                image_tensor, mask_tensor, filename, original_image, original_mask = dataset[idx]
            else:
                # Temporarily enable return_path
                old_return_path = dataset.return_path if hasattr(dataset, 'return_path') else False
                dataset.return_path = True
                image_tensor, mask_tensor, filename, original_image, original_mask = dataset[idx]
                dataset.return_path = old_return_path
            
            # Move to device
            image_batch = image_tensor.unsqueeze(0).to(device)
            
            # Forward pass with activation saving
            prediction = model(image_batch, save_activations=True)
            
            # Convert to numpy
            pred_np = prediction[0, 0].cpu().numpy()
            mask_np = mask_tensor[0].cpu().numpy()
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Generate base filename
            base_name = f"sample_{i+1:03d}_{Path(filename).stem}"
            
            # 1. Save original image (TIF, 300 DPI)
            save_tif_with_dpi(
                original_image,
                os.path.join(subdirs['images'], f"{base_name}.tif"),
                dpi=CONFIG['DPI']
            )
            
            # 2. Save ground truth mask (TIF, 300 DPI)
            save_tif_with_dpi(
                original_mask,
                os.path.join(subdirs['masks'], f"{base_name}.tif"),
                dpi=CONFIG['DPI']
            )
            
            # 3. Save prediction (TIF, 300 DPI)
            pred_uint8 = (pred_np * 255).astype(np.uint8)
            save_tif_with_dpi(
                pred_uint8,
                os.path.join(subdirs['predictions'], f"{base_name}.tif"),
                dpi=CONFIG['DPI']
            )
            
            # 4. Save feature maps for key layers
            feature_dir = os.path.join(subdirs['feature_maps'], base_name)
            os.makedirs(feature_dir, exist_ok=True)
            
            key_layers = ['enc1', 'pool1', 'enc2', 'pool2', 'enc3', 'pool3', 
                         'bottleneck', 'dec3', 'dec1', 'output']
            
            for layer_name in key_layers:
                if layer_name in model.activations:
                    visualize_feature_map(
                        model.activations[layer_name],
                        layer_name,
                        feature_dir
                    )
            
            # 5. Save heatmaps
            heatmap_dir = os.path.join(subdirs['heatmaps'], base_name)
            os.makedirs(heatmap_dir, exist_ok=True)
            
            # Create heatmaps for different layers
            for layer_name in ['enc1', 'enc3', 'bottleneck', 'dec1', 'output']:
                if layer_name in model.activations:
                    # Get feature map
                    feat = model.activations[layer_name]
                    if len(feat.shape) == 4:
                        feat = feat[0]
                    if feat.shape[0] > 1:
                        feat = torch.mean(feat, dim=0)
                    else:
                        feat = feat[0]
                    
                    feat_np = feat.cpu().detach().numpy()
                    
                    # Create overlay
                    overlay = create_heatmap_overlay(original_image, feat_np, alpha=0.6)
                    
                    # Save as PNG for heatmap visualization
                    plt.figure(figsize=(10, 10))
                    plt.imshow(overlay)
                    plt.title(f'Heatmap: {layer_name}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(heatmap_dir, f'heatmap_{layer_name}.png'),
                        dpi=150,
                        bbox_inches='tight'
                    )
                    plt.close()
                    
                    # Also save the overlay as TIF
                    save_tif_with_dpi(
                        overlay,
                        os.path.join(heatmap_dir, f'heatmap_{layer_name}.tif'),
                        dpi=CONFIG['DPI']
                    )
            
            # 6. Save comparison figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Ground truth
            axes[0, 1].imshow(original_mask, cmap='viridis')
            axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Prediction
            axes[0, 2].imshow(pred_np, cmap='viridis')
            axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            
            # Feature map (enc1)
            if 'enc1' in model.activations:
                feat = model.activations['enc1'][0]
                feat = torch.mean(feat, dim=0).cpu().numpy()
                axes[1, 0].imshow(feat, cmap='viridis')
                axes[1, 0].set_title('Feature Map: enc1', fontsize=12)
                axes[1, 0].axis('off')
            
            # Feature map (bottleneck)
            if 'bottleneck' in model.activations:
                feat = model.activations['bottleneck'][0]
                feat = torch.mean(feat, dim=0).cpu().numpy()
                axes[1, 1].imshow(feat, cmap='viridis')
                axes[1, 1].set_title('Feature Map: bottleneck', fontsize=12)
                axes[1, 1].axis('off')
            
            # Difference map
            diff = np.abs(pred_np - mask_np)
            im = axes[1, 2].imshow(diff, cmap='hot')
            axes[1, 2].set_title('Absolute Error', fontsize=12)
            axes[1, 2].axis('off')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(subdirs['comparisons'], f'{base_name}_comparison.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Clear activations for next sample
            model.activations = {}
    
    # Create summary report
    summary_path = os.path.join(vis_dir, 'visualization_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("VISUALIZATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples processed: {num_samples}\n")
        f.write(f"Output DPI: {CONFIG['DPI']}\n\n")
        f.write("Directory Structure:\n")
        f.write(f"  - original_images/: Original RGB images (TIF, {CONFIG['DPI']} DPI)\n")
        f.write(f"  - ground_truth_masks/: Ground truth masks (TIF, {CONFIG['DPI']} DPI)\n")
        f.write(f"  - predictions/: Model predictions (TIF, {CONFIG['DPI']} DPI)\n")
        f.write(f"  - feature_maps/: Feature maps from all layers (PNG)\n")
        f.write(f"  - heatmaps/: Heatmap overlays (PNG and TIF)\n")
        f.write(f"  - comparisons/: Side-by-side comparisons (PNG, 300 DPI)\n")
    
    print(f"\n✓ All visualizations saved to: {vis_dir}")
    print(f"  - {num_samples} samples processed")
    print(f"  - Images, masks, predictions: TIF format @ {CONFIG['DPI']} DPI")
    print(f"  - Feature maps and heatmaps: organized by sample")
    
    return vis_dir

# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data(tiles_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split data into train/val/test sets"""
    
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    images_dir = os.path.join(tiles_dir, 'images')
    masks_dir = os.path.join(tiles_dir, 'masks')
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.endswith(('.png', '.jpg', '.tif'))])
    
    print(f"Total images found: {len(image_files)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(image_files)
    
    # Calculate split sizes
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    # Create directories and copy files
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_img_dir = os.path.join(output_dir, split_name, 'images')
        split_mask_dir = os.path.join(output_dir, split_name, 'masks')
        
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        for fname in files:
            # Copy image
            src_img = os.path.join(images_dir, fname)
            dst_img = os.path.join(split_img_dir, fname)
            shutil.copy2(src_img, dst_img)
            
            # Copy mask
            src_mask = os.path.join(masks_dir, fname)
            dst_mask = os.path.join(split_mask_dir, fname)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
    
    print(f"✓ Data split completed: {output_dir}")
    return output_dir

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def calculate_metrics(predictions, targets):
    """Calculate comprehensive metrics"""
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, save_dir):
    """Training loop with comprehensive monitoring"""
    
    history = {
        'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
    }
    
    best_val_r2 = -float('inf')
    patience_counter = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        # for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        # for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, masks = batch[0].to(device), batch[1].to(device)


            # images, masks = images.to(device), masks.to(device)
            
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()
        train_preds.append(outputs.detach())
        train_targets.append(masks.detach())
        
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_metrics = calculate_metrics(train_preds, train_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            # for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                

                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_preds.append(outputs)
                val_targets.append(masks)
        
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_metrics = calculate_metrics(val_preds, val_targets)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_metrics['mae'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['train_r2'].append(train_metrics['r2'])
        
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}, "
              f"RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_metrics['r2'],
                'val_loss': val_loss,
                'config': CONFIG
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"  ✓ New best model saved! R²: {val_metrics['r2']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['PATIENCE']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
    
    return history

def plot_training_history(history, save_dir):
    """Create comprehensive training plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(epochs, history['train_rmse'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_rmse'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # R²
    axes[1, 1].plot(epochs, history['train_r2'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_r2'], 'r-', label='Validation', linewidth=2)
    axes[1, 1].set_title('R² Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline with comprehensive visualization"""
    
    print("\n" + "="*60)
    print("ENHANCED CANOPY HEIGHT MODEL TRAINING")
    print("="*60)
    print(f"Tiles directory: {TILES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Split data
    splits_dir = os.path.join(OUTPUT_DIR, 'data_splits')
    if not os.path.exists(splits_dir):
        splits_dir = split_data(TILES_DIR, splits_dir)
    else:
        print(f"\nUsing existing data splits: {splits_dir}")
    
    # Step 2: Create datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_dataset = CanopyHeightDataset(
        os.path.join(splits_dir, 'train', 'images'),
        os.path.join(splits_dir, 'train', 'masks'),
        augment=True,
        return_path=True
    )
    
    val_dataset = CanopyHeightDataset(
        os.path.join(splits_dir, 'val', 'images'),
        os.path.join(splits_dir, 'val', 'masks'),
        augment=False,
        return_path=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Step 3: Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = UNetCanopyHeight(in_channels=3, out_channels=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Step 4: Setup training
    criterion = CombinedLoss(mse_weight=1.0, l1_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    # Step 5: Train
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        CONFIG['EPOCHS'], OUTPUT_DIR
    )
    
    # Step 6: Save results
    print("\n" + "="*60)
    print("SAVING TRAINING RESULTS")
    print("="*60)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'history': history
    }, os.path.join(OUTPUT_DIR, 'final_model.pth'))
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
    
    # Create plots
    plot_training_history(history, OUTPUT_DIR)
    
    # Save config
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Step 7: Generate comprehensive visualizations
    # Load best model for visualizations
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save visualizations for validation set
    vis_dir = save_all_visualizations(
        model, 
        val_dataset, 
        OUTPUT_DIR, 
        num_samples=CONFIG['SAVE_SAMPLE_COUNT']
    )
    
    # Final summary
    best_r2 = max(history['val_r2'])
    best_mae = min(history['val_mae'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Best Validation R²: {best_r2:.4f}")
    print(f"Best Validation MAE: {best_mae:.4f}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated Files:")
    print("  Models:")
    print("    - best_model.pth (best validation R²)")
    print("    - final_model.pth (final epoch)")
    print("  Metrics:")
    print("    - training_history.csv")
    print("    - training_history.png")
    print("    - config.json")
    print("  Visualizations:")
    print(f"    - {vis_dir}")
    print(f"      ├── original_images/ (TIF @ {CONFIG['DPI']} DPI)")
    print(f"      ├── ground_truth_masks/ (TIF @ {CONFIG['DPI']} DPI)")
    print(f"      ├── predictions/ (TIF @ {CONFIG['DPI']} DPI)")
    print("      ├── feature_maps/ (by sample)")
    print("      ├── heatmaps/ (PNG & TIF)")
    print("      └── comparisons/ (PNG @ 300 DPI)")

if __name__ == "__main__":

    main()
