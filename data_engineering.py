"""
Here is refined Quality-Filtered Tiling System for Canopy Height Mapping
We will only saves tiles with meaningful canopy height variation
Includes all necessary library installations
"""
"""
# The codes, follow these princicples
Tile is good quality if:
# 1. Has sufficient variation (not constant)
# 2. Has sufficient pixels with meaningful height
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    packages = {
        'rasterio': 'rasterio',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'gdal': 'osgeo'  # GDAL bindings
    }
    
    print("Installing required packages...")
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f" {package_name} already installed")
        except ImportError:
            print(f"Installing {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f" {package_name} installed successfully")
            except Exception as e:
                print(f" Failed to install {package_name}: {e}")
                if package_name == 'gdal':
                    print("  GDAL requires special installation. Try: conda install -c conda-forge gdal")

# Run installation
install_requirements()

# Now import after installation
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio import windows
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    print("WARNING: rasterio not available. Will use GDAL if available.")
    RASTERIO_AVAILABLE = False

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
    gdal.UseExceptions()
except ImportError:
    print("WARNING: GDAL not available. Will use rasterio if available.")
    GDAL_AVAILABLE = False

if not RASTERIO_AVAILABLE and not GDAL_AVAILABLE:
    raise ImportError("Neither rasterio nor GDAL is available. Please install one of them.")

class QualityFilteredTiler:
    """Advanced tiling system with quality filtering for canopy height data"""
    
    def __init__(self, 
                 tile_size: int = 256,
                 overlap: int = 32,
                 min_height_threshold: float = 0.1,
                 min_height_pixels: float = 0.15,
                 min_variation: float = 0.05,
                 output_format: str = 'PNG'):
        """
        Initialize tiler with quality filtering
        
        Args:
            tile_size: Size of each tile (square)
            overlap: Overlap between tiles
            min_height_threshold: Minimum normalized height to consider as "vegetation" (0-1)
            min_height_pixels: Minimum fraction of pixels above threshold (e.g., 0.15 = 15%)
            min_variation: Minimum standard deviation in height values (0-1)
            output_format: Output format ('PNG' or 'TIFF')
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_height_threshold = min_height_threshold
        self.min_height_pixels = min_height_pixels
        self.min_variation = min_variation
        self.output_format = output_format.upper()
        
        print(f"Quality-Filtered Tiler Initialized:")
        print(f"  Tile size: {tile_size}x{tile_size}")
        print(f"  Overlap: {overlap} pixels")
        print(f"  Min height threshold: {min_height_threshold}")
        print(f"  Min height pixel ratio: {min_height_pixels*100:.1f}%")
        print(f"  Min variation (std): {min_variation}")
        print(f"  Output format: {output_format}")
    
    def assess_tile_quality(self, chm_tile: np.ndarray, chm_min: float, chm_max: float) -> Tuple[bool, dict]:
        """
        Assess if a CHM tile contains meaningful canopy height data
        
        Args:
            chm_tile: CHM tile data (raw values)
            chm_min: Global minimum CHM value for normalization
            chm_max: Global maximum CHM value for normalization
            
        Returns:
            (is_good_quality, quality_metrics)
        """
        # Normalize tile to [0, 1]
        if chm_max > chm_min:
            chm_normalized = (chm_tile - chm_min) / (chm_max - chm_min)
        else:
            chm_normalized = chm_tile
        
        chm_normalized = np.clip(chm_normalized, 0, 1)
        
        # Calculate quality metrics
        mean_height = np.mean(chm_normalized)
        std_height = np.std(chm_normalized)
        max_height = np.max(chm_normalized)
        
        # Count pixels above threshold
        pixels_with_height = np.sum(chm_normalized > self.min_height_threshold)
        height_pixel_ratio = pixels_with_height / chm_normalized.size
        
        quality_metrics = {
            'mean_height': float(mean_height),
            'std_height': float(std_height),
            'max_height': float(max_height),
            'height_pixel_ratio': float(height_pixel_ratio),
            'has_variation': std_height > self.min_variation,
            'has_height': height_pixel_ratio > self.min_height_pixels
        }
        
        # Tile is good quality if:
        # 1. Has sufficient variation (not constant)
        # 2. Has sufficient pixels with meaningful height
        is_good_quality = (quality_metrics['has_variation'] and 
                          quality_metrics['has_height'])
        
        return is_good_quality, quality_metrics
    
    def read_raster_with_rasterio(self, raster_path: str):
        """Read raster using rasterio"""
        with rasterio.open(raster_path) as src:
            data = src.read()
            profile = src.profile
            nodata = src.nodata
            
            # Get statistics from a sample
            sample_window = windows.Window(0, 0, min(1024, src.width), min(1024, src.height))
            sample_data = src.read(window=sample_window, masked=True)
            
            if len(data.shape) == 3 and data.shape[0] == 1:
                data = data[0]
                sample_data = sample_data[0]
            
            # Calculate statistics
            valid_data = sample_data[~sample_data.mask] if hasattr(sample_data, 'mask') else sample_data.flatten()
            valid_data = valid_data[~np.isnan(valid_data)]
            
            if len(valid_data) > 0:
                stats = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
            else:
                stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
            
            return data, profile, nodata, stats
    
    def read_raster_with_gdal(self, raster_path: str):
        """Read raster using GDAL"""
        dataset = gdal.Open(raster_path)
        if dataset is None:
            raise ValueError(f"Could not open {raster_path}")
        
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        
        # Get statistics
        sample_data = data[:min(1024, data.shape[0]), :min(1024, data.shape[1])]
        valid_data = sample_data[~np.isnan(sample_data)]
        if nodata is not None:
            valid_data = valid_data[valid_data != nodata]
        
        if len(valid_data) > 0:
            stats = {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data))
            }
        else:
            stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        profile = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'count': dataset.RasterCount
        }
        
        dataset = None  # Close dataset
        
        return data, profile, nodata, stats
    
    def create_filtered_tiles(self, ortho_path: str, chm_path: str, output_dir: str):
        """Create tiles with quality filtering"""
        
        print("\nCREATING QUALITY-FILTERED TILES")
        print("=" * 60)
        
        # Create output directories
        images_dir = os.path.join(output_dir, 'images')
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Read rasters
        print("Reading orthomosaic...")
        if RASTERIO_AVAILABLE:
            ortho_data, ortho_profile, ortho_nodata, ortho_stats = self.read_raster_with_rasterio(ortho_path)
        else:
            ortho_data, ortho_profile, ortho_nodata, ortho_stats = self.read_raster_with_gdal(ortho_path)
        
        print(f"  Dimensions: {ortho_profile['width']} x {ortho_profile['height']}")
        print(f"  Value range: {ortho_stats['min']:.2f} - {ortho_stats['max']:.2f}")
        
        print("Reading CHM...")
        if RASTERIO_AVAILABLE:
            chm_data, chm_profile, chm_nodata, chm_stats = self.read_raster_with_rasterio(chm_path)
        else:
            chm_data, chm_profile, chm_nodata, chm_stats = self.read_raster_with_gdal(chm_path)
        
        print(f"  Dimensions: {chm_profile['width']} x {chm_profile['height']}")
        print(f"  Value range: {chm_stats['min']:.2f} - {chm_stats['max']:.2f}")
        
        # Ensure dimensions match
        if ortho_data.shape[-2:] != chm_data.shape[-2:]:
            print("ERROR: Orthomosaic and CHM dimensions don't match!")
            return None
        
        # Handle multi-band ortho
        if len(ortho_data.shape) == 3:
            height, width = ortho_data.shape[1], ortho_data.shape[2]
        else:
            height, width = ortho_data.shape[0], ortho_data.shape[1]
            ortho_data = ortho_data[np.newaxis, :, :]  # Add band dimension
        
        # Calculate tile grid
        n_tiles_x = (width - self.overlap) // self.stride
        n_tiles_y = (height - self.overlap) // self.stride
        total_tiles = n_tiles_x * n_tiles_y
        
        print(f"\nProcessing {total_tiles} potential tiles...")
        
        # Process tiles
        saved_tiles = 0
        skipped_low_quality = 0
        skipped_invalid = 0
        quality_metrics_list = []
        
        pbar = tqdm(total=total_tiles, desc="Processing tiles")
        
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # Calculate tile bounds
                x_start = j * self.stride
                y_start = i * self.stride
                x_end = min(x_start + self.tile_size, width)
                y_end = min(y_start + self.tile_size, height)
                
                # Extract tiles
                ortho_tile = ortho_data[:, y_start:y_end, x_start:x_end]
                chm_tile = chm_data[y_start:y_end, x_start:x_end]
                
                # Check for valid data
                if np.all(np.isnan(chm_tile)) or (chm_nodata is not None and np.all(chm_tile == chm_nodata)):
                    skipped_invalid += 1
                    pbar.update(1)
                    continue
                
                # Resize if needed
                if chm_tile.shape != (self.tile_size, self.tile_size):
                    chm_tile = cv2.resize(chm_tile, (self.tile_size, self.tile_size), 
                                        interpolation=cv2.INTER_LINEAR)
                    
                    ortho_tile_resized = np.zeros((ortho_tile.shape[0], self.tile_size, self.tile_size), 
                                                 dtype=ortho_tile.dtype)
                    for band in range(ortho_tile.shape[0]):
                        ortho_tile_resized[band] = cv2.resize(ortho_tile[band], 
                                                              (self.tile_size, self.tile_size),
                                                              interpolation=cv2.INTER_LINEAR)
                    ortho_tile = ortho_tile_resized
                
                # Assess tile quality
                is_good, quality_metrics = self.assess_tile_quality(
                    chm_tile, chm_stats['min'], chm_stats['max']
                )
                
                quality_metrics_list.append(quality_metrics)
                
                if not is_good:
                    skipped_low_quality += 1
                    pbar.update(1)
                    continue
                
                # Normalize data for saving
                # Normalize orthomosaic
                ortho_normalized = np.zeros_like(ortho_tile, dtype=np.float32)
                for band in range(ortho_tile.shape[0]):
                    band_min = ortho_stats['min']
                    band_max = ortho_stats['max']
                    if band_max > band_min:
                        ortho_normalized[band] = (ortho_tile[band] - band_min) / (band_max - band_min)
                    else:
                        ortho_normalized[band] = ortho_tile[band]
                
                ortho_normalized = np.clip(ortho_normalized, 0, 1)
                
                # Normalize CHM
                chm_normalized = (chm_tile - chm_stats['min']) / (chm_stats['max'] - chm_stats['min'])
                chm_normalized = np.clip(chm_normalized, 0, 1)
                
                # Save tile
                tile_id = f"tile_{saved_tiles:04d}"
                
                # Convert to RGB for image
                if ortho_normalized.shape[0] >= 3:
                    ortho_rgb = np.transpose(ortho_normalized[:3], (1, 2, 0))
                else:
                    ortho_rgb = np.stack([ortho_normalized[0]] * 3, axis=-1)
                
                # Convert to 8-bit
                ortho_8bit = (ortho_rgb * 255).astype(np.uint8)
                chm_8bit = (chm_normalized * 255).astype(np.uint8)
                
                # Save
                ortho_save_path = os.path.join(images_dir, f"{tile_id}.png")
                chm_save_path = os.path.join(masks_dir, f"{tile_id}.png")
                
                cv2.imwrite(ortho_save_path, cv2.cvtColor(ortho_8bit, cv2.COLOR_RGB2BGR))
                cv2.imwrite(chm_save_path, chm_8bit)
                
                saved_tiles += 1
                pbar.update(1)
        
        pbar.close()
        
        # Summary statistics
        print(f"\nTILING COMPLETE:")
        print(f"  Total tiles evaluated: {total_tiles}")
        print(f"  Saved (good quality): {saved_tiles}")
        print(f"  Skipped (low quality): {skipped_low_quality}")
        print(f"  Skipped (invalid data): {skipped_invalid}")
        print(f"  Success rate: {saved_tiles/total_tiles*100:.1f}%")
        
        # Save statistics
        stats_summary = {
            'input_files': {
                'orthomosaic': ortho_path,
                'chm': chm_path
            },
            'tiling_parameters': {
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'min_height_threshold': self.min_height_threshold,
                'min_height_pixels': self.min_height_pixels,
                'min_variation': self.min_variation
            },
            'raster_statistics': {
                'orthomosaic': ortho_stats,
                'chm': chm_stats
            },
            'processing_results': {
                'total_evaluated': total_tiles,
                'saved': saved_tiles,
                'skipped_low_quality': skipped_low_quality,
                'skipped_invalid': skipped_invalid
            },
            'quality_metrics_summary': {
                'mean_height_avg': float(np.mean([m['mean_height'] for m in quality_metrics_list])),
                'std_height_avg': float(np.mean([m['std_height'] for m in quality_metrics_list])),
                'height_pixel_ratio_avg': float(np.mean([m['height_pixel_ratio'] for m in quality_metrics_list]))
            }
        }
        
        stats_path = os.path.join(output_dir, 'tiling_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        print(f"\nStatistics saved: {stats_path}")
        
        # Create visualization
        self.create_quality_visualization(images_dir, masks_dir, output_dir)
        
        return stats_summary
    
    def create_quality_visualization(self, images_dir: str, masks_dir: str, output_dir: str):
        """Create visualization showing tile quality"""
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])[:12]
        
        if not image_files:
            print("No tiles to visualize")
            return
        
        n_samples = len(image_files)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 4, n_rows * 6))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif n_rows == 1:
            axes = axes.reshape(2, n_cols)
        elif n_cols == 1:
            axes = axes.reshape(n_rows * 2, 1)
        
        for i, filename in enumerate(image_files):
            row = i // n_cols
            col = i % n_cols
            
            img_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Display image
            axes[row * 2, col].imshow(image)
            axes[row * 2, col].set_title(f'Image: {filename}', fontsize=10)
            axes[row * 2, col].axis('off')
            
            # Display mask with statistics
            mask_norm = mask.astype('float32') / 255.0
            axes[row * 2 + 1, col].imshow(mask, cmap='viridis')
            axes[row * 2 + 1, col].set_title(
                f'CHM: mean={mask_norm.mean():.3f}, std={mask_norm.std():.3f}', 
                fontsize=9
            )
            axes[row * 2 + 1, col].axis('off')
        
        # Hide unused subplots
        for i in range(len(image_files), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row * 2 < axes.shape[0] and col < axes.shape[1]:
                axes[row * 2, col].axis('off')
                axes[row * 2 + 1, col].axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'quality_filtered_tiles_visualization.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {viz_path}")

def main():
    """Main execution function"""
    
    print("QUALITY-FILTERED CANOPY HEIGHT TILING SYSTEM")
    print("=" * 60)
    
    # Your specific paths
    base_dir = r"E:\hunts\Symbiose_Management\for_UAS_data\Carbon_Market\Mine"
    codes_dir = r"E:\hunts\Symbiose_Management\for_UAS_data\Carbon_Market\codes"
    
    ortho_file = "Ortho e502n4785_4786.tif"
    chm_file = "CHM_05.tif"
    
    ortho_path = os.path.join(base_dir, ortho_file)
    chm_path = os.path.join(base_dir, chm_file)
    
    # Output directory
    output_dir = os.path.join(codes_dir, "tiles_quality_filtered")
    
    # Check input files
    if not os.path.exists(ortho_path):
        print(f"ERROR: Orthomosaic not found: {ortho_path}")
        return False
    
    if not os.path.exists(chm_path):
        print(f"ERROR: CHM not found: {chm_path}")
        return False
    
    print(f"\nInput files:")
    print(f"  Orthomosaic: {ortho_path}")
    print(f"  CHM: {chm_path}")
    print(f"  Output: {output_dir}")
    
    # Initialize tiler with quality filtering
    tiler = QualityFilteredTiler(
        tile_size=256,
        overlap=32,
        min_height_threshold=0.1,  # 10% of max height
        min_height_pixels=0.15,    # At least 15% of pixels with height
        min_variation=0.05,         # Standard deviation > 0.05
        output_format='PNG'
    )
    
    # Create tiles
    try:
        stats = tiler.create_filtered_tiles(ortho_path, chm_path, output_dir)
        
        if stats and stats['processing_results']['saved'] > 0:
            print(f"\n SUCCESS!")
            print(f"   Created {stats['processing_results']['saved']} high-quality tiles")
            print(f"   Output directory: {output_dir}")
            print(f"\nNext steps:")
            print(f"  1. Review tiles in: {output_dir}")
            print(f"  2. Run data splitting on these filtered tiles")
            print(f"  3. Train model - should see much better R² scores!")
            return True
        else:
            print(f"\n WARNING: No good quality tiles were created!")
            print(f"   This suggests your CHM data may not contain sufficient canopy height information.")
            print(f"   Consider:")
            print(f"   - Lowering min_height_threshold")
            print(f"   - Lowering min_height_pixels")
            print(f"   - Checking if your CHM file contains actual height data")
            return False
            
    except Exception as e:
        print(f"\nERROR during tiling: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n Quality-filtered tiling completed successfully!")
    else:

        print("\n Tiling encountered issues. Check the messages above.")
