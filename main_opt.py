import cv2
import numpy as np
import rasterio
from scipy.ndimage import convolve

# Load RGB satellite images
img_rgb = cv2.imread('data/20230215-SE2B-CGG-GBR-MS3-L3-RGB-preview.jpg')

# Open the LiDAR image using rasterio
with rasterio.open('data/DSM_TQ0075_P_12757_20230109_20230315.tif') as lidar_src:
    img_lidar = lidar_src.read(1)  # Read the first band (assuming single-band image)

# Resize RGB image to match LiDAR dimensions
img_interp = cv2.resize(img_rgb, (img_lidar.shape[1], img_lidar.shape[0]), interpolation=cv2.INTER_CUBIC)

# Ensure LiDAR values are non-negative
img_lidar = np.maximum(img_lidar, 0)

# Define kernel size (window around each pixel)
kernel_size = 11  # 5 pixels on each side (5*2 + 1)

# Compute local differences
def local_filter(lidar):
    center_value = lidar[kernel_size // 2, kernel_size // 2]
    diff = np.abs(lidar - center_value)
    if np.sum(diff) == 0:
        return np.ones_like(diff) / diff.size  # Prevent division by zero
    return diff / np.sum(diff)

# Apply local filter to LiDAR image
filtered_lidar = convolve(img_lidar, np.ones((kernel_size, kernel_size)), mode='reflect')
norm_weights = np.abs(filtered_lidar - img_lidar[:, :, None])  # Broadcasting

# Normalize weights
if np.sum(norm_weights) == 0:
    norm_weights = np.ones_like(norm_weights)  # Prevent division by zero
norm_weights /= np.sum(norm_weights, axis=(1, 2), keepdims=True)

# Apply filter to each color channel
img_up = np.zeros_like(img_interp, dtype=np.float32)
for c in range(3):
    img_up[:, :, c] = convolve(img_interp[:, :, c], np.ones((kernel_size, kernel_size)), mode='reflect')

# Normalize final output
img_up = np.clip(img_up, 0, 255).astype(np.uint8)

# Save the output image
cv2.imwrite('out.jpg', img_up)
