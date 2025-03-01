import cv2
import numpy as np
import rasterio

# Load RGB satellite images
img_rgb = cv2.imread('data/20230215-SE2B-CGG-GBR-MS3-L3-RGB-preview.jpg')

# Open the LiDAR image using rasterio
with rasterio.open('data/DSM_TQ0075_P_12757_20230109_20230315.tif') as lidar_src:
    depth = lidar_src.read(1)  # Read the first band (assuming single-band image)

depth = np.maximum(depth, 0)
img_nn = cv2.resize(img_rgb, depth.shape[::-1], interpolation=cv2.INTER_NEAREST)

img_up = np.zeros_like(img_nn)
h, w = img_nn.shape[:2]
for i in range(h):
    for j in range(w):
        print(i ,j)
        x1, x2 = max(0, j-1), min(w, j+2)
        y1, y2 = max(0, i-1), min(h, i+2)
        filter = depth[y1: y2, x1:x2] - depth[i, j]
        filter = np.abs(filter)
        filter = np.exp(-filter)
        filter[1, 1] = 0
        filter /= np.sum(filter)
        for c in range(3):
            img_up[i, j, c] = np.sum(img_nn[y1:y2, x1:x2, c] * filter)
cv2.imwrite('out_nn.jpg', img_up)

