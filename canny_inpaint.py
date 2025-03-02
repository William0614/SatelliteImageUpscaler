import cv2
import numpy as np
import rasterio

# Load RGB satellite images
img_rgb = cv2.imread('data/20230215-SE2B-CGG-GBR-MS3-L3-RGB-preview.jpg')

# Open the LiDAR image using rasterio
with rasterio.open('data/DSM_TQ0075_P_12757_20230109_20230315.tif') as lidar_src:
    depth = lidar_src.read(1)  # Read the first band (assuming single-band image)

depth = np.maximum(depth, 0)
depth = depth / np.max(depth) * 255
depth = depth.astype(np.uint8)
depth = cv2.GaussianBlur(depth, (3, 3), 0)
edges = cv2.Canny(depth, 50, 70, 20)
cv2.imwrite('out/depth_edges.jpg', edges)

img_nn = cv2.resize(img_rgb, depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
edges_downscaled = cv2.resize(edges, img_rgb.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
img_inpaint = cv2.resize(cv2.inpaint(img_rgb, edges_downscaled, 1, cv2.INPAINT_NS), depth.shape[::-1], interpolation=cv2.INTER_CUBIC)
img_out = cv2.bitwise_and(img_nn, img_nn, mask=edges) + cv2.bitwise_and(img_inpaint, img_inpaint, mask=255-edges)
cv2.imwrite('out/canny_inpaint.jpg', img_out)

