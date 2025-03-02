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
shift_x = 2
if shift_x > 0:
    depth[:, shift_x:] = depth[:, :-shift_x]
    depth[:shift_x] = 0
else:
    depth[:, :shift_x] = depth[:, :-shift_x]
    depth[:, shift_x:] = 0
shift_y = 2
if shift_y > 0:
    depth[shift_y:] = depth[:-shift_y]
    depth[:shift_y] = 0
else:
    depth[:shift_y] = depth[-shift_y:]
    depth[shift_y:] = 0
depth = cv2.GaussianBlur(depth, (3, 3), 0)
edges = cv2.Canny(depth, 40, 70, 20)
# edges = cv2.dilate(edges, np.ones((3, 3)), iterations=1)
cv2.imwrite('out/depth_edges.jpg', edges)

img_interp = cv2.resize(img_rgb, depth.shape[::-1], interpolation=cv2.INTER_CUBIC)
img_nn = cv2.resize(img_rgb, depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img_sharp = cv2.filter2D(img_nn, -1, kernel)

# ret,thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1000]
# cv2.imwrite('contours.jpg', cv2.drawContours(img_interp.copy(), contours, -1, (0,255,0), 3))
# for cnt in contours:
#     mask = np.zeros(img_sharp.shape, dtype=np.uint8)
#     cv2.drawContours(mask, cnt, -1, (255, 255, 255), -1)
#     sums = np.sum(np.where(mask>0, img_sharp, 0), axis=(0, 1))
#     counts = np.count_nonzero(mask, axis=(0, 1))
#     img_sharp = np.where(mask > 0,
#                          sums / counts,
#                          img_sharp)

img_out = cv2.bitwise_and(img_interp, img_interp, mask=255-edges) + cv2.bitwise_and(img_sharp, img_sharp, mask=edges)
cv2.imwrite('out/canny_sharp.jpg', img_out)

# Blend sharpened image using the edge mask
edges_coloured = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
alpha = 0.3  # Controls the sharpening intensity
output = np.where(edges_coloured>0, img_sharp*alpha + img_interp*(1-alpha), img_interp).astype(np.uint8)
cv2.imwrite('out/canny_sharp_blended.jpg', output)
