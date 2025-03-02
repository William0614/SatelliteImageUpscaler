# Satellite Image Upscaler

## Implementation
1. Upscale RGB image to LiDAR image size using [bicubic interpolation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
2. Enhance upscaled RGB image with [super-resolution]()
3. Filter enhanced image using joint bilateral filtering.

## Bilateral Joint Filter
For each pixel, assign colour as a mix of the colours of the pixel surrounding it.
Filter size 3x3. 
Assign weights to filter as the softmax of the absolute difference of the depths.
Center pixel has 0 weight.
Filter is normalised to sum up to 1.

## Deep Learning Models

Super Resolution

## Result


