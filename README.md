# Satellite Image Upscaler

## Demo
```
git clone https://github.com/William0614/SatelliteImageUpscaler.git
cd SatelliteImageUpscaler
pip installl -r requiremnts.txt
```
Run the jupyter notebook SuperResolutionUpscale.ipynb.

## Implementation
1. Upscale RGB image using 
[bicubic interpolation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
2. Further enhance by upscaling RGB image with 
[super-resolution](https://www.kaggle.com/models/kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2/1?tfhub-redirect=true )
3. Filter enhanced image using joint bilateral filtering.

## Bilateral Joint Filter
For each pixel, assign colour as a mix of the colours of the pixels surrounding it.
Filter size is 3x3. 
Assign weights to filter as the softmax of the negative absolute difference of the depths.
Center pixel has 0 weight.

1. Calculate the absolute depth difference for each pixel in the filter box.
2. Calculate the exponential of the negative depth differences to assign more weights to pixels with higher depth similarity.
3. Assign 0 to center pixel to ignore the target pixel.
4. Normalise the filter.
5. Apply the filter to the RGB image.

## Deep Learning Models
Super Resolution

## Edge Detection
Apply [Canny edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) 
on LiDAR depth image to assign higher weights to edge values and increase the resolution by 
increasing contrast between pixel around edges and the edge pixels.

## Result
[Result](https://github.com/William0614/SatelliteImageUpscaler/blob/main/out/super_bilateral.jpeg) 
can be found in the out folder.

