import cv2
import numpy as np
import rasterio
import math

# Load RGB satellite images
img_rgb = cv2.imread('data/20230215-SE2B-CGG-GBR-MS3-L3-RGB-preview.jpg')

# Open the LiDAR image using rasterio
with rasterio.open('data/DSM_TQ0075_P_12757_20230109_20230315.tif') as lidar_src:
    depth = lidar_src.read(1)  # Read the first band (assuming single-band image)

depth = np.maximum(depth, 0)


def gauss(img, spatialKern, rangeKern):
    gaussianSpatial = 1 / math.sqrt(2 * math.pi * (
                spatialKern ** 2))  # gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    gaussianRange = 1 / math.sqrt(2 * math.pi * (rangeKern ** 2))  # gaussian function to calcualte the range kernel
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)
    xx = -spatialKern + np.arange(2 * spatialKern + 1)
    yy = -spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx, yy)
    spatialGS = gaussianSpatial * np.exp(-(x ** 2 + y ** 2) / (2 * (
                gaussianSpatial ** 2)))  # calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2)
    return matrix, spatialGS


def padImage(img, spatialKern):  # pad array with mirror reflections of itself.
    img = np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')
    return img


def jointBilateralFilter(img, depth, spatialKern, rangeKern):
    ch = img.shape[2]
    h, w = depth.shape[:2]
    orgImg = padImage(img, spatialKern)  # pad image with no flash
    secondImg = padImage(depth, spatialKern)  # pad image with flash
    matrix, spatialGS = gauss(img, spatialKern, rangeKern)  # apply gaussian function

    outputImg = np.zeros((h, w, ch), np.uint8)  # create a matrix the size of the image
    summ = 1
    for x in range(spatialKern, spatialKern + h):
        for y in range(spatialKern, spatialKern + w):
            for i in range(ch):  # iterate through the image's height, width and channel
                # apply the equation that is mentioned in the pdf file
                neighbourhood = secondImg[x - spatialKern: x + spatialKern + 1, y - spatialKern: y + spatialKern + 1]  # get neighbourhood of pixels
                central = secondImg[x, y]  # get central pixel
                res = matrix[abs(neighbourhood - central)]  # subtract them
                summ = summ * res * spatialGS  # multiply them with the spatial kernel
                norm = np.sum(res)  # normalization term
                outputImg[x - spatialKern, y - spatialKern, i] = np.sum(
                    res * orgImg[x - spatialKern: x + spatialKern + 1, y - spatialKern: y + spatialKern + 1,
                          i]) / norm  # apply full equation of JBF(img,img1)
    return outputImg


spatialKern = 30  # 27
rangeKern = 20  # 10
filteredimg = jointBilateralFilter(img_rgb, depth, spatialKern, rangeKern)
cv2.imshow('input', img_rgb)  # show original no flash image
cv2.imshow('JointBilateralFilter', filteredimg)  # show image after joint bilateral filter is applied