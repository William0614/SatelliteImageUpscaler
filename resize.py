import cv2


im = cv2.imread('data/20230215-SE2B-CGG-GBR-MS3-L3-RGB-preview.jpg')
h, w = 5000, 5000
cv2.imwrite('out/inter_cubic.jpg', cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC))
