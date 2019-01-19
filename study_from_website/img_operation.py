import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

### Read image
img = cv2.imread('nongyu1.jpeg')

cv2.namedWindow('xxx', cv2.WINDOW_NORMAL)
# cv2.namedWindow('xxx', cv2.WINDOW_FREERATIO)
# cv2.namedWindow('xxx', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('xxx', cv2.WINDOW_KEEPRATIO)

cv2.imshow('xxx', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(imutils.opencv2matplotlib(img))
plt.title('xxx')
plt.show()


### Resize image
'''
cv2.resize(src,dsize,dst=None,fx=None,fy=None,interpolation=None)
scr:原图
dsize:输出图像尺寸
fx:沿水平轴的比例因子
fy:沿垂直轴的比例因子
interpolation:插值方法
'''

img = cv2.imread('nongyu1.jpeg')
# resize_img = cv2.resize(img, None, fx=2, fy =2, interpolation=cv2.INTER_CUBIC)

height, width, _ = img.shape
resize_img = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

cv2.imshow('xxx', resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


### image add, subtract, multiply, bitwise_and, bitwise_or, bitwise_not
img1 = cv2.imread('apple.jpeg')
img2 = cv2.imread('orange.jpeg')

# result = cv2.add(img1, img2)
# result = cv2.subtract(img1, img2)
# result = cv2.divide(img1, img2)
# result = cv2.multiply(img1, img2)

# result = cv2.bitwise_and(img1, img2)
# result = cv2.bitwise_or(img1, img2)
result = cv2.bitwise_not(img1)


cv2.imshow('XXX',result)
cv2.waitKey(0)
cv2.destroyAllWindows()







