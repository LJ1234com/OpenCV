import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

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
