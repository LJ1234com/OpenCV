import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv2.IMREAD_GRAYSCALE <=> 0
# cv2.IMREAD_COLOR <=> 1
# cv2.IMREAD_UNCHANGED <=> -1
img = cv2.imread('nongyu1.jpeg', cv2.IMREAD_UNCHANGED)

# cv2.imshow('XXX', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()

cv2.imwrite('img_gray', img)
