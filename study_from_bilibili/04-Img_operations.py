import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('nongyu1.jpeg', cv2.IMREAD_UNCHANGED)

px = img[55, 55]
print(px)

roi = img[100:200, 100:200]
print(roi)

img[100:200, 100:200] = [225, 225, 225]

img[300:500, 800:1000] = img[200:400, 400:600]

cv2.imshow('XXX', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
