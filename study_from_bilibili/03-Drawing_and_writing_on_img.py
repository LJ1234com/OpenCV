import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('nongyu1.jpeg', cv2.IMREAD_UNCHANGED)

cv2.line(img, (0, 0), (150, 150), (225, 225, 225), 15)
cv2.rectangle(img, (30, 30), (100, 100), (0, 225, 0), 5)
cv2.circle(img, (200, 200), 100, (0, 0, 225), -1)

pts = np.array([[100, 100], [200, 300], [100, 400], [50,300]])
cv2.polylines(img, [pts], True, (0, 225, 225), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Draw Line', (300, 300), font, 1, (200, 225, 225), 2, cv2.LINE_AA)
# 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细


cv2.imshow('XXX', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
