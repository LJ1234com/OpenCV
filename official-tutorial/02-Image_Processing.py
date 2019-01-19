import numpy as np
import cv2
import matplotlib.pyplot as plt


##################### Changing Colorspaces #######################

##################### Geometric Transformations of Images #######################

##################### Image Thresholding #######################

##################### Smoothing Images #######################

##################### Morphological Transformations #######################

##################### Image Gradients ##################### 

##################### Canny Edge Detection ##################### 
img = cv2.imread('nongyu.jpeg', 0)
edges = cv2.Canny(img, 100, 200)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()

##################### Image Pyramids #######################
###### 读入两幅图像
A = cv2.imread('apple.jpeg')
B = cv2.imread('orange.jpeg')


###### 构建苹果和橘子的高斯金字塔
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

##### 根据高斯金字塔计算拉普拉斯金字塔
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

##### 在拉普拉斯的每一层进行图像融合
# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
    LS.append(ls)

##### 根据融合后的图像金字塔重建原始图像
# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:int(cols/2)], B[:,int(cols/2):]))


cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)

##################### Contours in OpenCV #######################
####### 1 - Contours
im = cv2.imread('nongyu1.jpeg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

## To draw all the contours in an image:
cv2.drawContours(im2, contours, -1, (0,255,0), 3)

## To draw an individual contour, say 4th contour:
cv2.drawContours(im2, contours, 3, (0,255,0), 3)

## But most of the time, below method will be useful:
cnt = contours[4]
cv2.drawContours(im2, [cnt], 0, (0,255,0), 3)

## Last two methods are same, but when you go forward, you will see last one is more useful.

cv2.imshow('mask',im2)
cv2.waitKey(0)

####### 2 - Contour Features
'''
 Moments:       calculate some features like center of mass of the object, area of the object etc
 Contour Area: 
 Contour Perimeter:  It is also called arc length.
 Contour Approximation:  It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify.
 Convex Hull: 
 Checking Convexity: check if a curve is convex or not,It just return whether True or False. Not a big deal.
 Bounding Rectangle: a) Straight Bounding Rectangle, b) Rotated Rectangle  
'''

im = cv2.imread('nongyu1.jpeg', 0)
ret, thresh = cv2.threshold(im, 127, 255, 0)
img,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt,True)

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

hull = cv2.convexHull(cnt)
k = cv2.isContourConvex(cnt)

print(M)
# print(cx)
# print(cy)
print(area)
print(perimeter)
print(approx)
print(hull)
print(k)

############## Bounding Rectangle ############
## Straight Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

## Rotated Rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)

########### Minimum Enclosing Circle ############
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img,center,radius,(0,255,0),2)

############### Fitting an Ellipse ###################
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img,ellipse,(0,255,0),2)

############### Fitting a Line ###################
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

####### 3 - Contour Features
im = cv2.imread('nongyu1.jpeg', 0)
ret, thresh = cv2.threshold(im, 127, 255, 0)
img,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]

### Aspect Ratio : It is the ratio of width to height of bounding rect of the object.
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h

## Extent: Extent is the ratio of contour area to bounding rectangle area.

area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area

## Solidity: Solidity is the ratio of contour area to its convex hull area.
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area

## Equivalent Diameter: Equivalent Diameter is the diameter of the circle whose area is same as the contour area.
area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)

## Orientation: Orientation is the angle at which object is directed. Following method also gives the Major Axis and Minor Axis lengths.
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

## Mask and Pixel Points: In some cases, we may need all the points which comprises that object. It can be done as follows:
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv2.findNonZero(mask)

## Maximum Value, Minimum Value and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)

## Mean Color or Mean Intensity
mean_val = cv2.mean(im,mask = mask)

## Extreme Points: Extreme Points means topmost, bottommost, rightmost and leftmost points of the object.
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

###### 4 - More Functions
## Convexity Defects
img = cv2.imread('nongyu1.jpeg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray, 127, 255,0)
im2,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
print(defects)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


## Point Polygon Test
dist = cv2.pointPolygonTest(cnt,(50,50),True)

## Match Shapes
'''
OpenCV comes with a function cv2.matchShapes() which enables us to compare two shapes, or two contours and returns a metric showing the similarity. 
The lower the result, the better match it is. It is calculated based on the hu-moment values. 
Different measurement methods are explained in the docs.
'''
img1 = cv2.imread('star.jpg',0)
img2 = cv2.imread('star2.jpg',0)
ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
im2,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
im2,contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]
ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print(ret)



##################### Histograms in OpenCV #######################














##################### Image Transforms in OpenCV #######################

##################### Template Matching #######################

##################### Hough Line Transform #######################

##################### Hough Circle Transform #######################

##################### Image Segmentation with Watershed Algorithm #######################

##################### Hough Circle Transform #######################


##################### Interactive Foreground Extraction using GrabCut Algorithm #######################
