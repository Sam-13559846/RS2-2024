import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
'''#my work before commentig everything out to test
#importing image and converting to different colour spaces
image1 = cv2.imread('5.png',0)
image1rgb= cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
image1rgbC = image.img_
#image1gray= cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#threshold
ret, thresh1 = cv2.threshold(image1rgb,205,255,cv2.THRESH_BINARY)
#ret, thresh2 = cv2.threshold(image1rgb,127,255,cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(image1rgb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv2.THRESH_BINARY,11,2)
plt.figure(figsize=[10,10])
plt.subplot(1,3,1)
plt.imshow(image1);plt.title("original image")#;plt.axis("off")
#plt.show()
plt.subplot(1,3,2)
plt.imshow(image1rgb);plt.title("image bgr to rgb")#;plt.axis("off")
#plt.show()
#plt.subplot(1,3,3)
#plt.imshow(image1gray,cmap='gray');plt.title("grey scale");plt.axis("off")
#plt.show()
#cv2. threshold(image1gray, 150, 255, cv2.THRESH_BINARY)
plt.subplot(2,3,1)
plt.imshow(thresh1);plt.title("binary");plt.axis("off")
plt.subplot(2,3,2)
plt.imshow(thresh2);plt.title("guassian threshold");plt.axis("off")
plt.show()
#end of my work before commenting out'''
img = cv.imread('3.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
#img = cv.medianBlur(img,3)
#blurring tkaes too much away from  the contoru of the chin, makes it hard to detect
#img1 = cv.medianBlur(img,33)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv.THRESH_BINARY,5,1)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv.THRESH_BINARY,9,3)
titles = ['Original Image', 'Global Thresholding (v = 127)',
'Adaptive Gaussian Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
#cv.imwrite('test.png', th3)
for i in range(4):
 plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
plt.show()