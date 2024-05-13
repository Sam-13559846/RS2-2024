import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import os as os
import svgwrite
#load in classifer
face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

'''def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False
'''


img = cv.imread('5.png')
print (img.shape)
shapeX = img.shape[0]
shapeY = img.shape[1]
print (shapeX)
print (shapeY)
boundaryExpansion = shapeY*0.1 -.2
print(boundaryExpansion)
#imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
greyImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(greyImg)
plt.show()
rgreyImage= cv.cvtColor(greyImg,cv.COLOR_BGR2RGB)
plt.imshow(rgreyImage)
plt.show()
X=0
Y=0
H=0
W=0
face = face_classifier.detectMultiScale(
    greyImg, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
)
for (x, y, w, h) in face:
    #cv.rectangle(img, (x-(0.1*shapeX), y-(0.1*shapeY)), (x + w(0.1*shapeX), y + h+(0.1*shapeY)), (0, 255, 0), 4)
    cv.rectangle(img, (x-(1), y-(1)), (x + w+(1), y + h+(1)), (0, 255, 0), 4)
    #croppedImg = greyImg[x:w, y:h]
    X=x
    print (X)
    Y=y
    print (Y)
    W=w
    print (W)
    H=h
    print (H)

print(X)
imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(imgRGB)
plt.show()
#croppedImg = img[X:(X+W),Y:(Y+H)]
croppedImg = greyImg[Y-40:Y+H+40,X-40:X+W+40]
plt.imshow(croppedImg)
plt.show()

ret,th1 = cv.threshold(croppedImg,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(croppedImg,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv.THRESH_BINARY,5,1)
th3 = cv.adaptiveThreshold(croppedImg,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
 cv.THRESH_BINARY,19,9)
titles = ['Original Image detecting face', 'Global Thresholding (v = 127)',
'Adaptive Gaussian Thresholding', 'Adaptive Gaussian Thresholding']
images = [imgRGB, th1, th2, th3]
cv.imwrite('test.png', th3)
for i in range(4):
 plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
plt.show()

contours, hierarchy = cv.findContours(th3,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
'''
LENGTH = len(contours)
status = np.zeros((LENGTH,1))


for i,cnt1 in enumerate(contours):
    x = i    
    if i != LENGTH-1:
        for j,cnt2 in enumerate(contours[i+1:]):
            x = x+1
            dist = find_if_close(cnt1,cnt2)
            if dist == True:
                val = min(status[i],status[x])
                status[x] = status[i] = val
            else:
                if status[x]==status[i]:
                    status[x] = i+1


unified = []
maximum = int(status.max())+1
for i in range(maximum):
    pos = np.where(status==i)[0]
    if pos.size != 0:
        cont = np.vstack(contours[i] for i in pos)
        hull = cv.convexHull(cont)
        unified.append(hull)


cv.drawContours(img,unified,-1,(0,255,0),2)
cv.drawContours(th3,unified,-1,255,-1)
'''

biggest_contour = max(contours, key = cv.contourArea)
# Draw the contour
cv.drawContours (img, contours, -1, (222,255,0), 4)
# Display the results
plt.figure(figsize=[10,10])
plt.imshow(img);plt.axis("off")
plt.show()
## integrating Long's code##


edges = cv.Canny(th3, 100, 200)

# Find contours
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create SVG object
svg = svgwrite.Drawing(output_svg_path, profile='tiny')

# Iterate through contours
for contour in contours:
    # Approximate contour to reduce points
    epsilon = 0.01 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # Convert contour points to SVG format
    points = [(int(point[0][0]), int(point[0][1])) for point in approx]
    svg.add(svg.polyline(points, stroke="black", fill="none"))

# Set the size of the SVG drawing
svg['width'] = '640px'  # Set the width of the SVG
svg['height'] = '480px'  # Set the height of the SVG

# Save SVG file
svg.save()

# Display the processed image
self.display_trace_outline(canvas_traced_outline_image)
print("\nSVG image saved:", output_svg_path)



# def trace_outline(self, canvas_traced_outline_image):
#         # Check if the captured image exists
#         file_path = os.path.join(self.home_directory, "rs2_ws", "img", "captured_picture_rmbg.png")
#         if not os.path.exists(file_path):
#             print("Image not found.")
#             return

#         output_svg_path = os.path.join(self.home_directory, "rs2_ws", "img", "outline_picture_rmbg.svg")

#         # Read the image
#         image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

#         # Apply Canny edge detection
#         edges = cv.Canny(image, 100, 200)

#         # Find contours
#         contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#         # Create SVG object
#         svg = svgwrite.Drawing(output_svg_path, profile='tiny')

#         # Iterate through contours
#         for contour in contours:
#             # Approximate contour to reduce points
#             epsilon = 0.01 * cv.arcLength(contour, True)
#             approx = cv.approxPolyDP(contour, epsilon, True)

#             # Convert contour points to SVG format
#             points = [(int(point[0][0]), int(point[0][1])) for point in approx]
#             svg.add(svg.polyline(points, stroke="black", fill="none"))

#         # Set the size of the SVG drawing
#         svg['width'] = '640px'  # Set the width of the SVG
#         svg['height'] = '480px'  # Set the height of the SVG

#         # Save SVG file
#         svg.save()

#         # Display the processed image
#         self.display_trace_outline(canvas_traced_outline_image)
#         print("\nSVG image saved:", output_svg_path)