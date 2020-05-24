import cv2
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


# Load the classifier
model = keras.models.load_model('model.h5')

# Read the input image
im = cv2.imread("photo.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

'''
For each rectangular region, calculate HOG features and predict
the digit using Linear SVM.
'''

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

     # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    if pt1<0:
        pt1=0
    if pt2<0:
        pt2=0
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    img_array = img_to_array(roi)
    #actual prediction
    nbr = model.predict(img_array.reshape(1,28,28))
    cv2.putText(im, str(int(np.argmax(nbr))), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.imwrite('predicted_results.jpg',im)
cv2.waitKey(0)