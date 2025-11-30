import os
import cv2
import argparse
import numpy as np
import imutils

# python scanner.py --image ./images/test_document2.jpg

# Argument parser to get the path of the image from command line
ap  = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned") # Path to the input image
args = vars(ap.parse_args())

# STEP 1: Read the image from disk and perform preprocessing like

# loading the image and resizing it.
image = cv2.imread(args["image"])

# Debuggin information
print("Image Path:", repr(args["image"]))
print("Exists on disk?:", os.path.exists(args["image"]))
if image is None:
    print("Error: Could not load image. Please check the path provided.")
    exit()

ratio = image.shape[0] / 500.0 # We keep track of the original image ratio to perform the scan on the original image instead of the resized one.
orig = image.copy() # Copy of the original image.
image = imutils.resize(image, height = 500) # Resize the image to a height of 500 pixels to speed up processing.

# grayscale conversion, to remove noise and find edges in the image.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image into a gray color
gray = cv2.GaussianBlur(gray, (5, 5), 0) # Configuration to the blur to easily find the edges of the image 
edged = cv2.Canny(gray, 75, 200) # Define the edges of the blurred image 

print("STEP 1: Edge Detection")
cv2.imshow("Image", image) # Show the original image
cv2.imshow("Edged", edged) # Show the edged image
cv2.waitKey(0)
cv2.destroyAllWindows()

# STEP 2: Find the contours in the edged image and keep the largest ones
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Countour detection, retrieves all of the contours from the edged image
contours = imutils.grab_contours(contours) # Grab the contours from the detection
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10] # Sort the contours based on their area keeping only the largest ones

# Contour seletion via looping over the contours
screenCnt = None
max_area = 0
image_area = image.shape[0] * image.shape[1]
for c in contours:
    peri = cv2.arcLength(c, True) # Calculate the perimeter of the contour
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) # Approximate the contour

    # If our approximated contour has four points, then we can assume that we have found our screen
    if len(approx) == 4:
        area = cv2.contourArea(approx)
        if area < 0.2 * image_area:  # Skip tiny rectangles (tables, logos, etc.)
            continue
        if area > max_area:
            screenCnt = approx
            max_area = area
        if area >= 0.5 * image_area:  # Early exit once the contour is reasonably big
            break

# If no contour with four points is found, use the largest contour
if screenCnt is None:
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    screenCnt = cv2.boxPoints(rect)
    screenCnt = np.asarray(screenCnt, dtype=int)
    
# show the outlines of the contours 
# print("STEP 2: Find contours of paper")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
