import os
import cv2
import argparse
import numpy as np
import imutils

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

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

# STEP 2: Find the document contour using external contours + convex hull

# Strengthen and connect edges first
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

dilated = cv2.dilate(edged, kernel, iterations=2)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# Find only external contours (ignore internal tables, text, etc.)
contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

if len(contours) == 0:
    print("No contours found!")
    cv2.imshow("Closed", closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

image_area = image.shape[0] * image.shape[1]

# Keep only large contours (parts of the page border)
big_contours = [c for c in contours if cv2.contourArea(c) > 0.1 * image_area]

if len(big_contours) == 0:
    # fallback: just use the largest contour
    big_contours = [max(contours, key=cv2.contourArea)]

# Merge all points and compute convex hull
all_points = np.vstack(big_contours)  # shape (N, 1, 2)
all_points = all_points.reshape(-1, 2)  # shape (N, 2)

hull = cv2.convexHull(all_points)

# Approximate the hull to a polygon
peri = cv2.arcLength(hull, True)
approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

# If we didn't get 4 points, be a bit more aggressive
if len(approx) != 4:
    approx = cv2.approxPolyDP(hull, 0.05 * peri, True)

screenCnt = approx

print("STEP 2: Document contour detected (convex hull over external contours)")
outline = image.copy()
cv2.drawContours(outline, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", outline)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Apply a perspective transform to obtain the top-down view of the document.

# screenCnt is the contour found in STEP 2, on the resized image
pts = screenCnt.reshape(-1, 2).astype("float32")

# Ensure we have exactly 4 points
if pts.shape[0] != 4:
    # Try: convex hull + approx to 4 points
    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2).astype("float32")

    if hull_pts.shape[0] >= 4:
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if approx.shape[0] == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            # Fallback: min-area rectangle (always 4 points)
            rect = cv2.minAreaRect(hull_pts)
            box = cv2.boxPoints(rect)
            pts = box.astype("float32")
    else:
        # Fallback if hull is too small
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        pts = box.astype("float32")

# Map points from resized image back to original image coordinates
pts_orig = pts * ratio  # ratio = orig.shape[0] / 500.0 from STEP 1

# Get top-down view of the original image
warped = four_point_transform(orig, pts_orig)

# Convert to grayscale
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# "Black and white paper" effect – OpenCV version of local threshold
warped_bw = cv2.adaptiveThreshold(
    warped_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # block size
    10   # offset C
)

print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped_bw, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()