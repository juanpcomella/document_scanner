# 📄 Document Scanner (OpenCV)

This project is a Python-based document scanner built with OpenCV.
It detects the outline of a paper sheet in a photo (even with shadows, folds, or missing edges), extracts it using a perspective transform, and returns a clean, flat, top-down scan.

The scanner supports:

✔️ Color scanning
✔️ Robust contour detection
✔️ Edge repair using convex hull
✔️ Automatic perspective correction
✔️ CamScanner-style preprocessing (optional)

## 📦 Requirements

Install dependencies:
```
pip install opencv-python imutils numpy
```

This project uses only OpenCV and NumPy — no external ML, no skimage.

##   ▶️ Running the Scanner

In the terminal:
```
python scanner.py --image "path/to/your/photo.jpg"
```

or using the short flag:
```
python scanner.py -i "images/test1.jpg"
```

The program will display:
 - The original image
 - The scanned (top-down) version

Close the windows to exit.

## 🧠 How It Works
### 1. Preprocessing

 - Resize image for faster processing
 - Convert to grayscale
 - Blur
 - Apply Canny edge detection

### 2. Document Detection

Unlike the original PyImageSearch tutorial, this project uses an improved, robust method:
 - Morphological dilation + closing to fix broken page edges
 - Extract external contours only
 - Filter out small contours
 - Merge remaining contours into a convex hull
 - Approximate to 4 corners
 - Fallback: minimum area rectangle
This ensures the page is detected even if:
 - part of the edge is in shadow
 - the background is similar to the paper
 - multiple internal rectangles exist
 - edges are broken into many pieces

### 3. Perspective Transform

 - Using the detected 4 points:
 - Scale contour back to full-resolution
 - Warp the original image using a 4-point transform
 - Output the corrected, top-down scan
