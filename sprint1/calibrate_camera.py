import numpy as np
import cv2
import glob
import pickle


import os
import glob

# 1. Check where we are running
current_dir = os.getcwd()
print(f"Current Working Directory: {current_dir}")

# 2. Check if folder exists
folder_path = os.path.join(current_dir, 'calibration_images')
if os.path.exists(folder_path):
    print(f"Folder 'calibration_images' FOUND at: {folder_path}")
    
    # 3. Check for contents
    files = os.listdir(folder_path)
    print(f"   Contents: {files[:5]} ... (showing first 5)")
    
    # 4. Check specifically for .jpg
    jpgs = glob.glob('calibration_images/*.jpg')
    print(f"   .jpg files found by glob: {len(jpgs)}")
    
    if len(jpgs) == 0:
        print("ERROR: Folder exists but contains no '.jpg' files.")
        print("   -> Check if your files are .jpeg, .png, or have capital .JPG extensions.")
else:
    print(f"ERROR: Folder 'calibration_images' NOT FOUND in {current_dir}")
    print("   -> Did you create the folder? Is the name exact?")

# --- CONFIGURATION ---
# Define the number of INNER corners in your checkerboard
# (e.g., if the board is 8x6 squares, the inner corners are 7x5)
CHECKERBOARD_DIMS = (9, 6) 

# Size of one square in real units (e.g., 30mm or 3cm)
# This is less critical for undistortion, but good practice.
SQUARE_SIZE = 30 

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# This defines the "ideal" flat board structure.
objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Load images
images = glob.glob('calibration_images/*.jpg') # Ensure format matches your phone's output

print(f"Found {len(images)} images. Starting processing...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # ret is a boolean: True if corners are found
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"Corners found in {fname}")
        objpoints.append(objp)
        
        # Increases accuracy by finding sub-pixel corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        
        # Optional: Draw and display the corners to verify
        # cv2.drawChessboardCorners(img, CHECKERBOARD_DIMS, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
    else:
        print(f"Warning: Could not find corners in {fname}")

cv2.destroyAllWindows()

# --- CALIBRATION ---
print("Calibrating camera... (this may take a moment)")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# --- OUTPUT RESULTS ---
print("\n calibration successful!")
print("\nCamera Matrix (K):\n", mtx)
print("\nDistortion Coefficients (D):\n", dist)

# --- SAVE DATA FOR SPRINT 1 USE ---
# You need these values for the next step (Warp Perspective)
data = {
    "camera_matrix": mtx,
    "dist_coeff": dist
}

with open("camera_calibration.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nCalibration data saved to 'camera_calibration.pkl'")




# --- VERIFICATION BLOCK ---

# 1. VISUALIZE DETECTED CORNERS
# Take the last processed image
img_with_corners = cv2.imread(images[0])
gray_corners = cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray_corners, CHECKERBOARD_DIMS, None)

if ret:
    cv2.drawChessboardCorners(img_with_corners, CHECKERBOARD_DIMS, corners, ret)
    cv2.imwrite('verification_1_corners_found.jpg', img_with_corners)
    print("\n[Check 1] Saved 'verification_1_corners_found.jpg'. Open this to see if corners are mapped correctly.")

# 2. VISUALIZE UNDISTORTION (The "Straight Lines" Check)
# We will take a raw image and apply the calibration matrix to it.
raw_img = cv2.imread(images[0])
h,  w = raw_img.shape[:2]

# Get the optimal new camera matrix (removes black edges if necessary)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
undistorted_img = cv2.undistort(raw_img, mtx, dist, None, newcameramtx)

# Crop the image (optional, if the undistortion adds black borders)
# x, y, w, h = roi
# undistorted_img = undistorted_img[y:y+h, x:x+w]

# Save the comparison
cv2.imwrite('verification_2_undistorted.jpg', undistorted_img)
print("[Check 2] Saved 'verification_2_undistorted.jpg'. Compare this with the original.")
print("   - Look at the edges of the checkerboard or straight lines in the background.")
print("   - In the undistorted image, they should be perfectly straight, not bowed.")

