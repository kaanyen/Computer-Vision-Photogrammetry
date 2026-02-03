import cv2
import numpy as np
import pickle
import sys

# --- CONFIGURATION ---
IMAGE_PATH = 'homography_setup.jpg'
CHECKERBOARD_DIMS = (9, 6)    # (Cols, Rows) - Inner corners
SQUARE_SIZE_CM = 3.86          # <--- VERIFY THIS!
PIXELS_PER_CM = 10            # Client Requirement

# --- CROP CONFIGURATION (From your previous success) ---
CROP_H_MIN = 0.4
CROP_H_MAX = 1.0
CROP_W_MIN = 0.2
CROP_W_MAX = 0.8

# 1. Load Calibration
try:
    with open("camera_calibration.pkl", "rb") as f:
        data = pickle.load(f)
        mtx = data["camera_matrix"]
        dist = data["dist_coeff"]
except FileNotFoundError:
    print("Error: 'camera_calibration.pkl' not found.")
    sys.exit()

# 2. Load & Undistort
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not read {IMAGE_PATH}")
    sys.exit()

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 3. APPLY THE CROP
y_start = int(h * CROP_H_MIN)
y_end = int(h * CROP_H_MAX)
x_start = int(w * CROP_W_MIN)
x_end = int(w * CROP_W_MAX)
cropped_img = undistorted_img[y_start:y_end, x_start:x_end]

# 4. SUPER DETECTION LOOP
print("Attempting detection...")
gray_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

found = False
final_corners = None
scale_factor = 1  # We track this to adjust points back later

# Strategy 1: Standard
print("   - Strategy 1: Standard...")
ret, corners = cv2.findChessboardCorners(gray_crop, CHECKERBOARD_DIMS, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
if ret:
    found = True
    final_corners = corners
    scale_factor = 1

# Strategy 2: Upscale (Make it bigger)
if not found:
    print("   - Strategy 2: Upscaling Image (2x)...")
    gray_resized = cv2.resize(gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    ret, corners = cv2.findChessboardCorners(gray_resized, CHECKERBOARD_DIMS, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        found = True
        final_corners = corners * 0.5 # Scale points back down
        scale_factor = 1 # Already adjusted above

# Strategy 3: Blur (Remove Asphalt Noise)
if not found:
    print("   - Strategy 3: Gaussian Blur...")
    gray_blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    ret, corners = cv2.findChessboardCorners(gray_blur, CHECKERBOARD_DIMS, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        found = True
        final_corners = corners

if not found:
    print("Error: Checkerboard STILL not found.")
    print("   Options:")
    print("   1. Is the board actually (9, 6)? Count the INNER CORNERS again.")
    print("   2. Try moving the board 1 meter closer to the car.")
    sys.exit()

print(f"Checkerboard detected! ({len(final_corners)} points)")

# 5. SHIFT POINTS BACK TO FULL IMAGE
final_corners[:, :, 0] += x_start
final_corners[:, :, 1] += y_start

# Visual Check
debug_full = undistorted_img.copy()
cv2.drawChessboardCorners(debug_full, CHECKERBOARD_DIMS, final_corners, True)
cv2.imwrite('debug_corners_full_image.jpg', debug_full)

# 6. Calculate Homography
src_pts = final_corners.reshape(-1, 2)

board_width_px = (CHECKERBOARD_DIMS[0] - 1) * SQUARE_SIZE_CM * PIXELS_PER_CM
board_height_px = (CHECKERBOARD_DIMS[1] - 1) * SQUARE_SIZE_CM * PIXELS_PER_CM

dst_pts = []
offset_x, offset_y = 600, 1500 

for i in range(CHECKERBOARD_DIMS[1]):
    for j in range(CHECKERBOARD_DIMS[0]):
        x = offset_x + (j * SQUARE_SIZE_CM * PIXELS_PER_CM)
        y = offset_y + (i * SQUARE_SIZE_CM * PIXELS_PER_CM)
        dst_pts.append([x, y])

dst_pts = np.array(dst_pts, dtype='float32')

H, status = cv2.findHomography(src_pts, dst_pts)

# 7. Generate Bird's-Eye View
map_width, map_height = 1200, 2000
warped_img = cv2.warpPerspective(undistorted_img, H, (map_width, map_height))

cv2.line(warped_img, (100, 100), (200, 100), (0, 0, 255), 5)
cv2.putText(warped_img, "10 cm", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite('verification_3_birdseye.jpg', warped_img)

data["homography_matrix"] = H
with open("geometry_pipeline.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nSuccess! Pipeline saved. Check 'verification_3_birdseye.jpg'")