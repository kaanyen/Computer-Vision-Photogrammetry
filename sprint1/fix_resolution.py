import cv2
import numpy as np
import pickle
import sys

# --- MANUAL CONFIGURATION (Based on your data) ---
PHOTO_W, PHOTO_H = 3358, 1884   # Your Setup Photo Resolution
VIDEO_W, VIDEO_H = 1920, 1080   # Your Video Resolution

# 1. Load Original Pipeline
try:
    with open("geometry_pipeline.pkl", "rb") as f:
        data = pickle.load(f)
        K_orig = data["camera_matrix"]
        D_orig = data["dist_coeff"]
        H_orig = data["homography_matrix"]
    print("Loaded original high-res pipeline.")
except FileNotFoundError:
    print("Error: 'geometry_pipeline.pkl' not found.")
    sys.exit()

# 2. Calculate Scaling Factors
# We need to know how much smaller the video is compared to the photo
sx = PHOTO_W / VIDEO_W
sy = PHOTO_H / VIDEO_H

print(f"   Original Size: {PHOTO_W}x{PHOTO_H}")
print(f"   Target Size:   {VIDEO_W}x{VIDEO_H}")
print(f"   Scale Factor:  x={sx:.4f}, y={sy:.4f}")

# 3. Scale the Matrices

# A. Scale Camera Matrix (K)
# K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# To scale DOWN, we divide the focal lengths and centers by the scale factor.
K_new = K_orig.copy()
K_new[0, 0] /= sx  # fx
K_new[0, 2] /= sx  # cx
K_new[1, 1] /= sy  # fy
K_new[1, 2] /= sy  # cy

# B. Scale Homography Matrix (H)
# H maps Source(Pixel) -> Dest(Map).
# The video pixels are smaller. We need to multiply them by the scale factor
# to "pretend" they are the original large photo pixels, so H_orig works.
S_up = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
])

# H_new = H_orig * S_up
H_new = np.matmul(H_orig, S_up)

# 4. Save the New "Video-Ready" Pipeline
data_new = {
    "camera_matrix": K_new,
    "dist_coeff": D_orig, 
    "homography_matrix": H_new
}

with open("geometry_pipeline_video.pkl", "wb") as f:
    pickle.dump(data_new, f)

print("\nSuccess! Created 'geometry_pipeline_video.pkl'")