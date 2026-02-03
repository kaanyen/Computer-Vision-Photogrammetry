import cv2
import numpy as np
import pickle
import sys

# --- CONFIGURATION ---
VIDEO_PATH = 'road_test.mp4'
OUTPUT_FILENAME = 'sprint1_demo_reel.mp4'
# Scale output to reduce file size (1.0 = full 3000x1080; 0.5 = 1500x540)
OUTPUT_SCALE = 0.5

# Map Configuration (The High-Res Math)
MAP_W_REAL = 4000  # Actual math width
MAP_H_REAL = 4000  # Actual math height
SHIFT_X = 1500     # Adjust this to center your road (same as previous script)

# 1. Load Pipeline
try:
    with open("geometry_pipeline_video.pkl", "rb") as f:
        data = pickle.load(f)
        mtx = data["camera_matrix"]
        dist = data["dist_coeff"]
        H = data["homography_matrix"]
    print("Loaded geometry pipeline.")
except FileNotFoundError:
    print("Error: 'geometry_pipeline_video.pkl' not found.")
    sys.exit()

# 2. Setup Video & Matrices
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create Translation Matrix for the Map
Translation = np.array([
    [1, 0, SHIFT_X],
    [0, 1, 0],
    [0, 0, 1]
])
H_final = np.matmul(Translation, H)

# 3. Calculate Dimensions for the Side-by-Side
# We want the Map to match the Video Height (e.g., 1080p)
target_h = video_h

# Calculate aspect ratio of the map to find new width
aspect_ratio_map = MAP_W_REAL / MAP_H_REAL
display_map_w = int(target_h * aspect_ratio_map)

# Total canvas size (full res)
total_w = video_w + display_map_w
total_h = video_h
# Scaled size for smaller file
output_w = int(total_w * OUTPUT_SCALE)
output_h = int(total_h * OUTPUT_SCALE)

print(f"Output Resolution: {output_w}x{output_h} (scale {OUTPUT_SCALE})")

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (output_w, output_h))

print("Processing... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # A. The Left Image (Raw Video)
    # We can use the raw frame, or the undistorted one. Undistorted is more 'honest'.
    left_view = cv2.undistort(frame, mtx, dist, None, mtx)

    # B. The Right Image (The Map)
    # 1. Warp to full high-res physics canvas first
    warped_full = cv2.warpPerspective(left_view, H_final, (MAP_W_REAL, MAP_H_REAL))
    
    # 2. Add overlays to the high-res map
    cv2.putText(warped_full, "10 px = 1 cm", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    cv2.line(warped_full, (50, 100), (150, 100), (0, 0, 255), 10)

    # 3. Resize to match video height
    right_view = cv2.resize(warped_full, (display_map_w, target_h))

    # C. Stitch Them Together
    # np.hstack stacks arrays horizontally
    combined = np.hstack((left_view, right_view))

    # D. Add Separator Line (Optional styling)
    cv2.line(combined, (video_w, 0), (video_w, total_h), (255, 255, 255), 4)

    # Save at scaled size for smaller file
    combined_small = cv2.resize(combined, (output_w, output_h))
    out.write(combined_small)
    
    # Show a smaller preview on your screen
    preview = cv2.resize(combined, (int(total_w/2), int(total_h/2)))
    cv2.imshow('Sprint 1 Demo Reel', preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved Demo Reel to {OUTPUT_FILENAME}")