import cv2
import numpy as np
import pickle
import sys

# --- CONFIGURATION ---
VIDEO_PATH = 'road_test.mp4'
# Preview only: no file written. Show this many seconds then stop.
PREVIEW_SECONDS = 5

# 1. Load the Video Pipeline
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

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30 # Fallback

# 2. SETUP THE WIDE CANVAS
# A standard lane is ~3.5m (3500px). We need a canvas slightly larger.
MAP_W = 4000   # 4 meters wide
MAP_H = 4000   # 4 meters long (scrolling)

# 3. DEFINE A SHIFT (To center the road)
# If your road is cut off, change these numbers!
SHIFT_X = 1500  # Shift the road to the right by 1500 pixels
SHIFT_Y = 0     # Shift down/up

# Create a Translation Matrix to move the road into our new wide window
Translation = np.array([
    [1, 0, SHIFT_X],
    [0, 1, SHIFT_Y],
    [0, 0, 1]
])

# Combine with existing Homography
H_final = np.matmul(Translation, H)

# Preview only: no file output
out = None
max_preview_frames = int(fps * PREVIEW_SECONDS)

print(f"Preview only ({PREVIEW_SECONDS} s). Press 'q' to quit early.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count > max_preview_frames:
        break

    # A. Undistort
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

    # B. Warp to Huge Canvas
    warped = cv2.warpPerspective(undistorted, H_final, (MAP_W, MAP_H))

    # C. Add Scale Reference (The "Truth")
    # 100 px line = 10 cm
    cv2.line(warped, (100, 100), (200, 100), (0, 0, 255), 10)
    cv2.putText(warped, "10 cm (Actual Size)", (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

    # D. Save disabled (preview only)
    if out is not None:
        out.write(warped)

    # E. Display a "Mini-Map" (For You)
    # We resize the 4000px image down to 800px just so it fits on your screen
    display_view = cv2.resize(warped, (800, 800))
    cv2.imshow('Sprint 1 Final: Full Road View', display_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print("Preview finished (no file saved).")