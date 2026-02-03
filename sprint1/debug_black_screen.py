import cv2
import numpy as np
import pickle
import sys

# --- CONFIGURATION ---
VIDEO_PATH = 'road_test.mp4'  # Ensure this matches your file name

# 1. Load Pipeline
try:
    with open("geometry_pipeline_video.pkl", "rb") as f:
        data = pickle.load(f)
        mtx = data["camera_matrix"]
        dist = data["dist_coeff"]
        H = data["homography_matrix"]
except FileNotFoundError:
    print("Error: Pipeline not found.")
    sys.exit()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"CRITICAL ERROR: Could not open '{VIDEO_PATH}'.")
    print("   -> Check the filename exactly.")
    print("   -> Try moving the video to the same folder as this script.")
    sys.exit()

print("Video file opened. Starting playback...")
print("   - LEFT WINDOW: Undistorted (Should look like normal video)")
print("   - RIGHT WINDOW: Warped (The Bird's Eye View)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break

    # 1. Undistort
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
    
    # 2. Warp
    # We use a large canvas to try and 'catch' the image if it's off-center
    map_h, map_w = 2000, 1500 
    warped = cv2.warpPerspective(undistorted, H, (map_w, map_h))

    # 3. DEBUG: Draw an 'X' on the warped image center to prove the window is working
    cv2.line(warped, (0,0), (map_w, map_h), (50, 50, 50), 2)
    cv2.line(warped, (map_w, 0), (0, map_h), (50, 50, 50), 2)

    # 4. Display Side-by-Side (Resize for screen)
    view_h = 600
    aspect_ratio = undistorted.shape[1] / undistorted.shape[0]
    view_w = int(view_h * aspect_ratio)
    
    show_raw = cv2.resize(undistorted, (view_w, view_h))
    show_warp = cv2.resize(warped, (view_w, view_h))

    cv2.imshow('Debug: Left=Normal, Right=Warped', np.hstack([show_raw, show_warp]))

    if cv2.waitKey(0) & 0xFF == ord('q'): # <--- NOTE: waitKey(0) pauses on first frame!
        break

cap.release()
cv2.destroyAllWindows()