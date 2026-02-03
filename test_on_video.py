import cv2
import numpy as np
import pickle
import sys

# --- CONFIGURATION ---
VIDEO_PATH = 'road_test.mp4'  # <--- REPLACE with your video filename
OUTPUT_FILENAME = 'sprint1_result.mp4'

# 1. Load the Pipeline (Calibration + Homography)
try:
    with open("geometry_pipeline_video.pkl", "rb") as f:
        data = pickle.load(f)
        mtx = data["camera_matrix"]
        dist = data["dist_coeff"]
        H = data["homography_matrix"]
    print("Loaded geometry pipeline.")
except FileNotFoundError:
    print("Error: 'geometry_pipeline.pkl' not found. Finish the setup step first.")
    sys.exit()

# 2. Open Video Source
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    sys.exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3. Setup Video Writer (to save the result)
# We set the output map size to 1000x1000 (adjust if you want more view)
map_w, map_h = 1000, 1500
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (map_w, map_h))

print(f"Processing {total_frames} frames... Press 'q' to quit early.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video

    # A. Undistort (Fix Lens Curvature)
    # Optimization: We usually calculate newcameramtx once outside the loop, 
    # but for simplicity, we do it here or use the standard undistort.
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

    # B. Warp Perspective (Bird's-Eye View)
    # Note: We use the same map_w, map_h as the video writer
    warped = cv2.warpPerspective(undistorted, H, (map_w, map_h))

    # C. Verification Overlays
    # Draw the 10cm scale line for proof
    cv2.line(warped, (50, 50), (150, 50), (0, 0, 255), 4)
    cv2.putText(warped, "10 cm", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Write frame count
    cv2.putText(warped, f"Frame: {frame_count}", (50, map_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save to file
    out.write(warped)

    # Optional: Display in window (resize to fit screen)
    display_view = cv2.resize(warped, (500, 750)) 
    cv2.imshow('Sprint 1: Top-Down View', display_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"   Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nDone! Result saved as '{OUTPUT_FILENAME}'")
print("   - Check that road lines remain PARALLEL.")
print("   - Check that objects don't change size as they move down the screen.")